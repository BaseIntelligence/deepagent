//! Pre-export workspace validation with agentic install-fix loop.
//!
//! Performs a complete end-to-end validation of a `SweTask` before it is
//! exported to disk. Uses an LLM agent with shell access to explore the
//! repository and produce reproducible install commands, then replays
//! everything from scratch in fresh containers to guarantee correctness.
//!
//! Validation flow:
//! 1. Prompt feasibility checks
//! 2. Start Docker sandbox, run install agent (LLM with shell + file tools)
//! 3. Verify test semantics on base commit (f2p FAIL, p2p PASS)
//! 4. Apply patch, verify (f2p PASS, p2p PASS)
//! 5. Fresh-container replay loop (up to 5 cycles):
//!    a. Brand-new container, replay recorded install_commands
//!    b. Run ALL tests
//!    c. If any step fails: agent gets another shot, new commands recorded
//!    d. Destroy and retry from scratch
//!    e. All pass on a clean container -> ACCEPT

use std::sync::Arc;

use super::docker_sandbox::DockerSandbox;
use super::sandbox_tools::{self, dispatch_tool, ToolOutput};
use super::test_generator::TestFile;
use super::SweTask;
use crate::llm::{GenerationRequest, LlmProvider, Message, ToolChoice, ToolDefinition};

/// Maximum number of fresh-container replay cycles.
const MAX_FRESH_CYCLES: usize = 5;

/// Maximum agent turns for install exploration.
const MAX_INSTALL_AGENT_TURNS: usize = 200;

/// Default model for the install agent (overridden by SWE_FORGE_INSTALL_MODEL env var).
const DEFAULT_INSTALL_MODEL: &str = "openai/gpt-4.1-mini";

const INSTALL_AGENT_SYSTEM_PROMPT: &str = r#"You are a DevOps agent. Your job is to install all dependencies for a software project in a fresh Docker container (python:3.12-slim with only git and python3 pre-installed).

You have these tools:
- `shell`: Execute shell commands. Returns stdout, stderr, exit code.
- `read_file`: Read file contents with line numbers.
- `list_dir`: List directory contents.
- `grep_files`: Search file contents with regex.
- `search_files`: Find files by glob pattern.
- `submit_install`: Submit the final working install commands.

WORKFLOW:
1. Explore the repo to determine the correct installation procedure:
   - Check README.md, CONTRIBUTING.md, Makefile, Dockerfile, docker-compose.yml
   - Check setup.py, pyproject.toml, setup.cfg, requirements.txt (Python)
   - Check package.json (JavaScript/TypeScript)
   - Check Cargo.toml (Rust), go.mod (Go), pom.xml / build.gradle (Java)
2. Install the runtime if needed:
   - Python: already available (python3)
   - Node.js: `apt-get update && apt-get install -y nodejs npm` or nodesource
   - Go: download from go.dev/dl/
   - Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y`
   - Java: `apt-get update && apt-get install -y default-jdk`
3. Install project dependencies via `shell`. If a command fails, read the error, fix it, retry.
4. Common fixes: install system packages first (build-essential, libffi-dev, etc.),
   use --break-system-packages for pip on newer systems, try different install variants.
5. Once everything works, call `submit_install` with ONLY the commands that succeeded (exit 0).
   The commands must be complete, self-contained, and reproducible from scratch.

IMPORTANT:
- Only include commands that exited with code 0 in your submission.
- Commands will be replayed in a BRAND NEW container, so they must install everything from scratch.
- Include apt-get for any system dependencies.
- Include runtime installation (Node, Go, Rust, Java) if needed.
- Do NOT include exploratory commands (ls, cat, etc.) -- only install commands."#;

/// Result of workspace validation.
#[derive(Debug, Clone)]
pub enum ValidationOutcome {
    Passed,
    Rejected { reason: String },
}

/// Pre-export workspace validator.
pub struct WorkspaceValidator {
    image_override: Option<String>,
    llm: Option<Arc<dyn LlmProvider>>,
}

impl WorkspaceValidator {
    pub fn new(image_override: Option<String>, llm: Option<Arc<dyn LlmProvider>>) -> Self {
        Self {
            image_override,
            llm,
        }
    }

    /// Run full end-to-end validation on a task.
    pub async fn validate(&self, task: &mut SweTask) -> Result<ValidationOutcome, anyhow::Error> {
        if let Some(reason) = check_prompt_feasibility(task) {
            return Ok(ValidationOutcome::Rejected { reason });
        }

        if task.fail_to_pass.is_empty() {
            return Ok(ValidationOutcome::Rejected {
                reason: "No fail_to_pass test commands".to_string(),
            });
        }

        let sandbox = match DockerSandbox::start(
            &task.repo,
            &task.base_commit,
            &task.language,
            self.image_override.as_deref(),
        )
        .await
        {
            Ok(s) => s,
            Err(e) => {
                return Ok(ValidationOutcome::Rejected {
                    reason: format!("Failed to start validation container: {e}"),
                });
            }
        };

        let result = self.run_validation(&sandbox, task).await;
        sandbox.destroy().await;

        // If the first validation passed, do fresh-container replay cycles
        if matches!(result, Ok(ValidationOutcome::Passed)) {
            return self.fresh_container_revalidation(task).await;
        }

        result
    }

    /// First-pass validation: install + test semantics in a single container.
    async fn run_validation(
        &self,
        sandbox: &DockerSandbox,
        task: &mut SweTask,
    ) -> Result<ValidationOutcome, anyhow::Error> {
        // --- Install via agent or static commands ---
        if let Some(ref llm) = self.llm {
            match self.run_install_agent(sandbox, task, llm).await {
                Ok(cmds) if !cmds.is_empty() => {
                    let combined = cmds.join(" && ");
                    task.install_config
                        .insert("install".to_string(), combined);
                    task.meta
                        .insert("install_source".to_string(), "llm-install-agent".to_string());
                    tracing::info!(task_id = %task.id, "Install agent succeeded");
                }
                Ok(_) => {
                    tracing::warn!(task_id = %task.id, "Install agent returned empty commands, using defaults");
                    // Fall through to static install below
                    if let Some(install_cmd) = task.install_config.get("install") {
                        if !install_cmd.is_empty() && !install_cmd.starts_with('#') {
                            let r = sandbox
                                .exec(&format!("cd /repo && {} 2>&1", install_cmd), 300_000)
                                .await;
                            if r.exit_code != 0 {
                                return Ok(ValidationOutcome::Rejected {
                                    reason: format!(
                                        "Install failed (exit={}): {}",
                                        r.exit_code,
                                        truncate_str(&r.stderr, 500),
                                    ),
                                });
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(task_id = %task.id, error = %e, "Install agent errored, trying static install");
                    if let Some(install_cmd) = task.install_config.get("install") {
                        if !install_cmd.is_empty() && !install_cmd.starts_with('#') {
                            let r = sandbox
                                .exec(&format!("cd /repo && {} 2>&1", install_cmd), 300_000)
                                .await;
                            if r.exit_code != 0 {
                                return Ok(ValidationOutcome::Rejected {
                                    reason: format!(
                                        "Install failed (exit={}): {}",
                                        r.exit_code,
                                        truncate_str(&r.stderr, 500),
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        } else {
            // No LLM: run static install commands
            let runtime_cmds = SweTask::runtime_install_commands(&task.install_config);
            if !runtime_cmds.is_empty() {
                let r = sandbox.exec(&format!("{} 2>&1", runtime_cmds), 300_000).await;
                if r.exit_code != 0 {
                    tracing::warn!(task_id = %task.id, "Runtime install failed (continuing)");
                }
            }
            if let Some(install_cmd) = task.install_config.get("install") {
                if !install_cmd.is_empty() && !install_cmd.starts_with('#') {
                    let r = sandbox
                        .exec(&format!("cd /repo && {} 2>&1", install_cmd), 300_000)
                        .await;
                    if r.exit_code != 0 {
                        return Ok(ValidationOutcome::Rejected {
                            reason: format!(
                                "Install failed (exit={}): {}",
                                r.exit_code,
                                truncate_str(&r.stderr, 500),
                            ),
                        });
                    }
                }
            }
        }

        // --- Copy test files ---
        if let Some(test_files_json) = task.meta.get("test_files") {
            if let Ok(files) = serde_json::from_str::<Vec<TestFile>>(test_files_json) {
                for tf in &files {
                    if let Err(e) = sandbox.write_file(&tf.path, &tf.content).await {
                        tracing::warn!(path = %tf.path, error = %e, "Failed to write test file");
                    }
                }
            }
        }

        // --- Base commit: fail_to_pass must FAIL ---
        for cmd in &task.fail_to_pass {
            let r = sandbox.exec(&format!("cd /repo && {}", cmd), 120_000).await;
            if r.exit_code == 0 {
                return Ok(ValidationOutcome::Rejected {
                    reason: format!("fail_to_pass already passes on base commit: {}", cmd),
                });
            }
        }

        // --- Base commit: pass_to_pass must PASS ---
        for cmd in &task.pass_to_pass {
            let r = sandbox.exec(&format!("cd /repo && {}", cmd), 120_000).await;
            if r.exit_code != 0 {
                return Ok(ValidationOutcome::Rejected {
                    reason: format!(
                        "pass_to_pass fails on base commit (exit={}): {}",
                        r.exit_code, cmd,
                    ),
                });
            }
        }

        // --- Apply patch ---
        if task.patch.trim().is_empty() {
            return Ok(ValidationOutcome::Rejected {
                reason: "Empty patch".to_string(),
            });
        }

        if let Err(e) = sandbox
            .write_file(".swe_forge_validation.patch", &task.patch)
            .await
        {
            return Ok(ValidationOutcome::Rejected {
                reason: format!("Failed to write patch file: {e}"),
            });
        }

        let apply = sandbox
            .exec(
                "cd /repo && git apply --allow-empty .swe_forge_validation.patch 2>&1",
                30_000,
            )
            .await;
        if apply.exit_code != 0 {
            let apply_3way = sandbox
                .exec(
                    "cd /repo && git apply --3way .swe_forge_validation.patch 2>&1",
                    30_000,
                )
                .await;
            if apply_3way.exit_code != 0 {
                return Ok(ValidationOutcome::Rejected {
                    reason: format!(
                        "Patch could not be applied: {}",
                        truncate_str(&apply_3way.stderr, 500),
                    ),
                });
            }
        }

        // Re-write test files (patch may have clobbered them)
        if let Some(test_files_json) = task.meta.get("test_files") {
            if let Ok(files) = serde_json::from_str::<Vec<TestFile>>(test_files_json) {
                for tf in &files {
                    let _ = sandbox.write_file(&tf.path, &tf.content).await;
                }
            }
        }

        // --- Patched commit: fail_to_pass must PASS ---
        for cmd in &task.fail_to_pass {
            let r = sandbox.exec(&format!("cd /repo && {}", cmd), 120_000).await;
            if r.exit_code != 0 {
                return Ok(ValidationOutcome::Rejected {
                    reason: format!(
                        "fail_to_pass still fails after patch (exit={}): {}",
                        r.exit_code, cmd,
                    ),
                });
            }
        }

        // --- Patched commit: pass_to_pass must still PASS ---
        for cmd in &task.pass_to_pass {
            let r = sandbox.exec(&format!("cd /repo && {}", cmd), 120_000).await;
            if r.exit_code != 0 {
                return Ok(ValidationOutcome::Rejected {
                    reason: format!(
                        "pass_to_pass fails after patch (regression, exit={}): {}",
                        r.exit_code, cmd,
                    ),
                });
            }
        }

        tracing::info!(task_id = %task.id, "Workspace validation PASSED (initial)");
        Ok(ValidationOutcome::Passed)
    }

    // ── Install agent ─────────────────────────────────────────────────────

    /// Run an LLM agent that explores the repo and produces working install commands.
    async fn run_install_agent(
        &self,
        sandbox: &DockerSandbox,
        task: &SweTask,
        llm: &Arc<dyn LlmProvider>,
    ) -> Result<Vec<String>, anyhow::Error> {
        let existing_install = task
            .install_config
            .get("install")
            .cloned()
            .unwrap_or_default();

        let user_msg = format!(
            "Repository: {repo}\n\
             Language: {lang}\n\
             Existing install command (may not work): {install}\n\n\
             The repo is cloned at /repo on the base commit. \
             Explore it, install all dependencies, then submit the working commands.",
            repo = task.repo,
            lang = task.language,
            install = if existing_install.is_empty() {
                "(none)"
            } else {
                &existing_install
            },
        );

        let submit_tool = ToolDefinition::function(
            "submit_install",
            "Submit the final working install commands. Only include commands that exited with code 0.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "install_commands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Shell commands that install all dependencies. Must be complete and reproducible from scratch in a fresh container."
                    }
                },
                "required": ["install_commands"]
            }),
        );

        let tools = vec![
            sandbox_tools::shell_tool(),
            sandbox_tools::read_file_tool(),
            sandbox_tools::list_dir_tool(),
            sandbox_tools::grep_files_tool(),
            sandbox_tools::search_files_tool(),
            submit_tool,
        ];

        let mut messages = vec![
            Message::system(INSTALL_AGENT_SYSTEM_PROMPT),
            Message::user(user_msg),
        ];

        for turn in 0..MAX_INSTALL_AGENT_TURNS {
            let request = GenerationRequest {
                model: install_model(),
                messages: messages.clone(),
                temperature: Some(0.2),
                max_tokens: Some(2000),
                top_p: None,
                response_format: None,
                tools: Some(tools.clone()),
                tool_choice: Some(ToolChoice::Mode("auto".to_string())),
            };

            let response = llm.generate(request).await?;
            let choice = match response.choices.first() {
                Some(c) => c.clone(),
                None => break,
            };

            if let Some(ref tool_calls) = choice.message.tool_calls {
                messages.push(Message::assistant_with_tool_calls(
                    choice.message.content.clone(),
                    tool_calls.clone(),
                ));

                for tc in tool_calls {
                    if tc.function.name == "submit_install" {
                        #[derive(serde::Deserialize)]
                        struct SubmitInstallArgs {
                            #[serde(default)]
                            install_commands: Vec<String>,
                        }
                        match serde_json::from_str::<SubmitInstallArgs>(&tc.function.arguments) {
                            Ok(args) => {
                                if args.install_commands.is_empty() {
                                    messages.push(Message::tool_result(
                                        &tc.id,
                                        "REJECTED: install_commands must not be empty. \
                                         Run install commands via shell first, verify they succeed, \
                                         then include them here.".to_string(),
                                    ));
                                    continue;
                                }
                                tracing::info!(
                                    task_id = %task.id,
                                    turn = turn,
                                    cmds = args.install_commands.len(),
                                    "Install agent submitted commands"
                                );
                                return Ok(args.install_commands);
                            }
                            Err(e) => {
                                messages.push(Message::tool_result(
                                    &tc.id,
                                    format!("Invalid submit_install args: {}", e),
                                ));
                            }
                        }
                    } else {
                        // Delegate to shared tool dispatch
                        let output = dispatch_tool(tc, sandbox, &task.id, turn).await;
                        let text = match output {
                            ToolOutput::Text(s) => s,
                            ToolOutput::Error(s) => s,
                        };
                        messages.push(Message::tool_result(&tc.id, text));
                    }
                }
                continue;
            }

            // No tool calls -- nudge the agent
            if !choice.message.content.trim().is_empty() {
                messages.push(Message::assistant(choice.message.content.clone()));
                messages.push(Message::user(
                    "Use the `shell` tool to explore the repo and install dependencies, \
                     then call `submit_install`.",
                ));
                continue;
            }

            break;
        }

        anyhow::bail!(
            "Install agent failed for {}: exhausted {} turns",
            task.id,
            MAX_INSTALL_AGENT_TURNS
        )
    }

    // ── Fresh-container replay ────────────────────────────────────────────

    /// Replay install + tests in brand-new containers, up to MAX_FRESH_CYCLES times.
    /// If install or tests fail, the agent gets another shot to fix the commands.
    async fn fresh_container_revalidation(
        &self,
        task: &mut SweTask,
    ) -> Result<ValidationOutcome, anyhow::Error> {
        for cycle in 0..MAX_FRESH_CYCLES {
            tracing::info!(
                task_id = %task.id, cycle = cycle + 1, max = MAX_FRESH_CYCLES,
                "Starting fresh-container replay cycle"
            );

            let sandbox = match DockerSandbox::start(
                &task.repo,
                &task.base_commit,
                &task.language,
                self.image_override.as_deref(),
            )
            .await
            {
                Ok(s) => s,
                Err(e) => {
                    return Ok(ValidationOutcome::Rejected {
                        reason: format!("Fresh replay cycle {}: container start failed: {e}", cycle + 1),
                    });
                }
            };

            // Replay install commands from scratch
            let install_ok = self.replay_install(&sandbox, task).await;
            if !install_ok {
                // Agent gets another shot to fix in this (still running) container
                if let Some(ref llm) = self.llm {
                    tracing::warn!(
                        task_id = %task.id, cycle = cycle + 1,
                        "Install replay failed, running agent to fix"
                    );
                    match self.run_install_agent(&sandbox, task, llm).await {
                        Ok(new_cmds) if !new_cmds.is_empty() => {
                            let combined = new_cmds.join(" && ");
                            task.install_config
                                .insert("install".to_string(), combined);
                            task.meta
                                .insert("install_source".to_string(), "llm-install-agent-fix".to_string());
                            sandbox.destroy().await;
                            continue; // retry with new commands
                        }
                        _ => {
                            sandbox.destroy().await;
                            continue;
                        }
                    }
                }
                sandbox.destroy().await;
                if cycle + 1 == MAX_FRESH_CYCLES {
                    return Ok(ValidationOutcome::Rejected {
                        reason: format!(
                            "Install failed after {} fresh-container cycles",
                            MAX_FRESH_CYCLES,
                        ),
                    });
                }
                continue;
            }

            // Copy test files
            if let Some(test_files_json) = task.meta.get("test_files") {
                if let Ok(files) = serde_json::from_str::<Vec<TestFile>>(test_files_json) {
                    for tf in &files {
                        let _ = sandbox.write_file(&tf.path, &tf.content).await;
                    }
                }
            }

            // Run all tests
            let tests_ok = self.run_all_tests(&sandbox, task).await;
            sandbox.destroy().await;

            if tests_ok {
                tracing::info!(
                    task_id = %task.id, cycle = cycle + 1,
                    "Workspace validation PASSED (fresh replay)"
                );
                return Ok(ValidationOutcome::Passed);
            }

            tracing::warn!(
                task_id = %task.id, cycle = cycle + 1,
                "Tests failed on fresh container, retrying"
            );
        }

        Ok(ValidationOutcome::Rejected {
            reason: format!(
                "Failed after {} fresh-container cycles",
                MAX_FRESH_CYCLES,
            ),
        })
    }

    /// Replay the recorded install commands on a fresh container.
    async fn replay_install(&self, sandbox: &DockerSandbox, task: &SweTask) -> bool {
        // Runtime install
        let runtime_cmds = SweTask::runtime_install_commands(&task.install_config);
        if !runtime_cmds.is_empty() {
            let r = sandbox.exec(&format!("{} 2>&1", runtime_cmds), 300_000).await;
            if r.exit_code != 0 {
                tracing::warn!(
                    task_id = %task.id,
                    exit = r.exit_code,
                    "Runtime install failed during replay"
                );
                return false;
            }
        }

        // Project install
        if let Some(install_cmd) = task.install_config.get("install") {
            if !install_cmd.is_empty() && !install_cmd.starts_with('#') {
                let r = sandbox
                    .exec(&format!("cd /repo && {} 2>&1", install_cmd), 300_000)
                    .await;
                if r.exit_code != 0 {
                    tracing::warn!(
                        task_id = %task.id,
                        exit = r.exit_code,
                        stderr = %truncate_str(&r.stderr, 300),
                        "Install replay failed"
                    );
                    return false;
                }
            }
        }

        true
    }

    /// Run f2p (must FAIL) and p2p (must PASS) tests on base commit,
    /// then apply patch and verify f2p PASS and p2p PASS.
    async fn run_all_tests(&self, sandbox: &DockerSandbox, task: &SweTask) -> bool {
        // Base commit: f2p must FAIL
        for cmd in &task.fail_to_pass {
            let r = sandbox.exec(&format!("cd /repo && {}", cmd), 120_000).await;
            if r.exit_code == 0 {
                tracing::warn!(
                    task_id = %task.id,
                    cmd = %cmd,
                    "Fresh replay: f2p already passes on base"
                );
                return false;
            }
        }

        // Base commit: p2p must PASS
        for cmd in &task.pass_to_pass {
            let r = sandbox.exec(&format!("cd /repo && {}", cmd), 120_000).await;
            if r.exit_code != 0 {
                tracing::warn!(
                    task_id = %task.id,
                    cmd = %cmd,
                    exit = r.exit_code,
                    "Fresh replay: p2p fails on base"
                );
                return false;
            }
        }

        // Apply patch
        if let Err(e) = sandbox
            .write_file(".swe_forge_validation.patch", &task.patch)
            .await
        {
            tracing::warn!(task_id = %task.id, error = %e, "Fresh replay: failed to write patch");
            return false;
        }

        let apply = sandbox
            .exec(
                "cd /repo && git apply --allow-empty .swe_forge_validation.patch 2>&1",
                30_000,
            )
            .await;
        if apply.exit_code != 0 {
            let apply_3way = sandbox
                .exec(
                    "cd /repo && git apply --3way .swe_forge_validation.patch 2>&1",
                    30_000,
                )
                .await;
            if apply_3way.exit_code != 0 {
                tracing::warn!(task_id = %task.id, "Fresh replay: patch apply failed");
                return false;
            }
        }

        // Re-write test files
        if let Some(test_files_json) = task.meta.get("test_files") {
            if let Ok(files) = serde_json::from_str::<Vec<TestFile>>(test_files_json) {
                for tf in &files {
                    let _ = sandbox.write_file(&tf.path, &tf.content).await;
                }
            }
        }

        // Patched: f2p must PASS
        for cmd in &task.fail_to_pass {
            let r = sandbox.exec(&format!("cd /repo && {}", cmd), 120_000).await;
            if r.exit_code != 0 {
                tracing::warn!(
                    task_id = %task.id,
                    cmd = %cmd,
                    exit = r.exit_code,
                    "Fresh replay: f2p still fails after patch"
                );
                return false;
            }
        }

        // Patched: p2p must still PASS
        for cmd in &task.pass_to_pass {
            let r = sandbox.exec(&format!("cd /repo && {}", cmd), 120_000).await;
            if r.exit_code != 0 {
                tracing::warn!(
                    task_id = %task.id,
                    cmd = %cmd,
                    exit = r.exit_code,
                    "Fresh replay: p2p fails after patch"
                );
                return false;
            }
        }

        true
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn install_model() -> String {
    std::env::var("SWE_FORGE_INSTALL_MODEL").unwrap_or_else(|_| DEFAULT_INSTALL_MODEL.to_string())
}

/// Check prompt feasibility without Docker.
pub fn check_prompt_feasibility(task: &SweTask) -> Option<String> {
    if task.prompt.trim().is_empty() {
        return Some("Prompt is empty".to_string());
    }

    if task.prompt.trim().len() < 100 {
        return Some(format!(
            "Prompt too short ({} chars, minimum 100)",
            task.prompt.trim().len(),
        ));
    }

    let prompt_lower = task.prompt.to_lowercase();
    for cmd in &task.fail_to_pass {
        if prompt_lower.contains(&cmd.to_lowercase()) {
            return Some(format!(
                "Prompt contains fail_to_pass command: {}",
                truncate_str(cmd, 100),
            ));
        }
    }

    if let Some(test_files_json) = task.meta.get("test_files") {
        if let Ok(files) = serde_json::from_str::<Vec<TestFile>>(test_files_json) {
            for tf in &files {
                let basename = std::path::Path::new(&tf.path)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                if !basename.is_empty() && prompt_lower.contains(&basename.to_lowercase()) {
                    return Some(format!("Prompt contains test file name: {}", basename));
                }
            }
        }
    }

    None
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max;
        while !s.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_feasibility_empty() {
        let mut task = SweTask::new("test-1", "owner/repo");
        task.prompt = String::new();
        assert!(check_prompt_feasibility(&task).is_some());
    }

    #[test]
    fn prompt_feasibility_too_short() {
        let mut task = SweTask::new("test-2", "owner/repo");
        task.prompt = "Fix the bug.".to_string();
        let result = check_prompt_feasibility(&task);
        assert!(result.is_some());
        assert!(result.unwrap().contains("too short"));
    }

    #[test]
    fn prompt_feasibility_ok() {
        let mut task = SweTask::new("test-3", "owner/repo");
        task.prompt = "This is a sufficiently long prompt that describes a real software engineering problem requiring changes to multiple files and careful understanding of the codebase architecture.".to_string();
        assert!(check_prompt_feasibility(&task).is_none());
    }

    #[test]
    fn prompt_feasibility_test_leak() {
        let mut task = SweTask::new("test-4", "owner/repo");
        task.prompt = "This is a sufficiently long prompt that describes a real software engineering problem. Run python -m pytest tests/test_foo.py to verify your changes work correctly.".to_string();
        task.fail_to_pass = vec!["python -m pytest tests/test_foo.py".to_string()];
        let result = check_prompt_feasibility(&task);
        assert!(result.is_some());
        assert!(result.unwrap().contains("fail_to_pass"));
    }

    #[test]
    fn prompt_feasibility_file_name_leak() {
        let mut task = SweTask::new("test-5", "owner/repo");
        task.prompt = "This is a sufficiently long prompt that describes a real software engineering problem. Make sure test_special_feature.py passes after your changes.".to_string();
        task.meta.insert(
            "test_files".to_string(),
            serde_json::to_string(&vec![TestFile {
                path: "tests/test_special_feature.py".to_string(),
                content: "pass".to_string(),
            }])
            .unwrap(),
        );
        let result = check_prompt_feasibility(&task);
        assert!(result.is_some());
        assert!(result.unwrap().contains("test file name"));
    }

    #[test]
    fn validation_outcome_debug() {
        let passed = ValidationOutcome::Passed;
        let rejected = ValidationOutcome::Rejected {
            reason: "test".to_string(),
        };
        assert!(format!("{:?}", passed).contains("Passed"));
        assert!(format!("{:?}", rejected).contains("test"));
    }

    #[test]
    fn truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn truncate_str_long() {
        let result = truncate_str("hello world this is long", 10);
        assert!(result.len() <= 14);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn validator_new_without_llm() {
        let v = WorkspaceValidator::new(None, None);
        assert!(v.llm.is_none());
        assert!(v.image_override.is_none());
    }

    #[test]
    fn validator_new_with_image() {
        let v = WorkspaceValidator::new(Some("custom:latest".to_string()), None);
        assert_eq!(v.image_override.as_deref(), Some("custom:latest"));
    }

    #[test]
    fn install_model_default() {
        // When env var is not set, should return default
        let model = install_model();
        assert!(!model.is_empty());
    }
}
