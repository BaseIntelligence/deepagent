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

use super::docker_sandbox::{DockerSandbox, SandboxOutput};
use super::sandbox_tools::{self, dispatch_tool, ToolOutput};
use super::test_generator::TestFile;
use super::SweTask;
use crate::llm::{GenerationRequest, LlmProvider, Message, ToolChoice, ToolDefinition};

/// Maximum number of fresh-container replay cycles.
const MAX_FRESH_CYCLES: usize = 5;

/// Maximum agent turns for install exploration.
const MAX_INSTALL_AGENT_TURNS: usize = 200;

/// Maximum agent turns for verification (more focused than install agent).
const MAX_VERIFY_AGENT_TURNS: usize = 100;

/// Default model for the install agent (overridden by SWE_FORGE_INSTALL_MODEL env var).
const DEFAULT_INSTALL_MODEL: &str = "openai/gpt-5.3-codex";

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

const VERIFY_AGENT_SYSTEM_PROMPT: &str = r#"You are a verification agent. Your job is to verify and fix install + test commands for a software project in a fresh Docker container (python:3.12-slim with only git and python3 pre-installed).

You have these tools:
- `shell`: Execute shell commands freely (for exploration and debugging)
- `read_file`: Read file contents
- `list_dir`: List directory contents
- `grep_files`: Search files with regex
- `search_files`: Find files by glob pattern
- `verify_command`: Execute a command and verify it succeeds/fails as expected. Returns structured result.
- `submit_verified_install`: Submit verified install commands after all checks pass.

CONTEXT:
- Install commands to verify: {install_commands}
- Test commands (fail_to_pass, must FAIL on base): {f2p}
- Test commands (pass_to_pass, must PASS on base): {p2p}
- Test files have been written to the container.

WORKFLOW:
1. Run each install command via `verify_command` and verify they all succeed (exit 0).
   If any fails, diagnose via `shell`, fix the command, and re-run.

2. After install, verify test tools are available:
   - For Python projects: `verify_command("which pytest || which python3 -m pytest", expect_success=true)`
   - For JS projects: `verify_command("which npm && which npx", expect_success=true)`
   - For Rust: `verify_command("which cargo", expect_success=true)`
   If test tools are missing, install them (e.g. `pip install --break-system-packages pytest`) and add to install_commands.

3. Verify each fail_to_pass test:
   `verify_command(cmd, expect_success=false)` -- must FAIL (exit != 0) on base commit.
   If it fails with "command not found" instead of a test failure, the tool is missing -- fix install.

4. Verify each pass_to_pass test:
   `verify_command(cmd, expect_success=true)` -- must PASS (exit 0) on base commit.
   If it fails with "command not found", fix install.

5. Once ALL verifications pass, call `submit_verified_install` with the complete, corrected install commands.

IMPORTANT:
- "command not found" errors mean the tool is NOT installed. Fix this by adding apt-get/pip install commands.
- Install commands must be complete and self-contained for a fresh container.
- Include `--break-system-packages` for pip commands on Python 3.12+.
- Common missing tools: pytest, tox, nox, coverage, mypy, black, flake8, eslint, jest, mocha.
- When a test fails with "No module named X", add `pip install --break-system-packages X` to install commands.

CRITICAL DISTINCTION - "command not found" vs real test failure:
- A fail_to_pass test that exits non-zero because "pytest: command not found" is BROKEN, not valid.
- fail_to_pass tests must fail because test ASSERTIONS fail, not because the test runner is missing.
- After fixing installs, RE-RUN all tests and verify the output contains actual test results.
- Look for patterns like "FAILED", "AssertionError", "expected X got Y" in test output to confirm real failures.
- If you see "command not found" or "No module named" in any test output, the install is incomplete."#;

/// Result of workspace validation.
#[derive(Debug, Clone)]
pub enum ValidationOutcome {
    Passed,
    Rejected { reason: String },
}

/// Result of running tests -- distinguishes real test failures from broken environments.
#[derive(Debug, Clone)]
enum TestRunResult {
    /// All tests behaved as expected.
    Ok,
    /// A test failed for a real logic/assertion reason.
    RealFailure { cmd: String, detail: String },
    /// A test failed because the environment is broken (missing tool, module, etc.).
    EnvironmentBroken { cmd: String, diagnosis: String },
}

/// Use LLM to classify whether a test failure is a genuine assertion failure
/// or a broken environment (missing tool, missing module, etc.).
/// Returns `true` if the failure is an environment issue (broken), `false` if real test failure.
async fn llm_diagnose_failure(
    llm: &Arc<dyn LlmProvider>,
    task_id: &str,
    cmd: &str,
    output: &SandboxOutput,
) -> bool {
    let diagnose_tool = ToolDefinition::function(
        "classify_failure",
        "Classify whether this test failure is a real test assertion failure or a broken environment issue.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "is_environment_issue": {
                    "type": "boolean",
                    "description": "true if the failure is caused by missing tools, missing modules, broken PATH, or incomplete installation. false if the test ran properly but assertions/checks failed."
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of your classification."
                }
            },
            "required": ["is_environment_issue", "reasoning"]
        }),
    );

    let stdout_trunc: String = output.stdout.chars().take(2000).collect();
    let stderr_trunc: String = output.stderr.chars().take(1000).collect();

    let user_msg = format!(
        "A test command failed. Classify whether this is a real test failure or a broken environment.\n\n\
         Command: {cmd}\n\
         Exit code: {exit}\n\n\
         stdout (truncated):\n```\n{stdout}\n```\n\n\
         stderr (truncated):\n```\n{stderr}\n```\n\n\
         IMPORTANT: A \"real test failure\" means the test runner executed properly and test assertions/checks \
         failed (e.g. pytest ran and reported FAILED tests, jest ran and reported failing expects, etc.).\n\
         An \"environment issue\" means the test could not even run properly -- missing command, missing module, \
         import error, command not found, permission denied on a tool binary, etc.\n\n\
         Call classify_failure with your verdict.",
        cmd = cmd,
        exit = output.exit_code,
        stdout = stdout_trunc,
        stderr = stderr_trunc,
    );

    let request = GenerationRequest {
        model: install_model(),
        messages: vec![
            Message::system(
                "You are a test output classifier. Analyze the test output and determine whether \
                 the failure is a real test assertion failure or a broken environment issue. \
                 You MUST call the classify_failure tool with your verdict."
                    .to_string(),
            ),
            Message::user(user_msg),
        ],
        temperature: Some(0.0),
        max_tokens: Some(500),
        top_p: None,
        response_format: None,
        tools: Some(vec![diagnose_tool]),
        tool_choice: Some(ToolChoice::force("classify_failure")),
    };

    match llm.generate(request).await {
        Ok(response) => {
            if let Some(choice) = response.choices.first() {
                if let Some(ref tool_calls) = choice.message.tool_calls {
                    if let Some(tc) = tool_calls.first() {
                        #[derive(serde::Deserialize)]
                        struct ClassifyResult {
                            is_environment_issue: bool,
                            #[serde(default)]
                            reasoning: String,
                        }
                        if let Ok(result) =
                            serde_json::from_str::<ClassifyResult>(&tc.function.arguments)
                        {
                            tracing::info!(
                                task_id = task_id,
                                cmd = cmd,
                                is_env = result.is_environment_issue,
                                reasoning = %result.reasoning,
                                "LLM diagnosed test failure"
                            );
                            return result.is_environment_issue;
                        }
                    }
                }
            }
            tracing::warn!(
                task_id = task_id,
                cmd = cmd,
                "LLM diagnosis returned no usable result, assuming real failure"
            );
            false
        }
        Err(e) => {
            tracing::warn!(task_id = task_id, cmd = cmd, error = %e, "LLM diagnosis failed, assuming real failure");
            false
        }
    }
}

// ── Verification tool definitions ─────────────────────────────────────────

fn verify_command_tool() -> ToolDefinition {
    ToolDefinition::function(
        "verify_command",
        "Execute a command and verify it succeeds/fails as expected. Returns structured result with exit_code, passed, stdout, stderr.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute and verify"
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Timeout in ms (default: 120000)"
                },
                "expect_success": {
                    "type": "boolean",
                    "description": "Whether the command should succeed (exit 0) or fail (exit != 0)"
                }
            },
            "required": ["command"]
        }),
    )
}

fn submit_verified_install_tool() -> ToolDefinition {
    ToolDefinition::function(
        "submit_verified_install",
        "Submit verified install commands after all checks pass. Only call this after verifying install AND test commands.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "install_commands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Complete install commands verified in this fresh container"
                },
                "test_tools_verified": {
                    "type": "boolean",
                    "description": "Whether test runner tools (pytest, npm test, cargo test, etc.) were verified to be available after install"
                }
            },
            "required": ["install_commands", "test_tools_verified"]
        }),
    )
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
                    task.install_config.insert("install".to_string(), combined);
                    // Remove runtime version keys -- the install agent includes
                    // runtime setup in its commands already.
                    for key in &["go", "node", "rust", "java"] {
                        task.install_config.remove(*key);
                    }
                    task.meta.insert(
                        "install_source".to_string(),
                        "llm-install-agent".to_string(),
                    );
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
                let r = sandbox
                    .exec(&format!("{} 2>&1", runtime_cmds), 300_000)
                    .await;
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
    /// Uses an agentic verification approach: an LLM agent diagnoses and fixes
    /// install/test failures, then a final blind replay confirms correctness.
    async fn fresh_container_revalidation(
        &self,
        task: &mut SweTask,
    ) -> Result<ValidationOutcome, anyhow::Error> {
        for cycle in 0..MAX_FRESH_CYCLES {
            tracing::info!(
                task_id = %task.id, cycle = cycle + 1, max = MAX_FRESH_CYCLES,
                "Starting fresh-container replay cycle"
            );

            // ── Step 1: Verification agent in a fresh container ───────────
            if let Some(ref llm) = self.llm {
                let verify_sandbox = match DockerSandbox::start(
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
                            reason: format!(
                                "Fresh replay cycle {}: verify container start failed: {e}",
                                cycle + 1
                            ),
                        });
                    }
                };

                // Copy test files before handing off to the verify agent
                if let Some(test_files_json) = task.meta.get("test_files") {
                    if let Ok(files) = serde_json::from_str::<Vec<TestFile>>(test_files_json) {
                        for tf in &files {
                            let _ = verify_sandbox.write_file(&tf.path, &tf.content).await;
                        }
                    }
                }

                match self.run_verify_agent(&verify_sandbox, task, llm).await {
                    Ok(new_cmds) if !new_cmds.is_empty() => {
                        let combined = new_cmds.join(" && ");
                        task.install_config.insert("install".to_string(), combined);
                        // Remove runtime version keys so replay_install doesn't
                        // run conflicting runtime_install_commands -- the verify
                        // agent already includes runtime setup in its commands.
                        for key in &["go", "node", "rust", "java"] {
                            task.install_config.remove(*key);
                        }
                        task.meta
                            .insert("install_source".to_string(), "llm-verify-agent".to_string());
                        tracing::info!(
                            task_id = %task.id, cycle = cycle + 1,
                            "Verify agent submitted corrected install commands"
                        );
                    }
                    Ok(_) => {
                        tracing::warn!(
                            task_id = %task.id, cycle = cycle + 1,
                            "Verify agent returned empty commands, keeping existing"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            task_id = %task.id, cycle = cycle + 1,
                            error = %e,
                            "Verify agent errored, keeping existing commands"
                        );
                    }
                }

                verify_sandbox.destroy().await;
            }

            // ── Step 2a: Base-commit tests in a fresh container ─────────
            let base_sandbox = match DockerSandbox::start(
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
                        reason: format!(
                            "Fresh replay cycle {}: base container start failed: {e}",
                            cycle + 1
                        ),
                    });
                }
            };

            let base_install_ok = self.replay_install(&base_sandbox, task).await;
            if !base_install_ok {
                tracing::warn!(
                    task_id = %task.id, cycle = cycle + 1,
                    "Blind replay install failed (base container) after verify agent, retrying"
                );
                base_sandbox.destroy().await;
                continue;
            }

            // Copy test files into base container
            if let Some(test_files_json) = task.meta.get("test_files") {
                if let Ok(files) = serde_json::from_str::<Vec<TestFile>>(test_files_json) {
                    for tf in &files {
                        let _ = base_sandbox.write_file(&tf.path, &tf.content).await;
                    }
                }
            }

            let base_result = self.run_base_tests(&base_sandbox, task).await;
            base_sandbox.destroy().await;

            match base_result {
                TestRunResult::Ok => {}
                TestRunResult::EnvironmentBroken {
                    ref cmd,
                    ref diagnosis,
                } => {
                    tracing::warn!(
                        task_id = %task.id, cycle = cycle + 1,
                        cmd = %cmd, diagnosis = %diagnosis,
                        "Base tests: LLM detected broken environment, retrying with verify agent"
                    );
                    continue;
                }
                TestRunResult::RealFailure {
                    ref cmd,
                    ref detail,
                } => {
                    tracing::warn!(
                        task_id = %task.id, cycle = cycle + 1,
                        cmd = %cmd, detail = %detail,
                        "Base tests: real test failure, retrying"
                    );
                    continue;
                }
            }

            // ── Step 2b: Patched-commit tests in another fresh container ──
            let patched_sandbox = match DockerSandbox::start(
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
                        reason: format!(
                            "Fresh replay cycle {}: patched container start failed: {e}",
                            cycle + 1
                        ),
                    });
                }
            };

            let patched_install_ok = self.replay_install(&patched_sandbox, task).await;
            if !patched_install_ok {
                tracing::warn!(
                    task_id = %task.id, cycle = cycle + 1,
                    "Blind replay install failed (patched container) after verify agent, retrying"
                );
                patched_sandbox.destroy().await;
                continue;
            }

            // Copy test files into patched container
            if let Some(test_files_json) = task.meta.get("test_files") {
                if let Ok(files) = serde_json::from_str::<Vec<TestFile>>(test_files_json) {
                    for tf in &files {
                        let _ = patched_sandbox.write_file(&tf.path, &tf.content).await;
                    }
                }
            }

            let patched_result = self.run_patched_tests(&patched_sandbox, task).await;
            patched_sandbox.destroy().await;

            match patched_result {
                TestRunResult::Ok => {
                    tracing::info!(
                        task_id = %task.id, cycle = cycle + 1,
                        "Workspace validation PASSED (fresh replay confirmed with isolated containers)"
                    );
                    return Ok(ValidationOutcome::Passed);
                }
                TestRunResult::EnvironmentBroken {
                    ref cmd,
                    ref diagnosis,
                } => {
                    tracing::warn!(
                        task_id = %task.id, cycle = cycle + 1,
                        cmd = %cmd, diagnosis = %diagnosis,
                        "Patched tests: LLM detected broken environment, retrying with verify agent"
                    );
                }
                TestRunResult::RealFailure {
                    ref cmd,
                    ref detail,
                } => {
                    tracing::warn!(
                        task_id = %task.id, cycle = cycle + 1,
                        cmd = %cmd, detail = %detail,
                        "Patched tests: real test failure, retrying"
                    );
                }
            }
        }

        Ok(ValidationOutcome::Rejected {
            reason: format!("Failed after {} fresh-container cycles", MAX_FRESH_CYCLES,),
        })
    }

    // ── Verify agent ──────────────────────────────────────────────────────

    /// Run an LLM verification agent that diagnoses and fixes install + test
    /// commands in a fresh container. Returns corrected install commands.
    async fn run_verify_agent(
        &self,
        sandbox: &DockerSandbox,
        task: &mut SweTask,
        llm: &Arc<dyn LlmProvider>,
    ) -> Result<Vec<String>, anyhow::Error> {
        let existing_install = task
            .install_config
            .get("install")
            .cloned()
            .unwrap_or_default();

        // Split existing install commands for the prompt
        let install_cmds_display = if existing_install.is_empty() {
            "(none)".to_string()
        } else {
            existing_install.clone()
        };

        let f2p_display = if task.fail_to_pass.is_empty() {
            "(none)".to_string()
        } else {
            task.fail_to_pass.join("\n  ")
        };

        let p2p_display = if task.pass_to_pass.is_empty() {
            "(none)".to_string()
        } else {
            task.pass_to_pass.join("\n  ")
        };

        // Build the system prompt with task details substituted
        let system_prompt = VERIFY_AGENT_SYSTEM_PROMPT
            .replace("{install_commands}", &install_cmds_display)
            .replace("{f2p}", &f2p_display)
            .replace("{p2p}", &p2p_display);

        let user_msg = format!(
            "Repository: {repo}\n\
             Language: {lang}\n\
             The repo is cloned at /repo on the base commit. \
             Test files have been written. \
             Verify the install commands, then verify the test commands, fix any issues, \
             and submit the corrected install commands via `submit_verified_install`.",
            repo = task.repo,
            lang = task.language,
        );

        let tools = vec![
            sandbox_tools::shell_tool(),
            sandbox_tools::read_file_tool(),
            sandbox_tools::list_dir_tool(),
            sandbox_tools::grep_files_tool(),
            sandbox_tools::search_files_tool(),
            verify_command_tool(),
            submit_verified_install_tool(),
        ];

        let mut messages = vec![Message::system(system_prompt), Message::user(user_msg)];

        for turn in 0..MAX_VERIFY_AGENT_TURNS {
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
                    match tc.function.name.as_str() {
                        "verify_command" => {
                            #[derive(serde::Deserialize)]
                            struct VerifyArgs {
                                command: String,
                                #[serde(default = "default_verify_timeout")]
                                timeout_ms: u64,
                                #[serde(default = "default_expect_success")]
                                expect_success: bool,
                            }
                            fn default_verify_timeout() -> u64 {
                                120_000
                            }
                            fn default_expect_success() -> bool {
                                true
                            }

                            match serde_json::from_str::<VerifyArgs>(&tc.function.arguments) {
                                Ok(args) => {
                                    let result = sandbox
                                        .exec(
                                            &format!("cd /repo && {}", args.command),
                                            args.timeout_ms,
                                        )
                                        .await;

                                    let passed = if args.expect_success {
                                        result.exit_code == 0
                                    } else {
                                        result.exit_code != 0
                                    };

                                    tracing::debug!(
                                        task_id = %task.id, turn = turn,
                                        cmd = %args.command,
                                        exit = result.exit_code,
                                        expect_success = args.expect_success,
                                        passed = passed,
                                        "Verify agent: verify_command"
                                    );

                                    let response_text = format!(
                                        "exit_code: {}\npassed: {}\nstdout:\n{}\nstderr:\n{}",
                                        result.exit_code,
                                        passed,
                                        truncate_str(&result.stdout, 3000),
                                        truncate_str(&result.stderr, 1500),
                                    );
                                    messages.push(Message::tool_result(&tc.id, response_text));
                                }
                                Err(e) => {
                                    messages.push(Message::tool_result(
                                        &tc.id,
                                        format!("Invalid verify_command args: {}", e),
                                    ));
                                }
                            }
                        }
                        "submit_verified_install" => {
                            #[derive(serde::Deserialize)]
                            struct SubmitVerifiedArgs {
                                #[serde(default)]
                                install_commands: Vec<String>,
                                #[serde(default)]
                                test_tools_verified: bool,
                            }

                            match serde_json::from_str::<SubmitVerifiedArgs>(&tc.function.arguments)
                            {
                                Ok(args) => {
                                    if args.install_commands.is_empty() {
                                        messages.push(Message::tool_result(
                                            &tc.id,
                                            "REJECTED: install_commands must not be empty. \
                                             Verify install and test commands first, then submit."
                                                .to_string(),
                                        ));
                                        continue;
                                    }
                                    if !args.test_tools_verified {
                                        messages.push(Message::tool_result(
                                            &tc.id,
                                            "WARNING: test_tools_verified is false. \
                                             Please verify test tools are available before submitting. \
                                             If you are sure they work, set test_tools_verified=true."
                                                .to_string(),
                                        ));
                                        continue;
                                    }
                                    tracing::info!(
                                        task_id = %task.id,
                                        turn = turn,
                                        cmds = args.install_commands.len(),
                                        test_tools_verified = args.test_tools_verified,
                                        "Verify agent submitted corrected commands"
                                    );
                                    return Ok(args.install_commands);
                                }
                                Err(e) => {
                                    messages.push(Message::tool_result(
                                        &tc.id,
                                        format!("Invalid submit_verified_install args: {}", e),
                                    ));
                                }
                            }
                        }
                        _ => {
                            // Delegate to shared tool dispatch (shell, read_file, etc.)
                            let output = dispatch_tool(tc, sandbox, &task.id, turn).await;
                            let text = match output {
                                ToolOutput::Text(s) => s,
                                ToolOutput::Error(s) => s,
                            };
                            messages.push(Message::tool_result(&tc.id, text));
                        }
                    }
                }
                continue;
            }

            // No tool calls -- nudge the agent
            if !choice.message.content.trim().is_empty() {
                messages.push(Message::assistant(choice.message.content.clone()));
                messages.push(Message::user(
                    "Use `verify_command` to check each install command and test command, \
                     fix any issues with `shell`, then call `submit_verified_install`.",
                ));
                continue;
            }

            break;
        }

        anyhow::bail!(
            "Verify agent failed for {}: exhausted {} turns",
            task.id,
            MAX_VERIFY_AGENT_TURNS
        )
    }

    /// Replay the recorded install commands on a fresh container.
    async fn replay_install(&self, sandbox: &DockerSandbox, task: &SweTask) -> bool {
        // Runtime install
        let runtime_cmds = SweTask::runtime_install_commands(&task.install_config);
        if !runtime_cmds.is_empty() {
            let r = sandbox
                .exec(&format!("{} 2>&1", runtime_cmds), 300_000)
                .await;
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

    /// Run base-commit tests in an isolated container: f2p must FAIL, p2p must PASS.
    /// Uses LLM to classify ambiguous failures as real vs environment issues.
    async fn run_base_tests(&self, sandbox: &DockerSandbox, task: &SweTask) -> TestRunResult {
        // Base commit: f2p must FAIL
        for cmd in &task.fail_to_pass {
            let r = sandbox.exec(&format!("cd /repo && {}", cmd), 120_000).await;
            if r.exit_code == 0 {
                tracing::warn!(
                    task_id = %task.id,
                    cmd = %cmd,
                    "Fresh replay (base): f2p already passes on base"
                );
                return TestRunResult::RealFailure {
                    cmd: cmd.clone(),
                    detail: "f2p already passes on base commit".to_string(),
                };
            }
            // f2p failed (good) -- but ask LLM if this is a real test failure or broken env
            if let Some(ref llm) = self.llm {
                if llm_diagnose_failure(llm, &task.id, cmd, &r).await {
                    tracing::warn!(
                        task_id = %task.id,
                        cmd = %cmd,
                        "Fresh replay (base): LLM says f2p fails due to broken environment"
                    );
                    return TestRunResult::EnvironmentBroken {
                        cmd: cmd.clone(),
                        diagnosis: "f2p fails due to broken environment, not test logic"
                            .to_string(),
                    };
                }
            }
            tracing::info!(
                task_id = %task.id,
                cmd = %cmd,
                "Fresh replay (base): f2p correctly fails"
            );
        }

        // Base commit: p2p must PASS
        for cmd in &task.pass_to_pass {
            let r = sandbox.exec(&format!("cd /repo && {}", cmd), 120_000).await;
            if r.exit_code != 0 {
                // p2p should pass -- ask LLM to classify the failure
                if let Some(ref llm) = self.llm {
                    if llm_diagnose_failure(llm, &task.id, cmd, &r).await {
                        tracing::warn!(
                            task_id = %task.id,
                            cmd = %cmd,
                            "Fresh replay (base): LLM says p2p fails due to broken environment"
                        );
                        return TestRunResult::EnvironmentBroken {
                            cmd: cmd.clone(),
                            diagnosis: "p2p fails due to broken environment".to_string(),
                        };
                    }
                }
                tracing::warn!(
                    task_id = %task.id,
                    cmd = %cmd,
                    exit = r.exit_code,
                    "Fresh replay (base): p2p fails on base"
                );
                return TestRunResult::RealFailure {
                    cmd: cmd.clone(),
                    detail: format!("p2p fails on base (exit {})", r.exit_code),
                };
            }
            tracing::info!(
                task_id = %task.id,
                cmd = %cmd,
                "Fresh replay (base): p2p passes"
            );
        }

        TestRunResult::Ok
    }

    /// Apply the patch and run tests in an isolated container: f2p must PASS, p2p must PASS.
    /// Uses LLM to classify ambiguous failures as real vs environment issues.
    async fn run_patched_tests(&self, sandbox: &DockerSandbox, task: &SweTask) -> TestRunResult {
        // Apply patch
        if let Err(e) = sandbox
            .write_file(".swe_forge_validation.patch", &task.patch)
            .await
        {
            tracing::warn!(task_id = %task.id, error = %e, "Fresh replay (patched): failed to write patch");
            return TestRunResult::RealFailure {
                cmd: "(patch apply)".to_string(),
                detail: format!("failed to write patch: {e}"),
            };
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
                tracing::warn!(task_id = %task.id, "Fresh replay (patched): patch apply failed");
                return TestRunResult::RealFailure {
                    cmd: "(patch apply)".to_string(),
                    detail: "patch apply failed".to_string(),
                };
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

        // Patched: f2p must PASS
        for cmd in &task.fail_to_pass {
            let r = sandbox.exec(&format!("cd /repo && {}", cmd), 120_000).await;
            if r.exit_code != 0 {
                if let Some(ref llm) = self.llm {
                    if llm_diagnose_failure(llm, &task.id, cmd, &r).await {
                        tracing::warn!(
                            task_id = %task.id,
                            cmd = %cmd,
                            "Fresh replay (patched): LLM says f2p fails due to broken environment"
                        );
                        return TestRunResult::EnvironmentBroken {
                            cmd: cmd.clone(),
                            diagnosis: "f2p fails due to broken environment after patch"
                                .to_string(),
                        };
                    }
                }
                tracing::warn!(
                    task_id = %task.id,
                    cmd = %cmd,
                    exit = r.exit_code,
                    "Fresh replay (patched): f2p still fails after patch"
                );
                return TestRunResult::RealFailure {
                    cmd: cmd.clone(),
                    detail: format!("f2p still fails after patch (exit {})", r.exit_code),
                };
            }
            tracing::info!(
                task_id = %task.id,
                cmd = %cmd,
                "Fresh replay (patched): f2p passes after patch"
            );
        }

        // Patched: p2p must still PASS
        for cmd in &task.pass_to_pass {
            let r = sandbox.exec(&format!("cd /repo && {}", cmd), 120_000).await;
            if r.exit_code != 0 {
                if let Some(ref llm) = self.llm {
                    if llm_diagnose_failure(llm, &task.id, cmd, &r).await {
                        tracing::warn!(
                            task_id = %task.id,
                            cmd = %cmd,
                            "Fresh replay (patched): LLM says p2p fails due to broken environment"
                        );
                        return TestRunResult::EnvironmentBroken {
                            cmd: cmd.clone(),
                            diagnosis: "p2p fails due to broken environment after patch"
                                .to_string(),
                        };
                    }
                }
                tracing::warn!(
                    task_id = %task.id,
                    cmd = %cmd,
                    exit = r.exit_code,
                    "Fresh replay (patched): p2p fails after patch"
                );
                return TestRunResult::RealFailure {
                    cmd: cmd.clone(),
                    detail: format!("p2p fails after patch (exit {})", r.exit_code),
                };
            }
            tracing::info!(
                task_id = %task.id,
                cmd = %cmd,
                "Fresh replay (patched): p2p still passes"
            );
        }

        TestRunResult::Ok
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
