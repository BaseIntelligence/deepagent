//! SWE-bench evaluation harness.
//!
//! Runs an external agent on mined SWE tasks inside Docker containers,
//! then verifies results by executing fail_to_pass / pass_to_pass test commands.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tracing::{info, warn};

use super::{validate_file_path, validate_git_ref, validate_repo_name, Rechecker, RecheckerConfig, SweTask};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub agent_dir: PathBuf,
    pub agent_cmd: String,
    pub agent_timeout_secs: u64,
    pub test_timeout_secs: u64,
    pub docker_image: String,
    pub keep_containers: bool,
    pub parallel: usize,
    /// Maximum rechecker attempts for setup errors (0 = disabled)
    pub rechecker_max_attempts: u32,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            agent_dir: PathBuf::from("."),
            agent_cmd: "python /agent/agent.py".to_string(),
            agent_timeout_secs: 600,
            test_timeout_secs: 120,
            docker_image: "python:3.12-slim".to_string(),
            keep_containers: false,
            parallel: 1,
            rechecker_max_attempts: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarnessStatus {
    Resolved,
    Unresolved,
    AgentError,
    TestError,
    SetupError,
    SanityFail,
}

impl std::fmt::Display for HarnessStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resolved => write!(f, "resolved"),
            Self::Unresolved => write!(f, "unresolved"),
            Self::AgentError => write!(f, "agent_error"),
            Self::TestError => write!(f, "test_error"),
            Self::SetupError => write!(f, "setup_error"),
            Self::SanityFail => write!(f, "sanity_fail"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub command: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub passed: bool,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessResult {
    pub task_id: String,
    pub repo: String,
    pub status: HarnessStatus,
    pub sanity_check: bool,
    pub fail_to_pass: Vec<TestResult>,
    pub pass_to_pass: Vec<TestResult>,
    pub agent_duration_secs: f64,
    pub total_duration_secs: f64,
    pub agent_output: String,
    pub error: Option<String>,
    pub container_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessSummary {
    pub total: usize,
    pub resolved: usize,
    pub unresolved: usize,
    pub agent_error: usize,
    pub test_error: usize,
    pub setup_error: usize,
    pub sanity_fail: usize,
    pub avg_agent_time_secs: f64,
    pub results: Vec<HarnessResult>,
}

// ---------------------------------------------------------------------------
// Docker helpers
// ---------------------------------------------------------------------------

async fn docker_exec(container: &str, cmd: &str, timeout_secs: u64) -> (i32, String, String) {
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        Command::new("docker")
            .args(["exec", container, "bash", "-c", cmd])
            .output(),
    )
    .await;

    match result {
        Ok(Ok(output)) => (
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stdout).to_string(),
            String::from_utf8_lossy(&output.stderr).to_string(),
        ),
        Ok(Err(e)) => (-1, String::new(), format!("exec error: {e}")),
        Err(_) => (
            -1,
            String::new(),
            format!("timed out after {timeout_secs}s"),
        ),
    }
}

async fn docker_rm(container: &str) {
    if let Err(e) = Command::new("docker")
        .args(["rm", "-f", container])
        .output()
        .await
    {
        tracing::debug!(container = container, error = %e, "Failed to remove container (may not exist)");
    }
}

async fn docker_write_file(container: &str, path: &str, content: &str) -> Result<()> {
    validate_file_path(path)?;

    use tokio::io::AsyncWriteExt;
    let tee_cmd = format!("cat > '/repo/{}'", path);
    let mut child = Command::new("docker")
        .args([
            "exec", "-i", "-w", "/repo", container, "bash", "-c", &tee_cmd,
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    if let Some(ref mut stdin) = child.stdin {
        stdin.write_all(content.as_bytes()).await?;
        stdin.shutdown().await?;
    }
    let output = child.wait_with_output().await?;
    if !output.status.success() {
        anyhow::bail!("write failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    Ok(())
}

fn container_name(task_id: &str) -> String {
    let safe = task_id.replace('/', "-").replace(' ', "_");
    format!("swe-harness-{safe}")
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max;
        while !s.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        format!("{}... [truncated]", &s[..end])
    }
}

// ---------------------------------------------------------------------------
// Per-task evaluation
// ---------------------------------------------------------------------------

async fn evaluate_task(task: &SweTask, config: &HarnessConfig) -> HarnessResult {
    let total_start = Instant::now();
    let cname = container_name(&task.id);
    let mut result = HarnessResult {
        task_id: task.id.clone(),
        repo: task.repo.clone(),
        status: HarnessStatus::SetupError,
        sanity_check: false,
        fail_to_pass: Vec::new(),
        pass_to_pass: Vec::new(),
        agent_duration_secs: 0.0,
        total_duration_secs: 0.0,
        agent_output: String::new(),
        error: None,
        container_id: Some(cname.clone()),
    };

    // 0. INPUT VALIDATION: reject shell-unsafe repo names and commit refs
    if let Err(e) = validate_repo_name(&task.repo) {
        result.error = Some(format!("Invalid repo name: {e}"));
        return result;
    }
    if !task.base_commit.is_empty() {
        if let Err(e) = validate_git_ref(&task.base_commit) {
            result.error = Some(format!("Invalid base commit: {e}"));
            return result;
        }
    }

    // 1. SETUP: start container
    info!(task_id = %task.id, "Starting container {}", cname);
    let agent_dir_abs = if let Ok(docker_dir) = std::env::var("DOCKER_AGENT_DIR") {
        PathBuf::from(docker_dir)
    } else {
        std::fs::canonicalize(&config.agent_dir).unwrap_or_else(|_| config.agent_dir.clone())
    };

    let docker_image = config.docker_image.clone();
    info!(task_id = %task.id, language = %task.language, image = %docker_image, "Selected Docker image");

    // Remove stale container if exists
    docker_rm(&cname).await;

    let openrouter_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    let openrouter_env = format!("OPENROUTER_API_KEY={}", openrouter_key);
    let chutes_key = std::env::var("CHUTES_API_KEY").unwrap_or_default();
    let chutes_env = format!("CHUTES_API_KEY={}", chutes_key);
    let github_token = std::env::var("GITHUB_TOKEN").unwrap_or_default();
    let github_token_env = format!("GITHUB_TOKEN={}", github_token);
    let start_output = Command::new("docker")
        .args([
            "run",
            "-d",
            "--name",
            &cname,
            "--network=host",
            "--memory=32g",
            "-e",
            &openrouter_env,
            "-e",
            &chutes_env,
            "-e",
            &github_token_env,
            "-w",
            "/repo",
            &docker_image,
            "sleep",
            "7200",
        ])
        .output()
        .await;

    match start_output {
        Ok(o) if o.status.success() => {
            info!(task_id = %task.id, "Container started");
        }
        Ok(o) => {
            result.error = Some(format!(
                "Failed to start container: {}",
                String::from_utf8_lossy(&o.stderr)
            ));
            return result;
        }
        Err(e) => {
            result.error = Some(format!("Docker not available: {e}"));
            return result;
        }
    }

    // Copy agent directory into the container (avoids bind-mount issues with nested mounts)
    let agent_src = format!("{}/.", agent_dir_abs.display());
    let agent_dst = format!("{}:/agent/", cname);
    let cp_result = Command::new("docker")
        .args(["cp", &agent_src, &agent_dst])
        .output()
        .await;
    match cp_result {
        Ok(o) if o.status.success() => {
            info!(task_id = %task.id, "Copied agent directory into container");
        }
        Ok(o) => {
            result.error = Some(format!(
                "Failed to copy agent into container: {}",
                String::from_utf8_lossy(&o.stderr)
            ));
            return result;
        }
        Err(e) => {
            result.error = Some(format!("docker cp failed: {e}"));
            return result;
        }
    }

    // Install system tools (always include python3 + pip for the agent)
    let (code, _, err) = docker_exec(
        &cname,
        "apt-get update -qq && apt-get install -y -qq git curl build-essential python3 python3-pip python3-venv > /dev/null 2>&1 && (command -v python >/dev/null 2>&1 || ln -sf $(command -v python3) /usr/local/bin/python)",
        180,
    )
    .await;
    if code != 0 {
        result.error = Some(format!(
            "Failed to install system deps: {}",
            truncate(&err, 500)
        ));
        return result;
    }

    // Clone repo (full clone for reliable checkout)
    let clone_cmd = format!("git clone https://github.com/{}.git /repo 2>&1", task.repo);
    let (code, _, err) = docker_exec(&cname, &clone_cmd, 600).await;
    if code != 0 {
        result.error = Some(format!("Clone failed: {}", truncate(&err, 500)));
        return result;
    }

    // Checkout base commit
    if !task.base_commit.is_empty() {
        let (code, _, err) = docker_exec(
            &cname,
            &format!("cd /repo && git checkout {} --force 2>&1", task.base_commit),
            60,
        )
        .await;
        if code != 0 {
            result.error = Some(format!("Checkout failed: {}", truncate(&err, 500)));
            return result;
        }
    }

    // Install language runtime from install_config version fields
    let runtime_cmds = SweTask::runtime_install_commands(&task.install_config);
    if !runtime_cmds.is_empty() {
        info!(task_id = %task.id, "Installing runtime: {}", truncate(&runtime_cmds, 120));
        let (code, _, err) = docker_exec(&cname, &format!("{} 2>&1", runtime_cmds), 300).await;
        if code != 0 {
            warn!(task_id = %task.id, "Runtime install failed (continuing): {}", truncate(&err, 200));
        }
    }

    // Install project deps from install_config with rechecker retry logic
    let mut setup_error: Option<String> = None;
    if let Some(install_cmd) = task.install_config.get("install") {
        if !install_cmd.is_empty() {
            info!(task_id = %task.id, "Installing deps: {}", install_cmd);
            let (code, _, err) =
                docker_exec(&cname, &format!("cd /repo && {} 2>&1", install_cmd), 300).await;
            if code != 0 {
                let err_msg = truncate(&err, 500);
                warn!(task_id = %task.id, "Install command failed: {}", err_msg);
                setup_error = Some(format!("Install failed: {}", err_msg));
            }
        }
    }

    // If install failed, use rechecker to try alternative strategies
    let mut rechecker_fix_applied = false;
    if setup_error.is_some() && config.rechecker_max_attempts > 0 {
        let rechecker_config = RecheckerConfig::with_max_attempts(config.rechecker_max_attempts);
        let rechecker = Rechecker::new(rechecker_config);

        info!(task_id = %task.id, "Attempting rechecker fixes for setup error");

        // Try alternative install strategies
        for attempt in 1..=config.rechecker_max_attempts {
            if let Some(next_cmd) = rechecker.get_next_install_attempt(
                task,
                setup_error.as_deref(),
                attempt,
            ) {
                info!(task_id = %task.id, attempt = attempt, "Trying alternative install: {}", truncate(&next_cmd, 100));

                // Remove existing container and start fresh for retry
                docker_rm(&cname).await;

                // Restart container with same config
                let start_output = Command::new("docker")
                    .args([
                        "run",
                        "-d",
                        "--name",
                        &cname,
                        "--network=host",
                        "--memory=32g",
                        "-w",
                        "/repo",
                        &docker_image,
                        "sleep",
                        "7200",
                    ])
                    .output()
                    .await;

                match start_output {
                    Ok(o) if o.status.success() => {
                        // Re-clone and checkout
                        let clone_cmd = format!("git clone https://github.com/{}.git /repo 2>&1", task.repo);
                        let (code, _, _) = docker_exec(&cname, &clone_cmd, 600).await;
                        if code != 0 {
                            warn!(task_id = %task.id, attempt = attempt, "Retry clone failed");
                            continue;
                        }

                        if !task.base_commit.is_empty() {
                            let (code, _, _) = docker_exec(
                                &cname,
                                &format!("cd /repo && git checkout {} --force 2>&1", task.base_commit),
                                60,
                            )
                            .await;
                            if code != 0 {
                                warn!(task_id = %task.id, attempt = attempt, "Retry checkout failed");
                                continue;
                            }
                        }

                        // Re-install runtime
                        let runtime_cmds = SweTask::runtime_install_commands(&task.install_config);
                        if !runtime_cmds.is_empty() {
                            let _ = docker_exec(&cname, &format!("{} 2>&1", runtime_cmds), 300).await;
                        }

                        // Try the alternative install command
                        let (code, _, err) = docker_exec(
                            &cname,
                            &format!("cd /repo && {} 2>&1", next_cmd),
                            300,
                        )
                        .await;

                        if code == 0 {
                            info!(task_id = %task.id, attempt = attempt, "Alternative install succeeded");
                            setup_error = None;
                            rechecker_fix_applied = true;
                            break; // Success!
                        } else {
                            warn!(task_id = %task.id, attempt = attempt, "Alternative install failed: {}", truncate(&err, 200));
                        }
                    }
                    _ => {
                        warn!(task_id = %task.id, attempt = attempt, "Failed to restart container for retry");
                    }
                }
            } else {
                warn!(task_id = %task.id, attempt = attempt, "No more alternative strategies available");
                break;
            }
        }

        // If we succeeded with a fix, copy agent files again
        if rechecker_fix_applied {
            let agent_src = format!("{}/.", agent_dir_abs.display());
            let agent_dst = format!("{}:/agent/", cname);
            let _ = Command::new("docker")
                .args(["cp", &agent_src, &agent_dst])
                .output()
                .await;
        }
    }

    // If setup still failed after all retries, return SetupError
    if let Some(err) = setup_error {
        result.status = HarnessStatus::SetupError;
        result.error = Some(err);
        if !config.keep_containers {
            docker_rm(&cname).await;
        }
        result.total_duration_secs = total_start.elapsed().as_secs_f64();
        return result;
    }

    // Install agent requirements
    let (code, _, _) = docker_exec(
        &cname,
        "test -f /agent/requirements.txt && pip install --break-system-packages -q -r /agent/requirements.txt 2>&1 || true",
        180,
    )
    .await;
    if code != 0 {
        warn!(task_id = %task.id, "Agent requirements install returned non-zero (continuing)");
    }

    // Copy test files into container
    if let Some(test_files_json) = task.meta.get("test_files") {
        if let Ok(files) =
            serde_json::from_str::<Vec<super::test_generator::TestFile>>(test_files_json)
        {
            for tf in &files {
                if let Err(e) = validate_file_path(&tf.path) {
                    warn!(task_id = %task.id, path = %tf.path, error = %e, "Skipping test file with invalid path");
                    continue;
                }
                let mkdir_cmd = format!("mkdir -p \"$(dirname '/repo/{}')\"", tf.path);
                docker_exec(&cname, &mkdir_cmd, 10).await;
                let write_result = docker_write_file(&cname, &tf.path, &tf.content).await;
                if let Err(e) = write_result {
                    warn!(task_id = %task.id, path = %tf.path, "Failed to copy test file: {}", e);
                }
            }
            info!(task_id = %task.id, "Copied {} test files into container", files.len());
        }
    }

    // 2. SANITY CHECK: fail_to_pass must fail, pass_to_pass must pass
    info!(task_id = %task.id, "Running sanity checks...");

    for cmd_str in &task.fail_to_pass {
        let (code, _, _) = docker_exec(
            &cname,
            &format!("cd /repo && {}", cmd_str),
            config.test_timeout_secs,
        )
        .await;
        if code == 0 {
            info!(task_id = %task.id, "Sanity fail: fail_to_pass command already passes: {}", cmd_str);
            result.status = HarnessStatus::SanityFail;
            result.error = Some(format!(
                "fail_to_pass command already passes on base commit: {}",
                cmd_str
            ));
            if !config.keep_containers {
                docker_rm(&cname).await;
            }
            result.total_duration_secs = total_start.elapsed().as_secs_f64();
            return result;
        }
    }

    for cmd_str in &task.pass_to_pass {
        let (code, _, _) = docker_exec(
            &cname,
            &format!("cd /repo && {}", cmd_str),
            config.test_timeout_secs,
        )
        .await;
        if code != 0 {
            info!(task_id = %task.id, "Sanity fail: pass_to_pass command fails on base commit: {}", cmd_str);
            result.status = HarnessStatus::SanityFail;
            result.error = Some(format!(
                "pass_to_pass command fails on base commit: {}",
                cmd_str
            ));
            if !config.keep_containers {
                docker_rm(&cname).await;
            }
            result.total_duration_secs = total_start.elapsed().as_secs_f64();
            return result;
        }
    }

    result.sanity_check = true;
    info!(task_id = %task.id, "Sanity check passed");

    // 3. RUN AGENT
    info!(task_id = %task.id, "Running agent: {}", config.agent_cmd);
    let prompt_escaped = task.prompt.replace('\'', "'\\''");
    let agent_full_cmd = format!(
        "cd /repo && {} --instruction '{}' 2>&1",
        config.agent_cmd, prompt_escaped
    );

    let agent_start = Instant::now();
    let (agent_code, agent_stdout, agent_stderr) =
        docker_exec(&cname, &agent_full_cmd, config.agent_timeout_secs).await;
    result.agent_duration_secs = agent_start.elapsed().as_secs_f64();
    result.agent_output = truncate(&format!("{agent_stdout}\n{agent_stderr}"), 10_000);

    if agent_code != 0 && agent_stderr.contains("timed out") {
        result.status = HarnessStatus::AgentError;
        result.error = Some(format!(
            "Agent timed out after {}s",
            config.agent_timeout_secs
        ));
        if !config.keep_containers {
            docker_rm(&cname).await;
        }
        result.total_duration_secs = total_start.elapsed().as_secs_f64();
        return result;
    }

    if agent_code != 0 {
        warn!(task_id = %task.id, exit_code = agent_code, "Agent exited with non-zero code (continuing to test)");
    }

    // 4. VERIFY: run all test commands
    info!(task_id = %task.id, "Verifying test results...");

    let mut all_f2p_pass = true;
    for cmd_str in &task.fail_to_pass {
        let cmd_start = Instant::now();
        let (code, stdout, stderr) = docker_exec(
            &cname,
            &format!("cd /repo && {}", cmd_str),
            config.test_timeout_secs,
        )
        .await;
        let passed = code == 0;
        if !passed {
            all_f2p_pass = false;
        }
        result.fail_to_pass.push(TestResult {
            command: cmd_str.clone(),
            exit_code: code,
            stdout: truncate(&stdout, 2000),
            stderr: truncate(&stderr, 2000),
            passed,
            duration_ms: cmd_start.elapsed().as_millis() as u64,
        });
    }

    let mut all_p2p_pass = true;
    for cmd_str in &task.pass_to_pass {
        let cmd_start = Instant::now();
        let (code, stdout, stderr) = docker_exec(
            &cname,
            &format!("cd /repo && {}", cmd_str),
            config.test_timeout_secs,
        )
        .await;
        let passed = code == 0;
        if !passed {
            all_p2p_pass = false;
        }
        result.pass_to_pass.push(TestResult {
            command: cmd_str.clone(),
            exit_code: code,
            stdout: truncate(&stdout, 2000),
            stderr: truncate(&stderr, 2000),
            passed,
            duration_ms: cmd_start.elapsed().as_millis() as u64,
        });
    }

    // Determine final status
    if all_f2p_pass && all_p2p_pass {
        result.status = HarnessStatus::Resolved;
        info!(task_id = %task.id, "RESOLVED");
    } else {
        result.status = HarnessStatus::Unresolved;
        let f2p_passed = result.fail_to_pass.iter().filter(|t| t.passed).count();
        let p2p_passed = result.pass_to_pass.iter().filter(|t| t.passed).count();
        info!(
            task_id = %task.id,
            "UNRESOLVED (f2p: {}/{}, p2p: {}/{})",
            f2p_passed, result.fail_to_pass.len(),
            p2p_passed, result.pass_to_pass.len()
        );
    }

    // 5. CLEANUP
    if !config.keep_containers {
        docker_rm(&cname).await;
    }

    result.total_duration_secs = total_start.elapsed().as_secs_f64();
    result
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load SweTask from a workspace.yaml file.
pub fn load_task(workspace_yaml: &Path) -> Result<SweTask> {
    let content = std::fs::read_to_string(workspace_yaml)?;
    let task: SweTask = serde_yaml::from_str(&content)?;
    Ok(task)
}

/// Discover all workspace.yaml files under a directory.
pub fn discover_tasks(input_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    fn walk(dir: &Path, paths: &mut Vec<PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    walk(&p, paths);
                } else if p
                    .file_name()
                    .map(|f| f == "workspace.yaml")
                    .unwrap_or(false)
                {
                    paths.push(p);
                }
            }
        }
    }
    walk(input_dir, &mut paths);
    paths.sort();
    Ok(paths)
}

/// Run the full harness on all tasks in a directory.
pub async fn run_harness(input_dir: &Path, config: &HarnessConfig) -> Result<HarnessSummary> {
    let yaml_paths = discover_tasks(input_dir)?;
    if yaml_paths.is_empty() {
        anyhow::bail!("No workspace.yaml files found in {}", input_dir.display());
    }

    info!(
        "Discovered {} tasks in {}",
        yaml_paths.len(),
        input_dir.display()
    );

    let mut tasks = Vec::new();
    for path in &yaml_paths {
        match load_task(path) {
            Ok(task) => tasks.push(task),
            Err(e) => warn!("Failed to load {}: {}", path.display(), e),
        }
    }

    if tasks.is_empty() {
        anyhow::bail!("No valid tasks found");
    }

    info!(
        "Loaded {} valid tasks, running with parallelism={}",
        tasks.len(),
        config.parallel
    );

    let mut results = Vec::new();

    // Process in chunks of `parallel` size
    for chunk in tasks.chunks(config.parallel) {
        let mut handles = Vec::new();
        for task in chunk {
            let task = task.clone();
            let cfg = config.clone();
            handles.push(tokio::spawn(
                async move { evaluate_task(&task, &cfg).await },
            ));
        }
        for handle in handles {
            match handle.await {
                Ok(r) => results.push(r),
                Err(e) => warn!("Task panicked: {e}"),
            }
        }
    }

    // Build summary
    let total = results.len();
    let resolved = results
        .iter()
        .filter(|r| matches!(r.status, HarnessStatus::Resolved))
        .count();
    let unresolved = results
        .iter()
        .filter(|r| matches!(r.status, HarnessStatus::Unresolved))
        .count();
    let agent_error = results
        .iter()
        .filter(|r| matches!(r.status, HarnessStatus::AgentError))
        .count();
    let test_error = results
        .iter()
        .filter(|r| matches!(r.status, HarnessStatus::TestError))
        .count();
    let setup_error = results
        .iter()
        .filter(|r| matches!(r.status, HarnessStatus::SetupError))
        .count();
    let sanity_fail = results
        .iter()
        .filter(|r| matches!(r.status, HarnessStatus::SanityFail))
        .count();

    let agent_times: Vec<f64> = results
        .iter()
        .filter(|r| r.agent_duration_secs > 0.0)
        .map(|r| r.agent_duration_secs)
        .collect();
    let avg_agent_time = if agent_times.is_empty() {
        0.0
    } else {
        agent_times.iter().sum::<f64>() / agent_times.len() as f64
    };

    Ok(HarnessSummary {
        total,
        resolved,
        unresolved,
        agent_error,
        test_error,
        setup_error,
        sanity_fail,
        avg_agent_time_secs: (avg_agent_time * 10.0).round() / 10.0,
        results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harness_config_default() {
        let config = HarnessConfig::default();
        assert_eq!(config.agent_timeout_secs, 600);
        assert_eq!(config.test_timeout_secs, 120);
        assert_eq!(config.docker_image, "python:3.12-slim");
        assert!(!config.keep_containers);
        assert_eq!(config.parallel, 1);
        assert_eq!(config.agent_cmd, "python /agent/agent.py");
        assert_eq!(config.rechecker_max_attempts, 3); // Default rechecker attempts
    }

    #[test]
    fn test_harness_config_rechecker_disabled() {
        let config = HarnessConfig {
            rechecker_max_attempts: 0,
            ..Default::default()
        };
        assert_eq!(config.rechecker_max_attempts, 0);
    }

    #[test]
    fn test_harness_config_custom_rechecker_attempts() {
        let config = HarnessConfig {
            rechecker_max_attempts: 5,
            ..Default::default()
        };
        assert_eq!(config.rechecker_max_attempts, 5);
    }

    #[test]
    fn test_harness_status_display() {
        assert_eq!(format!("{}", HarnessStatus::Resolved), "resolved");
        assert_eq!(format!("{}", HarnessStatus::Unresolved), "unresolved");
        assert_eq!(format!("{}", HarnessStatus::AgentError), "agent_error");
        assert_eq!(format!("{}", HarnessStatus::TestError), "test_error");
        assert_eq!(format!("{}", HarnessStatus::SetupError), "setup_error");
        assert_eq!(format!("{}", HarnessStatus::SanityFail), "sanity_fail");
    }

    #[test]
    fn test_container_name_basic() {
        let name = container_name("owner/repo-123");
        assert_eq!(name, "swe-harness-owner-repo-123");
    }

    #[test]
    fn test_container_name_with_spaces() {
        let name = container_name("owner/repo name");
        assert_eq!(name, "swe-harness-owner-repo_name");
    }

    #[test]
    fn test_container_name_slashes() {
        let name = container_name("org/sub/repo");
        assert_eq!(name, "swe-harness-org-sub-repo");
    }

    #[test]
    fn test_truncate_short() {
        let result = truncate("hello", 10);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_truncate_long() {
        let result = truncate("hello world this is a long string", 10);
        assert!(result.len() <= 25); // 10 + "... [truncated]"
        assert!(result.ends_with("... [truncated]"));
    }

    #[test]
    fn test_truncate_exact_boundary() {
        let result = truncate("12345", 5);
        assert_eq!(result, "12345");
    }

    #[test]
    fn test_truncate_unicode() {
        let result = truncate("héllo wörld", 5);
        assert!(result.ends_with("... [truncated]"));
    }

    #[test]
    fn test_discover_tasks_empty_dir() {
        let tmp = std::env::temp_dir().join("swe_forge_test_discover_empty");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let paths = discover_tasks(&tmp).unwrap();
        assert!(paths.is_empty());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_discover_tasks_finds_workspace_yaml() {
        let tmp = std::env::temp_dir().join("swe_forge_test_discover_yaml");
        let _ = std::fs::remove_dir_all(&tmp);
        let sub = tmp.join("task-1");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("workspace.yaml"), "id: test").unwrap();

        let paths = discover_tasks(&tmp).unwrap();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].ends_with("workspace.yaml"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_harness_result_serialization() {
        let result = HarnessResult {
            task_id: "test-1".to_string(),
            repo: "owner/repo".to_string(),
            status: HarnessStatus::Resolved,
            sanity_check: true,
            fail_to_pass: vec![],
            pass_to_pass: vec![],
            agent_duration_secs: 10.5,
            total_duration_secs: 15.0,
            agent_output: "output".to_string(),
            error: None,
            container_id: Some("cname".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"task_id\":\"test-1\""));
        assert!(json.contains("\"resolved\""));
    }

    #[test]
    fn test_harness_summary_serialization() {
        let summary = HarnessSummary {
            total: 5,
            resolved: 3,
            unresolved: 1,
            agent_error: 0,
            test_error: 0,
            setup_error: 1,
            sanity_fail: 0,
            avg_agent_time_secs: 12.5,
            results: vec![],
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("\"total\":5"));
        assert!(json.contains("\"resolved\":3"));
    }

    #[test]
    fn test_test_result_serialization() {
        let tr = TestResult {
            command: "pytest test.py".to_string(),
            exit_code: 0,
            stdout: "ok".to_string(),
            stderr: String::new(),
            passed: true,
            duration_ms: 1500,
        };
        let json = serde_json::to_string(&tr).unwrap();
        assert!(json.contains("\"passed\":true"));
        assert!(json.contains("\"duration_ms\":1500"));
    }

    // =========================================================================
    // Test Semantics Validation Integration Tests
    // =========================================================================
    // These tests verify the core test validation logic (VAL-TEST-001 to 005):
    // - fail_to_pass tests fail on base commit
    // - pass_to_pass tests pass on base commit
    // - Patch application works correctly
    // - Tests pass after valid patch application
    // - Tests still fail after invalid/no patch

    /// Helper: Check if Docker is available
    async fn docker_available() -> bool {
        match tokio::process::Command::new("docker")
            .args(["ps"])
            .output()
            .await
        {
            Ok(output) => output.status.success(),
            Err(_) => false,
        }
    }

    /// Helper: Create a mock SweTask with known test behaviors
    fn create_mock_task(
        task_id: &str,
        fail_to_pass: Vec<String>,
        pass_to_pass: Vec<String>,
        patch: &str,
    ) -> SweTask {
        let mut task = SweTask::new(task_id, "octocat/Hello-World");
        task.language = "python".to_string();
        task.base_commit = "7fd1a60b01f91b314f59955a4e4d4e80d8edf11d".to_string();
        task.fail_to_pass = fail_to_pass;
        task.pass_to_pass = pass_to_pass;
        task.patch = patch.to_string();
        task.prompt = "Fix the bug in the code".to_string();
        task
    }

    /// Helper: Run test commands directly on a sandbox and collect results
    async fn run_test_commands(
        sandbox: &super::super::docker_sandbox::DockerSandbox,
        commands: &[String],
    ) -> Vec<(i32, bool)> {
        let mut results = Vec::new();
        for cmd in commands {
            let output = sandbox.exec(&format!("cd /repo && {}", cmd), 30_000).await;
            results.push((output.exit_code, output.exit_code == 0));
        }
        results
    }

    /// VAL-TEST-001: fail_to_pass tests MUST fail (exit != 0) when run on the base commit
    #[tokio::test]
    async fn test_semantics_fail_to_pass_fails_on_base() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        use super::super::docker_sandbox::DockerSandbox;

        // Create a task with fail_to_pass command that should fail on base
        let task = create_mock_task(
            "test-semantics-f2p-fails-base",
            vec!["python3 -c 'import sys; sys.exit(1)'".to_string()], // Always fails
            vec![],                                                   // No p2p tests
            "",                                                       // No patch
        );

        // Start sandbox at base commit
        let sandbox = DockerSandbox::start(
            &task.repo,
            &task.base_commit,
            &task.language,
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to start sandbox");

        // Run fail_to_pass command - should fail (exit != 0)
        let results = run_test_commands(&sandbox, &task.fail_to_pass).await;
        assert_eq!(results.len(), 1);
        assert_ne!(
            results[0].0, 0,
            "fail_to_pass command should fail on base commit"
        );

        sandbox.destroy().await;
    }

    /// VAL-TEST-002: pass_to_pass tests MUST pass (exit == 0) when run on the base commit
    #[tokio::test]
    async fn test_semantics_pass_to_pass_passes_on_base() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        use super::super::docker_sandbox::DockerSandbox;

        // Create a task with pass_to_pass command that should pass on base
        let task = create_mock_task(
            "test-semantics-p2p-passes-base",
            vec![],                                                   // No f2p tests
            vec!["python3 -c 'import sys; sys.exit(0)'".to_string()], // Always passes
            "",                                                       // No patch
        );

        // Start sandbox at base commit
        let sandbox = DockerSandbox::start(
            &task.repo,
            &task.base_commit,
            &task.language,
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to start sandbox");

        // Run pass_to_pass command - should pass (exit == 0)
        let results = run_test_commands(&sandbox, &task.pass_to_pass).await;
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].0, 0,
            "pass_to_pass command should pass on base commit"
        );

        sandbox.destroy().await;
    }

    /// VAL-TEST-003: fail_to_pass tests MUST pass (exit == 0) after the valid PR patch is applied
    #[tokio::test]
    async fn test_semantics_fail_to_pass_passes_after_valid_patch() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        use super::super::docker_sandbox::DockerSandbox;

        // Create a patch that creates a test file that will pass
        let patch = r#"diff --git a/test_fix.py b/test_fix.py
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/test_fix.py
@@ -0,0 +1 @@
+# This file makes the fix exist
"#;

        // Task with fail_to_pass that checks for the fix file
        let task = create_mock_task(
            "test-semantics-f2p-passes-after-patch",
            vec!["python3 -c 'import os; assert os.path.exists(\"test_fix.py\"); print(\"Fix verified\")'".to_string()],
            vec![],
            patch,
        );

        // Start sandbox at base commit
        let sandbox = DockerSandbox::start(
            &task.repo,
            &task.base_commit,
            &task.language,
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to start sandbox");

        // Verify fail_to_pass FAILS on base (before patch)
        let base_results = run_test_commands(&sandbox, &task.fail_to_pass).await;
        assert_ne!(
            base_results[0].0, 0,
            "fail_to_pass should fail on base before patch"
        );

        // Apply the patch
        sandbox
            .write_file(".swe_forge_validation.patch", &task.patch)
            .await
            .unwrap();
        let apply_result = sandbox
            .exec(
                "cd /repo && git apply --allow-empty .swe_forge_validation.patch 2>&1",
                30_000,
            )
            .await;
        assert_eq!(
            apply_result.exit_code, 0,
            "Patch should apply successfully: {}",
            apply_result.stderr
        );

        // Verify fail_to_pass PASSES after patch
        let patched_results = run_test_commands(&sandbox, &task.fail_to_pass).await;
        assert_eq!(
            patched_results[0].0, 0,
            "fail_to_pass should pass after valid patch"
        );

        sandbox.destroy().await;
    }

    /// VAL-TEST-004: pass_to_pass tests MUST still pass (exit == 0) after the PR patch is applied
    #[tokio::test]
    async fn test_semantics_pass_to_pass_still_passes_after_patch() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        use super::super::docker_sandbox::DockerSandbox;

        // Create a patch that adds a new file but doesn't break existing functionality
        let patch = r#"diff --git a/new_feature.py b/new_feature.py
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/new_feature.py
@@ -0,0 +1 @@
+# New feature added without breaking existing tests
"#;

        // Task with pass_to_pass that checks something that should still work after patch
        let task = create_mock_task(
            "test-semantics-p2p-still-passes",
            vec![],
            vec![
                "python3 -c 'import os; os.listdir(\".\"); print(\"Basic functionality works\")'"
                    .to_string(),
            ],
            patch,
        );

        // Start sandbox at base commit
        let sandbox = DockerSandbox::start(
            &task.repo,
            &task.base_commit,
            &task.language,
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to start sandbox");

        // Verify pass_to_pass PASSES on base
        let base_results = run_test_commands(&sandbox, &task.pass_to_pass).await;
        assert_eq!(base_results[0].0, 0, "pass_to_pass should pass on base");

        // Apply the patch
        sandbox
            .write_file(".swe_forge_validation.patch", &task.patch)
            .await
            .unwrap();
        let apply_result = sandbox
            .exec(
                "cd /repo && git apply --allow-empty .swe_forge_validation.patch 2>&1",
                30_000,
            )
            .await;
        assert_eq!(apply_result.exit_code, 0, "Patch should apply successfully");

        // Verify pass_to_pass STILL PASSES after patch (no regression)
        let patched_results = run_test_commands(&sandbox, &task.pass_to_pass).await;
        assert_eq!(
            patched_results[0].0, 0,
            "pass_to_pass should still pass after patch (no regression)"
        );

        sandbox.destroy().await;
    }

    /// VAL-TEST-005: fail_to_pass tests MUST still fail (exit != 0) after an invalid patch
    #[tokio::test]
    async fn test_semantics_fail_to_pass_still_fails_after_invalid_patch() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        use super::super::docker_sandbox::DockerSandbox;

        // Create an "invalid" patch that creates a file but doesn't fix the actual issue
        // We create a new file that is NOT the one the test checks for
        let invalid_patch = r#"diff --git a/wrong_fix.py b/wrong_fix.py
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/wrong_fix.py
@@ -0,0 +1 @@
+# This is not the required fix file
"#;

        // Task with fail_to_pass that checks for a specific fix file that won't exist
        let task = create_mock_task(
            "test-semantics-f2p-still-fails",
            vec!["python3 -c 'import os; assert os.path.exists(\"required_fix.py\"), \"Fix file not found\"; print(\"Fix verified\")'".to_string()],
            vec![],
            invalid_patch,
        );

        // Start sandbox at base commit
        let sandbox = DockerSandbox::start(
            &task.repo,
            &task.base_commit,
            &task.language,
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to start sandbox");

        // Verify fail_to_pass FAILS on base
        let base_results = run_test_commands(&sandbox, &task.fail_to_pass).await;
        assert_ne!(base_results[0].0, 0, "fail_to_pass should fail on base");

        // Apply the invalid patch
        sandbox
            .write_file(".swe_forge_validation.patch", &task.patch)
            .await
            .unwrap();
        let apply_result = sandbox
            .exec(
                "cd /repo && git apply .swe_forge_validation.patch 2>&1",
                30_000,
            )
            .await;
        assert_eq!(
            apply_result.exit_code, 0,
            "Patch should apply successfully: {}",
            apply_result.stderr
        );

        // Verify fail_to_pass STILL FAILS after invalid patch (wrong_fix.py != required_fix.py)
        let patched_results = run_test_commands(&sandbox, &task.fail_to_pass).await;
        assert_ne!(
            patched_results[0].0, 0,
            "fail_to_pass should still fail after invalid patch"
        );

        sandbox.destroy().await;
    }

    /// Combined test: Full test semantics validation with both f2p and p2p tests
    /// Verifies all test semantics in a single integrated workflow
    #[tokio::test]
    async fn test_semantics_full_validation_workflow() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        use super::super::docker_sandbox::DockerSandbox;

        // Create a patch that fixes the failing test
        let fix_patch = r#"diff --git a/fix.py b/fix.py
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/fix.py
@@ -0,0 +1,2 @@
+# This is the fix
+fixed = True
"#;

        // Task with both fail_to_pass and pass_to_pass tests
        let task = create_mock_task(
            "test-semantics-full-workflow",
            // fail_to_pass: checks that the fix exists
            vec!["python3 -c 'import os; assert os.path.exists(\"fix.py\"), \"Fix not found\"; print(\"Fix verified!\")'".to_string()],
            // pass_to_pass: checks basic functionality always works
            vec!["python3 -c 'print(\"Basic check passes\")'".to_string()],
            fix_patch,
        );

        // Start sandbox at base commit
        let sandbox = DockerSandbox::start(
            &task.repo,
            &task.base_commit,
            &task.language,
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to start sandbox");

        // --- BASE COMMIT VALIDATION ---
        tracing::info!("Testing base commit semantics");

        // fail_to_pass must FAIL on base
        let f2p_base = run_test_commands(&sandbox, &task.fail_to_pass).await;
        assert_eq!(f2p_base.len(), 1);
        assert_ne!(
            f2p_base[0].0, 0,
            "VAL-TEST-001: fail_to_pass must FAIL on base commit (exit != 0)"
        );

        // pass_to_pass must PASS on base
        let p2p_base = run_test_commands(&sandbox, &task.pass_to_pass).await;
        assert_eq!(p2p_base.len(), 1);
        assert_eq!(
            p2p_base[0].0, 0,
            "VAL-TEST-002: pass_to_pass must PASS on base commit (exit == 0)"
        );

        // --- APPLY PATCH ---
        sandbox
            .write_file(".swe_forge_validation.patch", &task.patch)
            .await
            .unwrap();
        let apply_result = sandbox
            .exec(
                "cd /repo && git apply --allow-empty .swe_forge_validation.patch 2>&1",
                30_000,
            )
            .await;
        assert_eq!(apply_result.exit_code, 0, "Patch should apply successfully");

        // --- PATCHED COMMIT VALIDATION ---
        tracing::info!("Testing patched commit semantics");

        // fail_to_pass must PASS after valid patch
        let f2p_patched = run_test_commands(&sandbox, &task.fail_to_pass).await;
        assert_eq!(f2p_patched.len(), 1);
        assert_eq!(
            f2p_patched[0].0, 0,
            "VAL-TEST-003: fail_to_pass must PASS after valid patch (exit == 0)"
        );

        // pass_to_pass must STILL PASS after patch (no regression)
        let p2p_patched = run_test_commands(&sandbox, &task.pass_to_pass).await;
        assert_eq!(p2p_patched.len(), 1);
        assert_eq!(
            p2p_patched[0].0, 0,
            "VAL-TEST-004: pass_to_pass must still PASS after patch (exit == 0)"
        );

        tracing::info!("Full test semantics validation workflow PASSED");

        sandbox.destroy().await;
    }

    /// Test harness sanity check behavior: when fail_to_pass passes on base, it should report SanityFail
    #[tokio::test]
    async fn test_harness_sanity_check_fail_to_pass_already_passes() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        // Create a task where fail_to_pass ALREADY passes on base (should trigger sanity fail)
        let task = create_mock_task(
            "test-sanity-f2p-already-passes",
            vec!["python3 -c 'print(\"already passes\")'".to_string()], // Already passes!
            vec!["python3 -c 'print(\"p2p passes\")'".to_string()],
            "", // No patch needed for sanity check test
        );

        // We can't easily run evaluate_task directly in tests (private),
        // but we can verify the harness would correctly detect this
        // by checking that the task's fail_to_pass command would pass on base
        use super::super::docker_sandbox::DockerSandbox;

        let sandbox = DockerSandbox::start(
            &task.repo,
            &task.base_commit,
            &task.language,
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to start sandbox");

        // Verify fail_to_pass passes on base (this would trigger SanityFail in harness)
        let result = sandbox
            .exec(&format!("cd /repo && {}", &task.fail_to_pass[0]), 30_000)
            .await;
        assert_eq!(
            result.exit_code, 0,
            "fail_to_pass command passes on base (would trigger SanityFail in harness)"
        );

        sandbox.destroy().await;
    }

    /// Test harness sanity check behavior: when pass_to_pass fails on base, it should report SanityFail
    #[tokio::test]
    async fn test_harness_sanity_check_pass_to_pass_already_fails() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        // Create a task where pass_to_pass ALREADY fails on base (should trigger sanity fail)
        let task = create_mock_task(
            "test-sanity-p2p-already-fails",
            vec!["python3 -c 'import sys; sys.exit(1)'".to_string()],
            vec!["python3 -c 'import sys; sys.exit(1)'".to_string()], // Already fails!
            "",
        );

        use super::super::docker_sandbox::DockerSandbox;

        let sandbox = DockerSandbox::start(
            &task.repo,
            &task.base_commit,
            &task.language,
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to start sandbox");

        // Verify pass_to_pass fails on base (this would trigger SanityFail in harness)
        let result = sandbox
            .exec(&format!("cd /repo && {}", &task.pass_to_pass[0]), 30_000)
            .await;
        assert_ne!(
            result.exit_code, 0,
            "pass_to_pass command fails on base (would trigger SanityFail in harness)"
        );

        sandbox.destroy().await;
    }
}
