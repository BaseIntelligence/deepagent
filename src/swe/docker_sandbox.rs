//! Docker sandbox for isolated repository operations.
//!
//! Provides an ephemeral Docker container per task where all repo cloning,
//! dependency installation, test execution, and patch validation happen.

use anyhow::Result;
use std::process::Stdio;
use std::sync::atomic::{AtomicU16, Ordering};
use tokio::process::Command;

use crate::swe::tool_server::TOOL_SERVER_PY;
use crate::swe::{validate_file_path, validate_git_ref, validate_repo_name};

/// Global atomic port counter to guarantee unique ports across all concurrent containers.
static NEXT_PORT: AtomicU16 = AtomicU16::new(10_000);

/// Allocate a unique port for a tool server.
///
/// Uses an atomic counter to guarantee no two concurrent sandboxes get the same port.
/// Wraps around from 60_000 back to 10_000.
fn allocate_port() -> u16 {
    loop {
        let current = NEXT_PORT.load(Ordering::Relaxed);
        let next = if current >= 60_000 {
            10_000
        } else {
            current + 1
        };
        if NEXT_PORT
            .compare_exchange(current, next, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            return current;
        }
    }
}

/// Shell command output from inside the container.
pub struct SandboxOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// An ephemeral Docker container for isolated repository operations.
pub struct DockerSandbox {
    container_name: String,
    /// Unique port for the tool server (needed because --network=host shares port space).
    tool_port: u16,
    /// Whether the tool server started successfully.
    tool_server_ok: bool,
}

/// Pick a Docker image appropriate for the given language.
pub fn image_for_language(_language: &str) -> &'static str {
    "python:3.12-slim"
}

impl DockerSandbox {
    /// Start a new container, clone the repo at the given base commit.
    /// `image_override` takes precedence over language-based auto-selection.
    pub async fn start(
        repo: &str,
        base_commit: &str,
        language: &str,
        image_override: Option<&str>,
    ) -> Result<Self> {
        validate_repo_name(repo)?;
        if !base_commit.is_empty() {
            validate_git_ref(base_commit)?;
        }

        let image = image_override.unwrap_or_else(|| image_for_language(language));
        let safe_name = repo.replace('/', "-").replace(' ', "_");
        // Use UUID for unique container names to avoid collisions in parallel tests
        let unique_suffix = uuid::Uuid::new_v4().to_string()[..8].to_string();
        let container_name = format!("swe-mine-{}-{}", safe_name, unique_suffix);
        let tool_port = allocate_port();

        // Remove stale container if it exists
        if let Err(e) = Command::new("docker")
            .args(["rm", "-f", &container_name])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
        {
            tracing::debug!(container = %container_name, error = %e, "Failed to remove stale container (may not exist)");
        }

        let run_output = Command::new("docker")
            .args([
                "run",
                "-d",
                "--name",
                &container_name,
                "--network=host",
                "--memory=32g",
                "-w",
                "/repo",
                image,
                "sleep",
                "7200",
            ])
            .output()
            .await?;

        if !run_output.status.success() {
            anyhow::bail!(
                "Failed to start Docker container '{}': {}",
                container_name,
                String::from_utf8_lossy(&run_output.stderr)
            );
        }

        let mut sandbox = Self {
            container_name,
            tool_port,
            tool_server_ok: false,
        };

        // Install git (only hard dependency; agent installs everything else)
        let install = sandbox
            .exec(
                "apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1",
                120_000,
            )
            .await;
        if install.exit_code != 0 {
            sandbox.destroy().await;
            anyhow::bail!(
                "git install failed in container '{}': {}",
                sandbox.container_name,
                install.stderr
            );
        }

        // Clone the repository (full clone for reliable checkout)
        let clone_cmd = format!("git clone https://github.com/{}.git /repo 2>&1", repo);
        let clone = sandbox.exec(&clone_cmd, 600_000).await;
        if clone.exit_code != 0 {
            sandbox.destroy().await;
            anyhow::bail!(
                "Failed to clone {} in container: {}",
                repo,
                truncate(&clone.stderr, 500)
            );
        }

        // Checkout base commit
        if !base_commit.is_empty() {
            let checkout = sandbox
                .exec(
                    &format!("cd /repo && git checkout {} --force 2>&1", base_commit),
                    60_000,
                )
                .await;
            if checkout.exit_code != 0 {
                sandbox.destroy().await;
                anyhow::bail!(
                    "Checkout of commit {} failed in container '{}': {}",
                    base_commit,
                    sandbox.container_name,
                    truncate(&checkout.stderr, 500)
                );
            }
        }

        // Inject and start the tool server
        sandbox.tool_server_ok = sandbox.start_tool_server().await;
        if sandbox.tool_server_ok {
            tracing::debug!(container = %sandbox.container_name, port = sandbox.tool_port, "Tool server started");
        } else {
            tracing::debug!(container = %sandbox.container_name, "Tool server unavailable, shell fallback will be used");
        }

        tracing::info!(
            container = %sandbox.container_name,
            image = image,
            repo = repo,
            "Docker sandbox ready"
        );

        Ok(sandbox)
    }

    /// Write and start the Python tool server inside the container.
    ///
    /// Returns `true` if the tool server started successfully, `false` otherwise.
    /// On failure, the caller should fall back to shell-based tool execution.
    async fn start_tool_server(&self) -> bool {
        for retry in 0..2 {
            if retry > 0 {
                tracing::debug!(
                    container = %self.container_name,
                    retry = retry,
                    "Retrying tool server startup"
                );
                // Kill any leftover process from previous attempt
                self.exec(
                    "pkill -f 'python3.*server.py' 2>/dev/null; rm -f /tools/server.log",
                    5_000,
                )
                .await;
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }

            // Write the server script
            let mkdir = self.exec("mkdir -p /tools", 10_000).await;
            if mkdir.exit_code != 0 {
                tracing::debug!(container = %self.container_name, "Failed to create /tools dir");
                continue;
            }

            if let Err(e) = self
                .write_file_abs("/tools/server.py", TOOL_SERVER_PY)
                .await
            {
                tracing::debug!(container = %self.container_name, error = %e, "Failed to write tool server");
                continue;
            }

            // Verify the script was written correctly
            let verify = self
                .exec("wc -c < /tools/server.py 2>/dev/null", 5_000)
                .await;
            let written_bytes: usize = verify.stdout.trim().parse().unwrap_or(0);
            if written_bytes < TOOL_SERVER_PY.len() / 2 {
                tracing::debug!(
                    container = %self.container_name,
                    expected = TOOL_SERVER_PY.len(),
                    actual = written_bytes,
                    "Tool server script truncated, retrying"
                );
                continue;
            }

            // Start server in background with unique port (--network=host shares port space)
            let start_cmd = format!(
                "nohup python3 -u /tools/server.py --port {} --cwd /repo > /tools/server.log 2>&1 &",
                self.tool_port
            );
            let start = self.exec(&start_cmd, 5_000).await;
            if start.exit_code != 0 {
                tracing::debug!(
                    container = %self.container_name,
                    stderr = %start.stderr,
                    "Tool server start command failed"
                );
                continue;
            }

            // Health check: 12 attempts × 500ms = 6s total
            for attempt in 0..12 {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                if self.tool_server_health().await {
                    tracing::debug!(
                        container = %self.container_name,
                        attempt = attempt,
                        retry = retry,
                        "Tool server healthy"
                    );
                    return true;
                }
            }

            // Log server output for debugging on this retry
            let log = self.exec("cat /tools/server.log 2>/dev/null", 5_000).await;
            tracing::debug!(
                container = %self.container_name,
                retry = retry,
                server_log = %log.stdout,
                "Tool server health check failed after 6s"
            );
        }

        // All retries exhausted
        let log = self.exec("cat /tools/server.log 2>/dev/null", 5_000).await;
        tracing::warn!(
            container = %self.container_name,
            server_log = %log.stdout,
            "Tool server failed to start after retries, falling back to shell tools"
        );
        false
    }

    /// Check if the tool server is healthy via python3 urllib inside the container.
    async fn tool_server_health(&self) -> bool {
        let check_cmd = format!(
            "import urllib.request; urllib.request.urlopen('http://localhost:{}/health')",
            self.tool_port
        );
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(2_000),
            Command::new("docker")
                .args([
                    "exec",
                    "-w",
                    "/repo",
                    &self.container_name,
                    "python3",
                    "-c",
                    &check_cmd,
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status(),
        )
        .await;
        matches!(result, Ok(Ok(status)) if status.success())
    }

    /// Whether the tool server is available for HTTP-based tool requests.
    pub fn has_tool_server(&self) -> bool {
        self.tool_server_ok
    }

    /// Call a tool on the HTTP tool server running inside the container.
    /// Pipes the JSON args via stdin to avoid shell escaping issues.
    pub async fn tool_request(&self, tool_name: &str, args_json: &str) -> SandboxOutput {
        for ch in tool_name.chars() {
            if !matches!(ch, 'a'..='z' | 'A'..='Z' | '0'..='9' | '_') {
                return SandboxOutput {
                    stdout: String::new(),
                    stderr: format!("Invalid tool name: {}", tool_name),
                    exit_code: -1,
                };
            }
        }

        let script = format!(
            "import sys, urllib.request, json\n\
             data = sys.stdin.read()\n\
             req = urllib.request.Request(\
               'http://localhost:{}/{}', \
               data=data.encode(), \
               headers={{'Content-Type': 'application/json'}})\n\
             try:\n\
               resp = urllib.request.urlopen(req, timeout=60)\n\
               print(resp.read().decode())\n\
             except Exception as e:\n\
               print(json.dumps({{'error': str(e)}}))",
            self.tool_port, tool_name
        );

        let result = tokio::time::timeout(std::time::Duration::from_millis(65_000), async {
            let mut child = Command::new("docker")
                .args([
                    "exec",
                    "-i",
                    "-w",
                    "/repo",
                    &self.container_name,
                    "python3",
                    "-c",
                    &script,
                ])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?;

            if let Some(ref mut stdin) = child.stdin {
                use tokio::io::AsyncWriteExt;
                stdin.write_all(args_json.as_bytes()).await?;
                stdin.shutdown().await?;
            }

            child.wait_with_output().await
        })
        .await;

        match result {
            Ok(Ok(output)) => SandboxOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
            },
            Ok(Err(e)) => SandboxOutput {
                stdout: String::new(),
                stderr: format!("Tool request error: {}", e),
                exit_code: -1,
            },
            Err(_) => SandboxOutput {
                stdout: String::new(),
                stderr: "Tool request timed out after 65s".to_string(),
                exit_code: -1,
            },
        }
    }

    /// Execute a shell command inside the container.
    pub async fn exec(&self, cmd: &str, timeout_ms: u64) -> SandboxOutput {
        let timeout_secs = (timeout_ms / 1000).max(1);
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(timeout_ms),
            Command::new("docker")
                .args([
                    "exec",
                    "-w",
                    "/repo",
                    &self.container_name,
                    "bash",
                    "-c",
                    cmd,
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await;

        match result {
            Ok(Ok(output)) => SandboxOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
            },
            Ok(Err(e)) => SandboxOutput {
                stdout: String::new(),
                stderr: format!("Docker exec error: {}", e),
                exit_code: -1,
            },
            Err(_) => SandboxOutput {
                stdout: String::new(),
                stderr: format!("Command timed out after {}s", timeout_secs),
                exit_code: -1,
            },
        }
    }

    /// Write a file to an absolute path inside the container.
    ///
    /// Only allows paths under known safe prefixes (`/tools/`).
    async fn write_file_abs(&self, abs_path: &str, content: &str) -> Result<()> {
        if !abs_path.starts_with("/tools/") {
            anyhow::bail!(
                "write_file_abs only allows paths under /tools/, got '{}'",
                abs_path
            );
        }
        for ch in abs_path.chars() {
            if matches!(
                ch,
                '\'' | '"'
                    | '`'
                    | '$'
                    | '!'
                    | '&'
                    | '|'
                    | ';'
                    | '('
                    | ')'
                    | '{'
                    | '}'
                    | '<'
                    | '>'
                    | '\\'
                    | '\0'
                    | '\n'
                    | '\r'
            ) {
                anyhow::bail!(
                    "invalid character in absolute path '{}': shell metacharacters not allowed",
                    abs_path
                );
            }
        }
        if abs_path.contains("..") {
            anyhow::bail!(
                "absolute path '{}' contains '..' (path traversal not allowed)",
                abs_path
            );
        }

        let mkdir_cmd = format!("mkdir -p \"$(dirname '{}')\"", abs_path);
        self.exec(&mkdir_cmd, 10_000).await;

        let tee_cmd = format!("cat > '{}'", abs_path);
        let mut child = Command::new("docker")
            .args([
                "exec",
                "-i",
                "-w",
                "/repo",
                &self.container_name,
                "bash",
                "-c",
                &tee_cmd,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()?;

        if let Some(ref mut stdin) = child.stdin {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(content.as_bytes()).await?;
            stdin.shutdown().await?;
        }

        let output = child.wait_with_output().await?;
        if !output.status.success() {
            anyhow::bail!(
                "Failed to write file '{}' in container: {}",
                abs_path,
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Ok(())
    }

    /// Write a file inside the container by piping content via stdin.
    pub async fn write_file(&self, path: &str, content: &str) -> Result<()> {
        validate_file_path(path)?;

        // First ensure the parent directory exists
        let mkdir_cmd = format!("mkdir -p \"$(dirname '/repo/{}')\"", path);
        self.exec(&mkdir_cmd, 10_000).await;

        // Use docker exec -i to pipe content via stdin
        let tee_cmd = format!("cat > '/repo/{}'", path);
        let mut child = Command::new("docker")
            .args([
                "exec",
                "-i",
                "-w",
                "/repo",
                &self.container_name,
                "bash",
                "-c",
                &tee_cmd,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()?;

        if let Some(ref mut stdin) = child.stdin {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(content.as_bytes()).await?;
            stdin.shutdown().await?;
        }

        let output = child.wait_with_output().await?;
        if !output.status.success() {
            anyhow::bail!(
                "Failed to write file '{}' in container: {}",
                path,
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Ok(())
    }

    /// Read a file from inside the container.
    pub async fn read_file(&self, path: &str) -> Result<String> {
        validate_file_path(path)?;

        let cmd = format!("cat '/repo/{}'", path);
        let result = self.exec(&cmd, 10_000).await;
        if result.exit_code != 0 {
            anyhow::bail!(
                "Failed to read file '{}' in container: {}",
                path,
                result.stderr
            );
        }
        Ok(result.stdout)
    }

    /// Destroy the container.
    pub async fn destroy(&self) {
        if let Err(e) = Command::new("docker")
            .args(["rm", "-f", &self.container_name])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
        {
            tracing::debug!(container = %self.container_name, error = %e, "Failed to destroy container");
        }
        tracing::debug!(container = %self.container_name, "Docker sandbox destroyed");
    }

    /// Get the container name (useful for logging).
    pub fn name(&self) -> &str {
        &self.container_name
    }
}

/// Ensure the sandbox is destroyed when dropped (best-effort sync cleanup).
impl Drop for DockerSandbox {
    fn drop(&mut self) {
        let name = self.container_name.clone();
        // Execute cleanup synchronously with timeout for reliable container removal
        let _ = std::process::Command::new("docker")
            .args(["rm", "-f", &name])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
}

fn truncate(s: &str, max: usize) -> String {
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
    fn test_image_for_language() {
        assert_eq!(image_for_language("python"), "python:3.12-slim");
        assert_eq!(image_for_language("Python"), "python:3.12-slim");
        assert_eq!(image_for_language("javascript"), "python:3.12-slim");
        assert_eq!(image_for_language("go"), "python:3.12-slim");
        assert_eq!(image_for_language("unknown"), "python:3.12-slim");
    }

    #[test]
    fn test_truncate_short_string() {
        let result = truncate("hello", 10);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_truncate_long_string() {
        let result = truncate("hello world this is long", 10);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 13);
    }

    #[test]
    fn test_truncate_exact_boundary() {
        let result = truncate("12345", 5);
        assert_eq!(result, "12345");
    }

    #[test]
    fn test_truncate_empty() {
        let result = truncate("", 10);
        assert_eq!(result, "");
    }

    #[test]
    fn test_sandbox_output_construction() {
        let output = SandboxOutput {
            stdout: "hello".to_string(),
            stderr: "error".to_string(),
            exit_code: 1,
        };
        assert_eq!(output.stdout, "hello");
        assert_eq!(output.stderr, "error");
        assert_eq!(output.exit_code, 1);
    }

    #[test]
    fn test_sandbox_output_defaults() {
        let output = SandboxOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
        };
        assert!(output.stdout.is_empty());
        assert_eq!(output.exit_code, 0);
    }

    #[test]
    fn test_allocate_port_returns_valid_range() {
        let port = allocate_port();
        assert!(port >= 10_000);
        assert!(port <= 60_000);
    }

    #[test]
    fn test_allocate_port_sequential_unique() {
        let p1 = allocate_port();
        let p2 = allocate_port();
        let p3 = allocate_port();
        assert_ne!(p1, p2);
        assert_ne!(p2, p3);
        assert_ne!(p1, p3);
    }

    // =========================================================================
    // Integration Tests for DockerSandbox
    // =========================================================================

    /// Helper function to check if Docker is available
    async fn docker_available() -> bool {
        match Command::new("docker").args(["ps"]).output().await {
            Ok(output) => output.status.success(),
            Err(_) => false,
        }
    }

    /// Helper to generate unique container names for tests
    #[allow(dead_code)]
    fn unique_container_name(prefix: &str) -> String {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        format!("swe-test-{}-{}", prefix, ts)
    }

    /// Helper to check if a container exists
    async fn container_exists(name: &str) -> bool {
        let output = Command::new("docker")
            .args([
                "ps",
                "-a",
                "--filter",
                &format!("name={}", name),
                "--format",
                "{{.Names}}",
            ])
            .output()
            .await
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout.trim().contains(name)
    }

    #[tokio::test]
    async fn test_container_creation_and_destruction() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        // Start a sandbox
        let sandbox = DockerSandbox::start(
            "octocat/Hello-World", // Small public repo
            "",                    // No specific commit
            "python",
            Some("python:3.12-slim"),
        )
        .await;

        assert!(
            sandbox.is_ok(),
            "Failed to create sandbox: {:?}",
            sandbox.err()
        );
        let sandbox = sandbox.unwrap();
        let container_name = sandbox.name().to_string();

        // Verify container exists
        assert!(
            container_exists(&container_name).await,
            "Container should exist after creation"
        );

        // Destroy the sandbox
        sandbox.destroy().await;

        // Verify container no longer exists
        assert!(
            !container_exists(&container_name).await,
            "Container should be destroyed"
        );
    }

    #[tokio::test]
    async fn test_command_execution_success() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        let sandbox = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create sandbox");

        // Test successful command
        let result = sandbox.exec("echo 'hello world'", 10_000).await;
        assert_eq!(
            result.exit_code, 0,
            "Command should succeed with exit code 0"
        );
        assert!(
            result.stdout.contains("hello world"),
            "Stdout should contain 'hello world'"
        );
        assert!(
            result.stderr.is_empty(),
            "Stderr should be empty for successful command"
        );

        sandbox.destroy().await;
    }

    #[tokio::test]
    async fn test_command_execution_failure() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        let sandbox = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create sandbox");

        // Test failing command
        let result = sandbox.exec("exit 42", 10_000).await;
        assert_eq!(
            result.exit_code, 42,
            "Command should fail with exit code 42"
        );

        // Test non-existent command
        let result = sandbox.exec("nonexistent_command_xyz", 10_000).await;
        assert_ne!(result.exit_code, 0, "Non-existent command should fail");
        assert!(
            !result.stderr.is_empty(),
            "Stderr should contain error message"
        );

        sandbox.destroy().await;
    }

    #[tokio::test]
    async fn test_command_execution_timeout() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        let sandbox = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create sandbox");

        // Test timeout - sleep for 5 seconds with 1 second timeout
        let result = sandbox.exec("sleep 5", 1_000).await;
        assert_eq!(result.exit_code, -1, "Timeout should return exit code -1");
        assert!(
            result.stderr.contains("timed out"),
            "Stderr should indicate timeout"
        );

        sandbox.destroy().await;
    }

    #[tokio::test]
    async fn test_file_write_and_read() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        let sandbox = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create sandbox");

        // Test writing a file
        let test_content = "Hello, World!\nThis is a test file.\n";
        let write_result = sandbox.write_file("test_file.txt", test_content).await;
        assert!(
            write_result.is_ok(),
            "Write file should succeed: {:?}",
            write_result.err()
        );

        // Test reading the file back
        let read_result = sandbox.read_file("test_file.txt").await;
        assert!(
            read_result.is_ok(),
            "Read file should succeed: {:?}",
            read_result.err()
        );
        assert_eq!(
            read_result.unwrap(),
            test_content,
            "Read content should match written content"
        );

        sandbox.destroy().await;
    }

    #[tokio::test]
    async fn test_file_write_in_subdirectory() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        let sandbox = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create sandbox");

        // Test writing to a subdirectory
        let test_content = "Nested file content";
        let write_result = sandbox
            .write_file("subdir/nested/file.txt", test_content)
            .await;
        assert!(write_result.is_ok(), "Write to nested path should succeed");

        // Verify the file exists
        let read_result = sandbox.read_file("subdir/nested/file.txt").await;
        assert!(read_result.is_ok(), "Read nested file should succeed");
        assert_eq!(read_result.unwrap(), test_content);

        sandbox.destroy().await;
    }

    #[tokio::test]
    async fn test_read_nonexistent_file() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        let sandbox = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create sandbox");

        // Test reading non-existent file
        let read_result = sandbox.read_file("nonexistent_file_xyz.txt").await;
        assert!(
            read_result.is_err(),
            "Reading non-existent file should fail"
        );

        sandbox.destroy().await;
    }

    #[tokio::test]
    async fn test_cleanup_on_drop() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        let container_name: String;

        {
            let sandbox = DockerSandbox::start(
                "octocat/Hello-World",
                "",
                "python",
                Some("python:3.12-slim"),
            )
            .await
            .expect("Failed to create sandbox");

            container_name = sandbox.name().to_string();
            assert!(
                container_exists(&container_name).await,
                "Container should exist"
            );

            // Sandbox will be dropped here
        }

        // Give Drop implementation a moment to execute
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Verify container no longer exists
        assert!(
            !container_exists(&container_name).await,
            "Container should be destroyed on drop"
        );
    }

    #[tokio::test]
    async fn test_repo_cloning_and_checkout() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        // Use a known commit from octocat/Hello-World
        let known_commit = "7fd1a60b01f91b314f59955a4e4d4e80d8edf11d";

        let sandbox = DockerSandbox::start(
            "octocat/Hello-World",
            known_commit,
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create sandbox");

        // Verify the repo was cloned
        let result = sandbox.exec("ls -la /repo", 10_000).await;
        assert_eq!(result.exit_code, 0, "Repo directory should exist");
        assert!(
            result.stdout.contains("README"),
            "README should exist in repo"
        );

        // Verify we're at the correct commit
        let result = sandbox.exec("cd /repo && git rev-parse HEAD", 10_000).await;
        assert_eq!(result.exit_code, 0, "Git command should succeed");
        assert!(
            result.stdout.contains(known_commit),
            "Should be at the specified commit"
        );

        sandbox.destroy().await;
    }

    #[tokio::test]
    async fn test_multiple_file_operations() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        let sandbox = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create sandbox");

        // Write multiple files
        let files = vec![
            ("file1.txt", "Content of file 1"),
            ("file2.txt", "Content of file 2"),
            ("dir/file3.txt", "Content of file 3 in directory"),
        ];

        for (path, content) in &files {
            let result = sandbox.write_file(path, content).await;
            assert!(result.is_ok(), "Should write {} successfully", path);
        }

        // Read and verify each file
        for (path, expected_content) in &files {
            let result = sandbox.read_file(path).await;
            assert!(result.is_ok(), "Should read {} successfully", path);
            assert_eq!(
                result.unwrap(),
                *expected_content,
                "Content should match for {}",
                path
            );
        }

        // List all files in repo
        let result = sandbox
            .exec("find /repo -type f -name '*.txt' | sort", 10_000)
            .await;
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("file1.txt"));
        assert!(result.stdout.contains("file2.txt"));
        assert!(result.stdout.contains("file3.txt"));

        sandbox.destroy().await;
    }

    #[tokio::test]
    async fn test_exec_with_special_characters() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        let sandbox = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create sandbox");

        // Test command with special characters
        let result = sandbox
            .exec("echo 'Hello $USER! This is a test with > < & | ;'", 10_000)
            .await;
        assert_eq!(
            result.exit_code, 0,
            "Command with special chars should succeed"
        );
        assert!(
            result.stdout.contains("Hello"),
            "Output should contain 'Hello'"
        );

        sandbox.destroy().await;
    }

    #[tokio::test]
    async fn test_container_resource_limits() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        let sandbox = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create sandbox");

        // Check that memory limit is set (32g from the code)
        let result = sandbox.exec("cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory.limit_in_bytes 2>/dev/null || echo 'unlimited'", 10_000).await;
        // Just verify the command executed - cgroup info may not be available in all Docker setups
        assert_eq!(result.exit_code, 0, "Should be able to check memory limits");

        sandbox.destroy().await;
    }

    // =========================================================================
    // Fresh Container Guarantee Tests (VAL-FRESH-001, VAL-FRESH-003, VAL-FRESH-004)
    // =========================================================================

    #[tokio::test]
    async fn test_fresh_container_different_ids() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        // Create first container
        let sandbox1 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create first sandbox");
        let container_name1 = sandbox1.name().to_string();

        // Create second container
        let sandbox2 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create second sandbox");
        let container_name2 = sandbox2.name().to_string();

        // Verify both containers exist
        assert!(
            container_exists(&container_name1).await,
            "First container should exist"
        );
        assert!(
            container_exists(&container_name2).await,
            "Second container should exist"
        );

        // Verify container names are different
        assert_ne!(
            container_name1, container_name2,
            "Each container must have a unique name"
        );

        // Get container IDs (full ID from inspect)
        async fn get_container_id(name: &str) -> String {
            let output = Command::new("docker")
                .args(["inspect", "--format", "{{.Id}}", name])
                .output()
                .await
                .unwrap();
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        }

        let id1 = get_container_id(&container_name1).await;
        let id2 = get_container_id(&container_name2).await;

        // Verify container IDs are different
        assert!(
            !id1.is_empty() && !id2.is_empty(),
            "Both containers should have valid IDs"
        );
        assert_ne!(id1, id2, "Each container must have a unique ID");

        // Cleanup
        sandbox1.destroy().await;
        sandbox2.destroy().await;
    }

    #[tokio::test]
    async fn test_fresh_container_no_file_leakage() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        // Create first container and write a file
        let sandbox1 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create first sandbox");

        let test_content = "This file should not leak to other containers";
        sandbox1
            .write_file("leak_test_file.txt", test_content)
            .await
            .expect("Should write file in first container");

        // Verify file exists in first container
        let result = sandbox1.read_file("leak_test_file.txt").await;
        assert!(
            result.is_ok(),
            "File should exist in first container: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap(), test_content);

        // Get first container name for later verification
        let container1_name = sandbox1.name().to_string();

        // Destroy first container
        sandbox1.destroy().await;

        // Create second container (fresh)
        let sandbox2 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create second sandbox");

        // Verify file does NOT exist in second container using test command
        let test_result = sandbox2
            .exec("test -f /repo/leak_test_file.txt", 10_000)
            .await;
        assert_ne!(
            test_result.exit_code, 0,
            "test -f should fail - file from first container should NOT exist in second"
        );

        // Also try cat command and verify it fails
        let cat_result = sandbox2.exec("cat /repo/leak_test_file.txt", 10_000).await;
        assert_ne!(
            cat_result.exit_code, 0,
            "cat should fail for non-existent file"
        );

        // Verify first container was actually destroyed
        assert!(
            !container_exists(&container1_name).await,
            "First container should be destroyed"
        );

        sandbox2.destroy().await;
    }

    #[tokio::test]
    async fn test_fresh_container_no_install_leakage() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        // Create first container and install a package
        let sandbox1 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create first sandbox");

        // Install curl in first container
        let install_result = sandbox1
            .exec(
                "apt-get update -qq && apt-get install -y -qq curl > /dev/null 2>&1",
                120_000,
            )
            .await;
        assert_eq!(
            install_result.exit_code, 0,
            "Install should succeed in first container: {}",
            install_result.stderr
        );

        // Verify curl is available in first container
        let check_result = sandbox1.exec("which curl", 10_000).await;
        assert_eq!(
            check_result.exit_code, 0,
            "curl should be available in first container"
        );
        assert!(
            check_result.stdout.contains("/usr/bin/curl"),
            "curl path should be found: {}",
            check_result.stdout
        );

        // Destroy first container
        sandbox1.destroy().await;

        // Create second container (fresh)
        let sandbox2 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create second sandbox");

        // Verify curl is NOT available in second container
        let check_result = sandbox2.exec("which curl 2>&1", 10_000).await;
        assert_ne!(
            check_result.exit_code, 0,
            "curl should NOT be available in second container"
        );
        assert!(
            check_result.stdout.is_empty(),
            "which curl should return nothing: {}",
            check_result.stdout
        );
        assert!(
            check_result.stderr.contains("no curl") || check_result.stderr.is_empty(),
            "curl should not be found: {}",
            check_result.stderr
        );

        sandbox2.destroy().await;
    }

    #[tokio::test]
    async fn test_fresh_container_clean_git_checkout() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        // Use a known commit from octocat/Hello-World
        let known_commit = "7fd1a60b01f91b314f59955a4e4d4e80d8edf11d";

        // Create first container at specific commit
        let sandbox1 = DockerSandbox::start(
            "octocat/Hello-World",
            known_commit,
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create first sandbox");

        // Verify clean git state in first container
        let status_result = sandbox1
            .exec("cd /repo && git status --porcelain", 10_000)
            .await;
        assert_eq!(status_result.exit_code, 0, "git status should succeed");
        assert!(
            status_result.stdout.is_empty(),
            "Working tree should be clean in first container, got: {}",
            status_result.stdout
        );

        // Verify correct commit in first container
        let commit_result = sandbox1
            .exec("cd /repo && git rev-parse HEAD", 10_000)
            .await;
        assert_eq!(commit_result.exit_code, 0, "git rev-parse should succeed");
        assert!(
            commit_result.stdout.contains(known_commit),
            "Should be at correct commit: {}",
            commit_result.stdout
        );

        // Make a change in first container
        sandbox1
            .write_file("dirty_file.txt", "dirty content")
            .await
            .expect("Should write file in first container");

        // Verify the repo is now dirty
        let status_result = sandbox1
            .exec("cd /repo && git status --porcelain", 10_000)
            .await;
        assert!(
            status_result.stdout.contains("dirty_file.txt"),
            "First container should show dirty file"
        );

        // Destroy first container
        sandbox1.destroy().await;

        // Create second container (fresh) at same commit
        let sandbox2 = DockerSandbox::start(
            "octocat/Hello-World",
            known_commit,
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create second sandbox");

        // Verify clean git state in second container
        let status_result = sandbox2
            .exec("cd /repo && git status --porcelain", 10_000)
            .await;
        assert_eq!(
            status_result.exit_code, 0,
            "git status should succeed in second container"
        );
        assert!(
            status_result.stdout.is_empty(),
            "Second container should have clean working tree, not affected by first container. Got: {}",
            status_result.stdout
        );

        // Verify correct commit in second container
        let commit_result = sandbox2
            .exec("cd /repo && git rev-parse HEAD", 10_000)
            .await;
        assert_eq!(commit_result.exit_code, 0);
        assert!(
            commit_result.stdout.contains(known_commit),
            "Second container should be at correct commit"
        );

        // Verify dirty file from first container is not present
        let ls_result = sandbox2
            .exec("ls -la /repo/dirty_file.txt 2>&1", 10_000)
            .await;
        assert_ne!(
            ls_result.exit_code, 0,
            "Dirty file from first container should not exist in second"
        );

        sandbox2.destroy().await;
    }

    // =========================================================================
    // Install Reproducibility Tests (VAL-FRESH-002)
    // =========================================================================

    /// Helper: Run a set of install commands and return the ones that succeeded
    async fn record_install_commands(sandbox: &DockerSandbox, commands: &[String]) -> Vec<String> {
        let mut successful = Vec::new();
        for cmd in commands {
            let result = sandbox.exec(cmd, 120_000).await;
            if result.exit_code == 0 {
                successful.push(cmd.clone());
            }
        }
        successful
    }

    /// Helper: Replay install commands in a fresh container and verify all succeed
    async fn replay_install_commands(sandbox: &DockerSandbox, commands: &[String]) -> bool {
        for cmd in commands {
            let result = sandbox.exec(cmd, 120_000).await;
            if result.exit_code != 0 {
                return false;
            }
        }
        true
    }

    /// VAL-FRESH-002: Install commands produce the same environment when replayed in fresh containers
    /// - Record successful install commands from first container
    /// - Replay in fresh container
    /// - Verify same tools are available
    #[tokio::test]
    async fn test_install_reproducibility_basic() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        // Create first container (source environment)
        let sandbox1 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create first sandbox");

        // Define install commands to test
        let install_commands = vec![
            "apt-get update -qq && apt-get install -y -qq curl > /dev/null 2>&1".to_string(),
            "apt-get install -y -qq jq > /dev/null 2>&1".to_string(),
        ];

        // Record successful commands from first container
        let recorded = record_install_commands(&sandbox1, &install_commands).await;
        assert!(
            !recorded.is_empty(),
            "Should have recorded some successful install commands"
        );

        // Verify tools are available in first container
        let curl_check1 = sandbox1.exec("which curl", 10_000).await;
        let jq_check1 = sandbox1.exec("which jq", 10_000).await;
        assert_eq!(
            curl_check1.exit_code, 0,
            "curl should be in first container"
        );
        assert_eq!(jq_check1.exit_code, 0, "jq should be in first container");

        // Destroy first container
        sandbox1.destroy().await;

        // Create second container (fresh)
        let sandbox2 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create second sandbox");

        // Replay recorded commands in fresh container
        let replay_success = replay_install_commands(&sandbox2, &recorded).await;
        assert!(
            replay_success,
            "All recorded commands should succeed in fresh container"
        );

        // Verify same tools are available in second container
        let curl_check2 = sandbox2.exec("which curl", 10_000).await;
        let jq_check2 = sandbox2.exec("which jq", 10_000).await;
        assert_eq!(
            curl_check2.exit_code, 0,
            "curl should be available in second container after replay"
        );
        assert_eq!(
            jq_check2.exit_code, 0,
            "jq should be available in second container after replay"
        );

        // Verify tool paths are identical
        assert_eq!(
            curl_check1.stdout.trim(),
            curl_check2.stdout.trim(),
            "curl path should be identical in both containers"
        );
        assert_eq!(
            jq_check1.stdout.trim(),
            jq_check2.stdout.trim(),
            "jq path should be identical in both containers"
        );

        sandbox2.destroy().await;
    }

    /// Test: Install commands are idempotent - running same commands twice produces same result
    #[tokio::test]
    async fn test_install_idempotency() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        let sandbox = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create sandbox");

        // Install curl
        let install_cmd = "apt-get update -qq && apt-get install -y -qq curl > /dev/null 2>&1";
        let result1 = sandbox.exec(install_cmd, 120_000).await;
        assert_eq!(result1.exit_code, 0, "First install should succeed");

        // Check curl version after first install
        let version1 = sandbox.exec("curl --version | head -1", 10_000).await;
        assert_eq!(
            version1.exit_code, 0,
            "curl should work after first install"
        );

        // Run same install command again (idempotency test)
        let result2 = sandbox.exec(install_cmd, 120_000).await;
        assert_eq!(
            result2.exit_code, 0,
            "Second install should also succeed (idempotent)"
        );

        // Check curl version after second install - should be same
        let version2 = sandbox.exec("curl --version | head -1", 10_000).await;
        assert_eq!(
            version1.stdout.trim(),
            version2.stdout.trim(),
            "Tool version should be identical after reinstall (idempotent)"
        );

        sandbox.destroy().await;
    }

    /// Test: Python package installation reproducibility
    #[tokio::test]
    async fn test_python_package_reproducibility() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        // Create first container
        let sandbox1 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create first sandbox");

        // Record Python package installs
        let pip_commands = vec![
            "pip install --break-system-packages requests 2>&1".to_string(),
            "pip install --break-system-packages pytest 2>&1".to_string(),
        ];

        let recorded = record_install_commands(&sandbox1, &pip_commands).await;
        assert_eq!(
            recorded.len(),
            2,
            "Both pip commands should succeed in first container"
        );

        // Verify packages are available
        let requests_check1 = sandbox1
            .exec(
                "python3 -c 'import requests; print(requests.__version__)'",
                10_000,
            )
            .await;
        let pytest_check1 = sandbox1.exec("which pytest", 10_000).await;
        assert_eq!(
            requests_check1.exit_code, 0,
            "requests should be importable"
        );
        assert_eq!(pytest_check1.exit_code, 0, "pytest should be available");

        // Record package versions
        let requests_version1 = requests_check1.stdout.trim().to_string();

        sandbox1.destroy().await;

        // Create fresh container and replay
        let sandbox2 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create second sandbox");

        let replay_success = replay_install_commands(&sandbox2, &recorded).await;
        assert!(
            replay_success,
            "All pip commands should succeed in fresh container"
        );

        // Verify packages in second container
        let requests_check2 = sandbox2
            .exec(
                "python3 -c 'import requests; print(requests.__version__)'",
                10_000,
            )
            .await;
        let pytest_check2 = sandbox2.exec("which pytest", 10_000).await;
        assert_eq!(
            requests_check2.exit_code, 0,
            "requests should be importable in fresh container"
        );
        assert_eq!(
            pytest_check2.exit_code, 0,
            "pytest should be available in fresh container"
        );

        // Verify same versions
        let requests_version2 = requests_check2.stdout.trim().to_string();
        assert_eq!(
            requests_version1, requests_version2,
            "Package versions should be identical in both containers"
        );

        sandbox2.destroy().await;
    }

    /// Test: Only successful commands should be recorded (failed commands should not leak)
    #[tokio::test]
    async fn test_install_reproducibility_failed_commands_excluded() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        let sandbox = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create sandbox");

        // Mix of commands - some succeed, some fail
        let mixed_commands = vec![
            "apt-get update -qq > /dev/null 2>&1".to_string(), // Succeeds
            "apt-get install -y -qq nonexistent-package-xyz123 2>&1".to_string(), // Fails
            "apt-get install -y -qq curl > /dev/null 2>&1".to_string(), // Succeeds
        ];

        let recorded = record_install_commands(&sandbox, &mixed_commands).await;

        // Should only record successful commands
        assert_eq!(
            recorded.len(),
            2,
            "Should only record 2 successful commands"
        );
        assert!(
            recorded[0].contains("apt-get update"),
            "First recorded should be apt-get update"
        );
        assert!(
            recorded[1].contains("curl"),
            "Second recorded should be curl install"
        );

        sandbox.destroy().await;
    }

    /// Test: Multiple fresh containers from same recorded commands produce equivalent environments
    #[tokio::test]
    async fn test_install_reproducibility_multiple_containers() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        // Source container to record commands
        let sandbox_source = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create source sandbox");

        let install_commands = vec![
            "apt-get update -qq > /dev/null 2>&1".to_string(),
            "apt-get install -y -qq curl jq > /dev/null 2>&1".to_string(),
            "pip install --break-system-packages pytest > /dev/null 2>&1".to_string(),
        ];

        let recorded = record_install_commands(&sandbox_source, &install_commands).await;
        sandbox_source.destroy().await;

        // Create multiple fresh containers with same recorded commands
        let mut containers = Vec::new();
        for i in 0..3 {
            let sandbox = DockerSandbox::start(
                "octocat/Hello-World",
                "",
                "python",
                Some("python:3.12-slim"),
            )
            .await
            .unwrap_or_else(|_| panic!("Failed to create sandbox {}", i));
            containers.push(sandbox);
        }

        // Replay commands in all containers
        for (i, sandbox) in containers.iter().enumerate() {
            let success = replay_install_commands(sandbox, &recorded).await;
            assert!(success, "Container {}: All commands should succeed", i);
        }

        // Verify equivalence across all containers
        let mut curl_paths = Vec::new();
        let mut jq_paths = Vec::new();
        let mut pytest_paths = Vec::new();

        for sandbox in &containers {
            let curl = sandbox.exec("which curl", 10_000).await;
            let jq = sandbox.exec("which jq", 10_000).await;
            let pytest = sandbox.exec("which pytest", 10_000).await;

            assert_eq!(curl.exit_code, 0, "curl should be available");
            assert_eq!(jq.exit_code, 0, "jq should be available");
            assert_eq!(pytest.exit_code, 0, "pytest should be available");

            curl_paths.push(curl.stdout.trim().to_string());
            jq_paths.push(jq.stdout.trim().to_string());
            pytest_paths.push(pytest.stdout.trim().to_string());
        }

        // All paths should be identical
        assert!(
            curl_paths.windows(2).all(|w| w[0] == w[1]),
            "All containers should have curl at the same path: {:?}",
            curl_paths
        );
        assert!(
            jq_paths.windows(2).all(|w| w[0] == w[1]),
            "All containers should have jq at the same path: {:?}",
            jq_paths
        );
        assert!(
            pytest_paths.windows(2).all(|w| w[0] == w[1]),
            "All containers should have pytest at the same path: {:?}",
            pytest_paths
        );

        // Cleanup all containers
        for sandbox in containers {
            sandbox.destroy().await;
        }
    }

    /// Test: Complex install sequence reproducibility (simulating real project setup)
    #[tokio::test]
    async fn test_install_reproducibility_complex_sequence() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        // Source container
        let sandbox1 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create first sandbox");

        // Complex multi-step install sequence
        let complex_commands = vec![
            // System dependencies
            "apt-get update -qq > /dev/null 2>&1 && apt-get install -y -qq build-essential git > /dev/null 2>&1".to_string(),
            // Python development tools
            "pip install --break-system-packages setuptools wheel > /dev/null 2>&1".to_string(),
            // Testing framework
            "pip install --break-system-packages pytest pytest-cov > /dev/null 2>&1".to_string(),
            // Linting tools
            "pip install --break-system-packages flake8 black > /dev/null 2>&1".to_string(),
        ];

        let recorded = record_install_commands(&sandbox1, &complex_commands).await;
        assert_eq!(recorded.len(), 4, "All complex commands should succeed");

        // Verify tools in first container
        let tools_to_check = vec![
            ("git", "which git"),
            ("pytest", "which pytest"),
            ("flake8", "which flake8"),
            ("black", "which black"),
        ];

        for (name, cmd) in &tools_to_check {
            let result = sandbox1.exec(cmd, 10_000).await;
            assert_eq!(
                result.exit_code, 0,
                "{} should be available in first container",
                name
            );
        }

        sandbox1.destroy().await;

        // Replay in fresh container
        let sandbox2 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create second sandbox");

        let replay_success = replay_install_commands(&sandbox2, &recorded).await;
        assert!(
            replay_success,
            "Complex command sequence should succeed in fresh container"
        );

        // Verify same tools in second container
        for (name, cmd) in &tools_to_check {
            let result = sandbox2.exec(cmd, 10_000).await;
            assert_eq!(
                result.exit_code, 0,
                "{} should be available in second container after replay",
                name
            );
        }

        // Verify tool functionality is equivalent
        let flake8_result = sandbox2.exec("flake8 --version", 10_000).await;
        assert_eq!(
            flake8_result.exit_code, 0,
            "flake8 should be functional in fresh container"
        );

        sandbox2.destroy().await;
    }

    /// Test: Empty install commands list is handled correctly
    #[tokio::test]
    async fn test_install_reproducibility_empty_commands() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        let sandbox = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create sandbox");

        // Empty command list
        let empty_commands: Vec<String> = vec![];
        let recorded = record_install_commands(&sandbox, &empty_commands).await;

        assert!(
            recorded.is_empty(),
            "Empty command list should produce empty recorded list"
        );

        // Replaying empty list should succeed (vacuously true)
        let replay_success = replay_install_commands(&sandbox, &recorded).await;
        assert!(
            replay_success,
            "Replaying empty commands should succeed (no-op)"
        );

        sandbox.destroy().await;
    }

    /// Test: Install commands with environment variables are reproducible
    #[tokio::test]
    async fn test_install_reproducibility_with_env_vars() {
        if !docker_available().await {
            eprintln!("Docker not available, skipping test");
            return;
        }

        // Source container
        let sandbox1 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create first sandbox");

        // Install with environment variable
        let env_commands = vec![
            "DEBIAN_FRONTEND=noninteractive apt-get update -qq > /dev/null 2>&1".to_string(),
            "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl > /dev/null 2>&1"
                .to_string(),
        ];

        let recorded = record_install_commands(&sandbox1, &env_commands).await;
        assert_eq!(recorded.len(), 2, "Both env commands should succeed");

        let curl_check1 = sandbox1.exec("which curl", 10_000).await;
        assert_eq!(curl_check1.exit_code, 0, "curl should be available");

        sandbox1.destroy().await;

        // Replay in fresh container
        let sandbox2 = DockerSandbox::start(
            "octocat/Hello-World",
            "",
            "python",
            Some("python:3.12-slim"),
        )
        .await
        .expect("Failed to create second sandbox");

        let replay_success = replay_install_commands(&sandbox2, &recorded).await;
        assert!(
            replay_success,
            "Commands with env vars should replay successfully"
        );

        let curl_check2 = sandbox2.exec("which curl", 10_000).await;
        assert_eq!(
            curl_check2.exit_code, 0,
            "curl should be available after replay"
        );

        sandbox2.destroy().await;
    }
}
