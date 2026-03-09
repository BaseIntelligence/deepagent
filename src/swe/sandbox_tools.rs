//! Shared tool definitions and dispatch for Docker sandbox agents.
//!
//! Both the test generator and the install agent need shell, read_file,
//! list_dir, grep_files, search_files tools. This module centralises
//! their definitions, argument types, and execution logic.

use crate::llm::{ToolCallInfo, ToolDefinition};
use crate::swe::docker_sandbox::DockerSandbox;
use crate::swe::validate_file_path;

// ── Tool definitions ──────────────────────────────────────────────────────

pub fn shell_tool() -> ToolDefinition {
    ToolDefinition::function(
        "shell",
        "Execute a shell command in the repository. Returns stdout, stderr, and exit code.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 30000)"
                }
            },
            "required": ["command"]
        }),
    )
}

pub fn read_file_tool() -> ToolDefinition {
    ToolDefinition::function(
        "read_file",
        "Read file contents with line numbers. Supports offset/limit for pagination.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to repo root or absolute)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line offset to start from (0-based, default: 0)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (omit for all)"
                }
            },
            "required": ["file_path"]
        }),
    )
}

pub fn list_dir_tool() -> ToolDefinition {
    ToolDefinition::function(
        "list_dir",
        "List directory contents. Skips .git, node_modules, __pycache__, etc.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Path to directory (default: '.' = repo root)"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List recursively (default: false)"
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (default: false)"
                }
            },
            "required": []
        }),
    )
}

pub fn grep_files_tool() -> ToolDefinition {
    ToolDefinition::function(
        "grep_files",
        "Search file contents with regex pattern. Returns matching lines with file paths and line numbers.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.py', '*.js')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: repo root)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of matching lines (default: 100)"
                }
            },
            "required": ["pattern"]
        }),
    )
}

pub fn search_files_tool() -> ToolDefinition {
    ToolDefinition::function(
        "search_files",
        "Find files matching a glob pattern. Skips .git, node_modules, etc.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. '*.py', '**/*.test.js', 'src/**/*.rs')"
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search from (default: repo root)"
                }
            },
            "required": ["pattern"]
        }),
    )
}

// ── Argument types ────────────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
pub struct ShellArgs {
    pub command: String,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
}
fn default_timeout() -> u64 {
    30_000
}

// ── Tool result enum ──────────────────────────────────────────────────────

pub enum ToolOutput {
    Text(String),
    Error(String),
}

// ── Dispatch for the exploration tools (shell, read_file, etc.) ───────────

/// Execute a tool call against a Docker sandbox.
/// Returns the textual output to feed back to the LLM.
pub async fn dispatch_tool(
    tc: &ToolCallInfo,
    sandbox: &DockerSandbox,
    task_id: &str,
    turn: usize,
) -> ToolOutput {
    match tc.function.name.as_str() {
        "shell" => {
            let args: ShellArgs = match serde_json::from_str(&tc.function.arguments) {
                Ok(a) => a,
                Err(e) => return ToolOutput::Error(format!("Invalid shell args: {}", e)),
            };
            let result = sandbox.exec(&args.command, args.timeout_ms).await;
            tracing::debug!(
                task_id = task_id, turn = turn,
                cmd = %args.command, exit = result.exit_code,
                "Agent shell"
            );
            ToolOutput::Text(format!(
                "exit_code: {}\nstdout:\n{}\nstderr:\n{}",
                result.exit_code,
                truncate_utf8(&result.stdout, 3000),
                truncate_utf8(&result.stderr, 1500),
            ))
        }
        "read_file" | "list_dir" | "grep_files" | "search_files" => {
            let tool_name = tc.function.name.as_str();
            let args_json = &tc.function.arguments;

            let output = if sandbox.has_tool_server() {
                let result = sandbox.tool_request(tool_name, args_json).await;
                let server_down = result.exit_code != 0
                    && result.stdout.is_empty()
                    && (result.stderr.contains("Connection refused")
                        || result.stderr.contains("URLError")
                        || result.stderr.contains("Tool request error"));
                if server_down {
                    shell_fallback(sandbox, tool_name, args_json).await
                } else {
                    parse_tool_response(&result)
                }
            } else {
                shell_fallback(sandbox, tool_name, args_json).await
            };

            tracing::debug!(
                task_id = task_id,
                turn = turn,
                tool = tool_name,
                "Agent tool call"
            );
            ToolOutput::Text(truncate_utf8(&output, 4000).to_string())
        }
        other => ToolOutput::Error(format!("Unknown tool: {}", other)),
    }
}

// ── Helpers (moved from test_generator) ───────────────────────────────────

pub fn parse_tool_response(result: &crate::swe::docker_sandbox::SandboxOutput) -> String {
    if !result.stdout.is_empty() {
        match serde_json::from_str::<serde_json::Value>(result.stdout.trim()) {
            Ok(v) => {
                if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                    err.to_string()
                } else if let Some(out) = v.get("output").and_then(|o| o.as_str()) {
                    let mut s = out.to_string();
                    if let Some(total) = v.get("total_lines").and_then(|t| t.as_u64()) {
                        if v.get("truncated")
                            .and_then(|t| t.as_bool())
                            .unwrap_or(false)
                        {
                            s.push_str(&format!(
                                "\n\n[Showing {}/{} lines. Use offset/limit to see more.]",
                                v.get("shown_lines").and_then(|sl| sl.as_u64()).unwrap_or(0),
                                total
                            ));
                        } else {
                            s.push_str(&format!("\n\n[{} total lines]", total));
                        }
                    }
                    s
                } else {
                    result.stdout.clone()
                }
            }
            Err(_) => result.stdout.clone(),
        }
    } else if !result.stderr.is_empty() {
        format!("Error: {}", truncate_utf8(&result.stderr, 1500))
    } else {
        "No output".to_string()
    }
}

fn sanitize_shell_arg(s: &str) -> String {
    s.replace('\'', "'\\''")
}

pub async fn shell_fallback(sandbox: &DockerSandbox, tool_name: &str, args_json: &str) -> String {
    let args: serde_json::Value = match serde_json::from_str(args_json) {
        Ok(v) => v,
        Err(e) => return format!("Invalid tool args: {}", e),
    };

    match tool_name {
        "read_file" => {
            let file_path = args.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
            if file_path.is_empty() {
                return "Error: missing file_path".to_string();
            }
            if let Err(e) = validate_file_path(file_path) {
                return format!("Error: invalid file_path: {}", e);
            }
            let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0);
            let limit = args.get("limit").and_then(|v| v.as_u64());
            let cmd = match limit {
                Some(lim) => format!(
                    "awk 'NR>{} && NR<={}{{print NR\": \"$0}}' '{}'",
                    offset,
                    offset + lim,
                    file_path
                ),
                None => format!("awk '{{print NR\": \"$0}}' '{}'", file_path),
            };
            let result = sandbox.exec(&cmd, 10_000).await;
            if result.exit_code != 0 {
                format!("Error reading file: {}", result.stderr)
            } else if result.stdout.is_empty() {
                "(empty file)".to_string()
            } else {
                result.stdout
            }
        }
        "list_dir" => {
            let dir = args
                .get("directory_path")
                .and_then(|v| v.as_str())
                .unwrap_or(".");
            if dir != "." {
                if let Err(e) = validate_file_path(dir) {
                    return format!("Error: invalid directory_path: {}", e);
                }
            }
            let cmd = format!("ls -la '{}'", sanitize_shell_arg(dir));
            let result = sandbox.exec(&cmd, 10_000).await;
            if result.exit_code != 0 {
                format!("Error listing directory: {}", result.stderr)
            } else {
                result.stdout
            }
        }
        "grep_files" => {
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            if pattern.is_empty() {
                return "Error: missing pattern".to_string();
            }
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            if path != "." {
                if let Err(e) = validate_file_path(path) {
                    return format!("Error: invalid path: {}", e);
                }
            }
            let include = args.get("include").and_then(|v| v.as_str());
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(100);
            let include_arg = match include {
                Some(glob) => format!(" --include='{}'", sanitize_shell_arg(glob)),
                None => String::new(),
            };
            let cmd = format!(
                "grep -rn --color=never{} '{}' '{}' | head -n {}",
                include_arg,
                sanitize_shell_arg(pattern),
                sanitize_shell_arg(path),
                limit
            );
            let result = sandbox.exec(&cmd, 30_000).await;
            if result.stdout.is_empty() {
                "No matches found.".to_string()
            } else {
                result.stdout
            }
        }
        "search_files" => {
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            if pattern.is_empty() {
                return "Error: missing pattern".to_string();
            }
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            if path != "." {
                if let Err(e) = validate_file_path(path) {
                    return format!("Error: invalid path: {}", e);
                }
            }
            let cmd = format!(
                "find '{}' -name '{}' -not -path '*/.git/*' -not -path '*/node_modules/*' | sort",
                sanitize_shell_arg(path),
                sanitize_shell_arg(pattern)
            );
            let result = sandbox.exec(&cmd, 30_000).await;
            if result.stdout.is_empty() {
                format!("No files matching '{}'", pattern)
            } else {
                result.stdout
            }
        }
        _ => format!("Unknown tool: {}", tool_name),
    }
}

pub fn truncate_utf8(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    &s[..end]
}
