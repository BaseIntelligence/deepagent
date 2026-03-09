# Architecture

Key architectural decisions and patterns discovered in swe-forge.

## Core Components

### 1. DockerSandbox (src/swe/docker_sandbox.rs)
- Ephemeral container per task
- Port allocation via atomic counter (10000-60000)
- Tool server for HTTP-based tool execution
- Automatic cleanup on Drop

### 2. Harness (src/swe/harness.rs)
- Entry point: `run_harness()` discovers and processes tasks
- Per-task evaluation: `evaluate_task()`
- Sanity checks: f2p must fail, p2p must pass on base commit
- Results: HarnessStatus enum (Resolved, Unresolved, SanityFail, etc.)

### 3. WorkspaceValidator (src/swe/workspace_validator.rs)
- Pre-export validation with fresh-container replay
- Up to 5 fresh-container cycles
- LLM-based failure diagnosis
- Install agent for discovering dependencies

### 4. TestGenerator (src/swe/test_generator.rs)
- Agentic multi-turn test generation (max 200 turns)
- Dual-commit validation (base + PR commit)
- Anti-hardcoding: rejects string-matching tests

## Key Data Structures

### SweTask
```rust
pub struct SweTask {
    pub id: String,
    pub repo: String,
    pub base_commit: String,
    pub patch: String,
    pub prompt: String,
    pub fail_to_pass: Vec<String>,
    pub pass_to_pass: Vec<String>,
    pub install_config: HashMap<String, String>,
    pub meta: HashMap<String, String>,
}
```

### HarnessResult
```rust
pub struct HarnessResult {
    pub task_id: String,
    pub status: HarnessStatus,
    pub sanity_check: bool,
    pub fail_to_pass: Vec<TestResult>,
    pub pass_to_pass: Vec<TestResult>,
    pub error: Option<String>,
    pub container_id: Option<String>,
}
```

## Test Semantics

Critical validation rules:

1. **On base commit (before fix):**
   - fail_to_pass tests MUST fail (exit != 0)
   - pass_to_pass tests MUST pass (exit == 0)

2. **After valid patch (after fix):**
   - fail_to_pass tests MUST pass (exit == 0)
   - pass_to_pass tests MUST still pass (exit == 0)

3. **Fresh container guarantee:**
   - Each cycle creates a brand new container
   - No state shared between cycles
   - Install commands replayed from scratch

## Docker Patterns

### Container Naming
```rust
format!("swe-harness-{}", task_id.replace('/', "-").replace(' ', "_"))
format!("swe-mine-{}-{}", repo_safe_name, timestamp_suffix)
```

### Cleanup Pattern
```rust
async fn docker_rm(container: &str) {
    let _ = Command::new("docker")
        .args(["rm", "-f", container])
        .status()
        .await;
}

// In Drop implementation
impl Drop for DockerSandbox {
    fn drop(&mut self) {
        let name = self.container_name.clone();
        std::thread::spawn(move || {
            let _ = std::process::Command::new("docker")
                .args(["rm", "-f", &name])
                .status();
        });
    }
}
```

## Resource Limits

Difficulty-based limits (src/docker/resources.rs):

| Difficulty | PIDs | Storage | Network |
|------------|------|---------|---------|
| easy | 100 | 1GB | None |
| medium | 200 | 2GB | Internal |
| hard | 500 | 5GB | Internal |

Memory: 32GB for all difficulty levels.

## Error Handling Patterns

- Library code uses typed errors (thiserror)
- CLI code uses anyhow::Result
- Docker errors are logged but don't always fail the task
- Timeout handling for all container operations
