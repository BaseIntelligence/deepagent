---
name: rust-test-worker
description: Creates Rust unit and integration tests for swe-forge components with Docker infrastructure
---

# rust-test-worker

## When to Use This Skill

Use this skill for features that require:
- Creating Rust unit tests with `#[cfg(test)] mod tests`
- Creating integration tests using Docker containers
- Testing swe-forge components (harness, workspace_validator, docker_sandbox)
- Testing resource limits and Docker infrastructure

## Work Procedure

### 1. Understanding Requirements (Read-Only)

Read the feature description and understand:
- Which component needs testing (docker_sandbox, harness, validator, etc.)
- What behavior needs verification
- What the test should validate

Key files to read:
- The source file being tested (e.g., `src/swe/docker_sandbox.rs`)
- Existing tests in that file for patterns
- Related modules for context

### 2. Planning Test Coverage

Identify test scenarios:
- Happy path: Normal operation
- Error cases: Invalid inputs, failures
- Edge cases: Empty inputs, boundaries
- Integration: Multiple components working together

For Docker tests:
- Container lifecycle (create, use, destroy)
- Command execution (success and failure cases)
- File operations (write, read, verify)
- Cleanup verification (no lingering resources)

### 3. Writing Tests (TDD Pattern)

Follow this order:
1. Write failing test first (RED)
2. Run test to confirm it fails as expected
3. Implement or verify implementation (GREEN)
4. Refactor if needed

Test structure:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    // Additional imports as needed

    #[tokio::test]  // For async tests
    async fn test_description() {
        // Arrange: Setup test state
        let temp_dir = tempfile::tempdir().unwrap();
        
        // Act: Execute the code under test
        let result = function_call().await;
        
        // Assert: Verify expected behavior
        assert!(result.is_ok());
    }
}
```

### 4. Docker Test Safety

Critical for Docker tests:
- Always use unique container names (timestamp suffix)
- Always clean up containers in test cleanup or Drop
- Use `docker rm -f` to force removal
- Handle cleanup errors gracefully (don't fail test on cleanup)

Example pattern:
```rust
async fn test_container_cleanup() {
    let container_name = format!("test-{}", timestamp());
    
    // Create and use container...
    
    // Explicit cleanup
    let _ = Command::new("docker")
        .args(["rm", "-f", &container_name])
        .status()
        .await;
    
    // Verify cleanup
    let output = Command::new("docker")
        .args(["ps", "-a", "--filter", &format!("name={}", container_name)])
        .output()
        .await
        .unwrap();
    assert!(output.stdout.is_empty());
}
```

### 5. Running Tests

Run tests incrementally:
```bash
# Run the specific test module
cargo test --lib <module_name>

# Run with output for debugging
cargo test --lib <module_name> -- --nocapture

# Run release mode (faster for integration tests)
cargo test --release --lib <module_name>
```

### 6. Verification

Verify:
- All new tests pass
- Existing tests still pass (no regressions)
- No Docker containers leaked (`docker ps -a`)
- Code compiles without warnings

### 7. Manual Verification (For Docker Tests)

When tests involve Docker:
- Run `docker ps -a` before and after tests
- Verify no test containers remain
- Check container logs if tests fail

## Example Handoff

```json
{
  "salientSummary": "Created comprehensive Docker sandbox integration tests covering container creation, command execution, file operations, and cleanup verification. All 12 new tests pass.",
  "whatWasImplemented": "Added integration tests to src/swe/docker_sandbox.rs: test_container_lifecycle (create/destroy), test_command_execution (exit codes), test_file_operations (write/read), test_cleanup_on_drop (resource cleanup), and 8 additional edge case tests. Tests use tempfile for isolation and verify no Docker containers leak.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "cargo test --lib docker_sandbox", "exitCode": 0, "observation": "12 tests passed, 0 failures"},
      {"command": "docker ps -a | grep swe-mine", "exitCode": 1, "observation": "No lingering containers"},
      {"command": "cargo clippy --lib -- -D warnings", "exitCode": 0, "observation": "No warnings"}
    ],
    "interactiveChecks": [],
    "testsAdded": [
      {
        "file": "src/swe/docker_sandbox.rs",
        "cases": [
          {"name": "test_container_creation", "verifies": "Docker container starts correctly with proper config"},
          {"name": "test_command_execution", "verifies": "Shell commands execute with correct exit codes"},
          {"name": "test_file_write_read", "verifies": "Files can be written and read in container"},
          {"name": "test_cleanup_on_drop", "verifies": "Containers are destroyed when sandbox is dropped"}
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

Return if:
- The source code being tested has bugs that prevent tests from passing
- Docker is not available or not working
- Tests require environment variables that aren't available
- The test design requires architectural changes to the source code
- Multiple test files need coordination (indicate need for follow-up features)
