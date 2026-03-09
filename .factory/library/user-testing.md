# User Testing Guide

Manual testing surface for swe-forge verification.

## Testing Surface

### 1. CLI Commands

The primary user interface is through cargo run:

```bash
# Harness command (main evaluation)
cargo run -- swe harness \
  --input ./hf-tasks/tasks \
  --parallel 1 \
  --json

# Mine command (task generation - not in scope for testing)
cargo run -- swe mine --help

# Fix-tasks command (repair existing tasks)
cargo run -- swe fix-tasks --help
```

### 2. Test Execution

Running the test suite:

```bash
# Unit tests
cargo test --lib

# Specific module
cargo test --lib docker_sandbox

# With output
cargo test --lib -- --nocapture

# Release mode (faster)
cargo test --release -- --test-threads=$(nproc)
```

### 3. Docker Verification

Manual Docker checks:

```bash
# Check no containers before tests
docker ps -a | grep swe-

# Check no containers after tests
docker ps -a | grep swe-

# Inspect a container
docker inspect <container_name>

# Check container logs
docker logs <container_name>
```

## Test Scenarios

### Scenario 1: Docker Infrastructure

```bash
# Verify Docker works
docker run --rm python:3.12-slim echo "Docker OK"

# Run Docker sandbox tests
cargo test --lib docker_sandbox -- --nocapture

# Verify cleanup
docker ps -a | grep swe-mine || echo "No lingering containers"
```

### Scenario 2: Harness Execution

```bash
# Pick a single task
mkdir -p /tmp/test-task
cp -r hf-tasks/tasks/<task-name> /tmp/test-task/

# Run harness on single task
cargo run -- swe harness \
  --input /tmp/test-task \
  --parallel 1 \
  --json 2>&1 | tee /tmp/harness_output.json

# Analyze results
cat /tmp/harness_output.json | jq '.results[0]'
```

### Scenario 3: Test Semantics Verification

For a specific task, verify:

1. **Base commit test behavior:**
   ```bash
   # In a fresh container
   git checkout <base_commit>
   # Run fail_to_pass command - should FAIL
   # Run pass_to_pass command - should PASS
   ```

2. **Post-patch test behavior:**
   ```bash
   # Apply patch
   git apply <patch>
   # Run fail_to_pass command - should now PASS
   # Run pass_to_pass command - should still PASS
   ```

## Expected Behaviors

### Harness Status Meanings

| Status | Meaning | When It Occurs |
|--------|---------|----------------|
| Resolved | All tests pass after agent | f2p passes, p2p passes |
| Unresolved | Some tests fail | f2p fails OR p2p fails |
| SanityFail | Test semantics wrong | f2p passes on base OR p2p fails on base |
| SetupError | Environment setup failed | Clone failed, install failed |
| AgentError | Agent execution failed | Agent timeout, agent crash |
| TestError | Test execution error | Test command invalid |

### Container Lifecycle

Expected flow:
1. Container created with unique name
2. Repo cloned and checked out
3. Dependencies installed
4. Tests executed
5. Results recorded
6. Container removed

## Isolation Notes

- Each task gets its own container
- Parallel execution uses separate containers
- Fresh-container replay creates multiple containers per task
- No shared state between containers

## Known Quirks

- Docker operations can be slow (use longer timeouts)
- --network=host means containers share host network namespace
- Container names must be unique (timestamp suffix used)
- Git checkout --force to handle any local changes
