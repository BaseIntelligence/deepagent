---
name: integration-test-worker
description: Runs end-to-end integration tests and validation on real swe-forge tasks using the harness and CLI
---

# integration-test-worker

## When to Use This Skill

Use this skill for features that require:
- Running the full swe-forge harness on real tasks
- Validating behavior with actual hf-tasks/ data
- End-to-end testing of the complete pipeline
- Benchmark and performance testing

## Work Procedure

### 1. Understanding the Test Target

Read the feature description to understand:
- What component is being validated (harness, validator, etc.)
- What real data should be used (hf-tasks/)
- What success looks like

Key files to read:
- `src/swe/harness.rs` for harness execution
- `src/cli/commands.rs` for CLI commands
- Sample workspace.yaml files in hf-tasks/

### 2. Preparation

Ensure environment is ready:
```bash
# Verify Docker is running
docker ps

# Check hf-tasks/ directory exists and has content
ls hf-tasks/tasks/ | head -5

# Verify environment variables if needed
echo $OPENROUTER_API_KEY
echo $GITHUB_TOKEN
```

### 3. Running Integration Tests

Use cargo run to execute the actual CLI:

```bash
# Run harness on hf-tasks
cargo run -- swe harness \
  --input ./hf-tasks/tasks \
  --parallel 1 \
  --json \
  2>&1 | tee harness_output.json

# Alternative: Run on subset
find ./hf-tasks/tasks -name "workspace.yaml" | head -5 | xargs dirname | \
  xargs -I {} cargo run -- swe harness --input {} --parallel 1
```

### 4. Validating Results

Parse and validate harness output:

```bash
# Check harness completed successfully
if echo "$output" | grep -q "error\|panic\|FAILED"; then
    echo "Harness encountered errors"
fi

# Count results by status
echo "$output" | grep -o '"status":"[^"]*"' | sort | uniq -c

# Verify expected outcomes (adjust based on known task quality)
# Known good tasks should show "resolved"
# Known bad tasks should show "unresolved" or "sanity_fail"
```

### 5. Manual Verification

For real task validation:
- Pick 2-3 specific tasks from hf-tasks/
- Examine their workspace.yaml files
- Verify the harness results match expectations
- Check logs for any anomalies

Example verification:
```bash
# Look at a specific task
cat hf-tasks/tasks/<task-name>/workspace.yaml | head -30

# Run harness on just that task
mkdir -p /tmp/single-task
cp -r hf-tasks/tasks/<task-name> /tmp/single-task/
cargo run -- swe harness --input /tmp/single-task --parallel 1 --json
```

### 6. Documentation

Document findings:
- Number of tasks tested
- Pass/fail/sanity_fail/error counts
- Any patterns observed (specific types of failures)
- Container cleanup verification

## Example Handoff

```json
{
  "salientSummary": "Ran harness on 68 real tasks from hf-tasks/. Results: 45 resolved, 12 unresolved, 8 sanity_fail, 3 setup_error. All containers cleaned up properly.",
  "whatWasImplemented": "Executed end-to-end harness validation on hf-tasks/ directory. Processed 68 workspace.yaml files through the full pipeline including discovery, loading, sanity checks, and test execution. Verified container cleanup and result classification.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "cargo run -- swe harness --input ./hf-tasks/tasks --parallel 1 --json", "exitCode": 0, "observation": "Completed processing 68 tasks, output valid JSON"},
      {"command": "docker ps -a | grep swe-harness", "exitCode": 1, "observation": "No lingering harness containers"},
      {"command": "echo '$output' | jq '.results | group_by(.status) | map({status: .[0].status, count: length})'", "exitCode": 0, "observation": "Status distribution as expected"}
    ],
    "interactiveChecks": [
      {"action": "Sampled 3 specific tasks and verified workspace.yaml structure", "observed": "All had required fields: id, repo, base_commit, fail_to_pass, pass_to_pass"}
    ],
    "testsAdded": []
  },
  "discoveredIssues": [
    {"severity": "info", "description": "12 tasks marked unresolved - some may be legitimate failures, others may indicate validation issues", "suggestedFix": "Review specific fail_to_pass/pass_to_pass commands for these tasks"},
    {"severity": "info", "description": "8 tasks marked sanity_fail - tests may not match expected semantics", "suggestedFix": "Investigate if these are test generation issues or real task problems"}
  ]
}
```

## When to Return to Orchestrator

Return if:
- Harness fails to run or produces errors
- Docker is unavailable
- Results don't match expected patterns (may indicate bugs)
- Environment variables missing for critical functionality
- Need to scope additional validation work
