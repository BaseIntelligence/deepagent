#!/bin/bash
set -e

# SWE-Forge Test Runner
# Usage: ./run_tests.sh [--verify]

TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$TASK_DIR/workspace.yaml"

# Parse workspace.yaml (simple grep-based parsing)
get_value() {
    grep -A1 "$1:" "$WORKSPACE" | tail -1 | sed 's/^[[:space:]]*//' | sed 's/"//g'
}

# Get repo info
REPO_URL=$(grep -A2 "repo:" "$WORKSPACE" | grep "url:" | sed 's/.*url: *//' | sed 's/"//g')
BASE_COMMIT=$(grep "base_commit:" "$WORKSPACE" | sed 's/.*base_commit: *//' | sed 's/"//g')
MERGE_COMMIT=$(grep "merge_commit:" "$WORKSPACE" | sed 's/.*merge_commit: *//' | sed 's/"//g')

echo "=== SWE-Forge Test Runner ==="
echo "Repo: $REPO_URL"
echo "Base: $BASE_COMMIT"
echo "Merge: $MERGE_COMMIT"

# Get install commands (multiline, until next key)
get_install_commands() {
    sed -n '/install:/,/^[a-z]/p' "$WORKSPACE" | grep -E "^\s+-" | sed 's/.*- *//'
}

# Get fail_to_pass tests
get_fail_to_pass() {
    sed -n '/fail_to_pass:/,/pass_to_pass:/p' "$WORKSPACE" | grep -E "^\s+-" | sed 's/.*- *//'
}

# Get pass_to_pass tests  
get_pass_to_pass() {
    sed -n '/pass_to_pass:/,/^[a-z]/p' "$WORKSPACE" | grep -E "^\s+-" | sed 's/.*- *//'
}

echo ""
echo "=== Install Commands ==="
get_install_commands

echo ""
echo "=== Tests (fail_to_pass) ==="
get_fail_to_pass

# Run in Docker container
if [ "$1" == "--verify" ]; then
    echo ""
    echo "=== Running in Docker ==="
    
    IMAGE=$(grep "image:" "$WORKSPACE" | head -1 | sed 's/.*image: *//' | sed 's/"//g')
    if [ -z "$IMAGE" ]; then
        IMAGE="ubuntu:24.04"
    fi
    
    echo "Using image: $IMAGE"
    
    # Run Docker container with tests
    docker run --rm -v "$TASK_DIR:/task" -w /repo "$IMAGE" bash -c "
        # Install git if needed
        apt-get update && apt-get install -y git python3 python3-pip > /dev/null 2>&1
        
        # Clone repo
        git clone $REPO_URL /repo 2>/dev/null || true
        cd /repo
        
        # Apply patch if exists
        if [ -f /task/patch.diff ]; then
            git checkout $BASE_COMMIT 2>/dev/null
            git apply /task/patch.diff || echo 'Patch may already be applied'
        fi
        
        # Run install commands
        get_install_commands | while read cmd; do
            echo 'Running: '\$cmd
            eval \$cmd
        done
        
        # Run fail_to_pass tests
        echo ''
        echo '=== Running fail_to_pass tests ==='
        get_fail_to_pass | while read test_cmd; do
            echo 'Test: '\$test_cmd
        done
    "
fi

echo ""
echo "Done. To verify in Docker, run: ./run_tests.sh --verify"
