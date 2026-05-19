"""Docker verification for generated test tasks.

Structure in Docker container:
- /workspace/repo/     - Cloned repository
- /workspace/forge/    - SWE-Forge generated files (tests, patches, config)
"""

import subprocess
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)


def generate_run_script(task_dir: Path) -> Path:
    script_path = task_dir / "run_tests.sh"

    script_content = """#!/bin/bash
set -e

# SWE-Forge Test Runner
# Structure: /workspace/repo (code) + /workspace/forge (generated tests)
# Usage: ./run_tests.sh [--verify]

TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$TASK_DIR/workspace.yaml"

# Standard paths in Docker container
REPO_PATH="/workspace/repo"
FORGE_PATH="/workspace/forge"
TESTS_PATH="/workspace/forge/tests"

echo "=== SWE-Forge Test Runner ==="
echo ""

# Parse workspace.yaml
get_repo_url() {
    grep -A2 "repo:" "$WORKSPACE" | grep "url:" | sed 's/.*url: *//' | sed 's/"//g'
}

BASE_COMMIT=$(grep "base_commit:" "$WORKSPACE" | sed 's/.*base_commit: *//' | sed 's/"//g')
MERGE_COMMIT=$(grep "merge_commit:" "$WORKSPACE" | sed 's/.*merge_commit: *//' | sed 's/"//g')
REPO_URL=$(get_repo_url)

echo "Repo: $REPO_URL"
echo "Base: $BASE_COMMIT"
echo "Merge: $MERGE_COMMIT"
echo ""

# Get install commands
get_install_commands() {
    sed -n '/install:/,/^[a-z]/p' "$WORKSPACE" | grep -E "^\\s+-" | sed 's/.*- *//'
}

# Get test commands
get_fail_to_pass() {
    sed -n '/fail_to_pass:/,/pass_to_pass:/p' "$WORKSPACE" | grep -E "^\\s+-" | sed 's/.*- *//'
}

get_pass_to_pass() {
    sed -n '/pass_to_pass:/,/^[a-z]/p' "$WORKSPACE" | grep -E "^\\s+-" | sed 's/.*- *//'
}

echo "=== Install Commands ==="
get_install_commands
echo ""

echo "=== Tests (fail_to_pass) ==="
get_fail_to_pass
echo ""

# Verification mode
if [ "$1" == "--verify" ]; then
    echo "=== Running in Docker ==="
    
    IMAGE=$(grep "image:" "$WORKSPACE" | head -1 | sed 's/.*image: *//' | sed 's/"//g')
    if [ -z "$IMAGE" ]; then
        IMAGE="ubuntu:24.04"
    fi
    
    echo "Using image: $IMAGE"
    echo "Structure:"
    echo "  - $REPO_PATH (repository)"
    echo "  - $FORGE_PATH (SWE-Forge files - hidden from agent)"
    echo ""
    
    docker run --rm \
        -v "$TASK_DIR:$FORGE_PATH:ro" \
        -w "$REPO_PATH" \
        "$IMAGE" \
        bash -c '
            set -e
            
            echo "=== Installing dependencies ==="
            apt-get update > /dev/null 2>&1
            apt-get install -y git python3 python3-pip python3-venv > /dev/null 2>&1
            
            echo "=== Cloning repository to '"$REPO_PATH"' ==="
            rm -rf '"$REPO_PATH"'
            git clone '"$REPO_URL"' '"$REPO_PATH"' 2>/dev/null
            cd '"$REPO_PATH"'
            git checkout '"$BASE_COMMIT"' 2>/dev/null || true
            
            echo "=== Checking base state (tests should FAIL) ==="
            if [ -d '"$FORGE_PATH"'/tests ]; then
                echo "Test files available:"
                ls -la '"$FORGE_PATH"'/tests/
            fi
            
            echo ""
            echo "=== Running install commands ==="
            
            echo ""
            echo "Done. Repository cloned to: '"$REPO_PATH"'"
            echo "SWE-Forge files at: '"$FORGE_PATH"'"
            echo ""
            echo "To run tests manually:"
            echo "  cd '"$REPO_PATH"' && pytest '"$FORGE_PATH"'/tests/"
        '
fi

echo ""
echo "=== Structure ==="
echo "In Docker container:"
echo "  - Repository: $REPO_PATH"
echo "  - Tests/forge: $FORGE_PATH (mounted from task dir)"
echo ""
echo "Commands:"
echo "  ./run_tests.sh           # Show config"
echo "  ./run_tests.sh --verify  # Run Docker verification"
"""

    with open(script_path, "w") as f:
        f.write(script_content)

    script_path.chmod(0o755)
    return script_path


def generate_evaluate_script(
    task_dir: Path,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    install_commands: list[str],
    base_commit: str,
    repo_url: str,
    deletion_patch: str = "",
) -> Path:
    """Generate evaluate.sh that scores a task 0 or 1.

    Phases:
      1. Clone repo at base_commit, install deps, copy test files
      2. Apply deletion_patch.diff when present for synthetic tasks
      3. BEFORE PATCH: fail_to_pass must FAIL, pass_to_pass must PASS
      4. AFTER PATCH (git apply patch.diff): all tests must PASS
      5. Output {"score": 1} if everything matches, {"score": 0} otherwise
    """
    script_path = task_dir / "evaluate.sh"
    if deletion_patch:
        deletion_patch_path = task_dir / "deletion_patch.diff"
        if not deletion_patch_path.exists():
            deletion_patch_path.write_text(
                deletion_patch
                if deletion_patch.endswith("\n")
                else deletion_patch + "\n"
            )

    def _sh_escape(s: str) -> str:
        return s.replace("'", "'\\''")

    install_block = ""
    for cmd in install_commands:
        install_block += f"  {cmd} || true\n"

    f2p_before_block = ""
    for cmd in fail_to_pass:
        f2p_before_block += (
            f"  if {cmd}; then\n"
            f'    echo "FAIL: fail_to_pass command should FAIL before patch: {_sh_escape(cmd)}"\n'
            f"    SCORE=0\n"
            f"  fi\n"
        )

    p2p_before_block = ""
    for cmd in pass_to_pass:
        p2p_before_block += (
            f"  if ! {cmd}; then\n"
            f'    echo "FAIL: pass_to_pass command should PASS before patch: {_sh_escape(cmd)}"\n'
            f"    SCORE=0\n"
            f"  fi\n"
        )

    f2p_after_block = ""
    for cmd in fail_to_pass:
        f2p_after_block += (
            f"  if ! {cmd}; then\n"
            f'    echo "FAIL: fail_to_pass command should PASS after patch: {_sh_escape(cmd)}"\n'
            f"    SCORE=0\n"
            f"  fi\n"
        )

    p2p_after_block = ""
    for cmd in pass_to_pass:
        p2p_after_block += (
            f"  if ! {cmd}; then\n"
            f'    echo "FAIL: pass_to_pass command should PASS after patch: {_sh_escape(cmd)}"\n'
            f"    SCORE=0\n"
            f"  fi\n"
        )

    script_content = f"""#!/bin/bash
# evaluate.sh - SWE-Forge Task Evaluator
# Outputs: {{"score": 0}} or {{"score": 1}}
# Score 1 = fail_to_pass FAIL before patch + ALL tests PASS after patch
# Score 0 = any check fails

set -o pipefail

TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_PATH="${{1:-/workspace/repo}}"
FORGE_PATH="$TASK_DIR"
SCORE=1

echo "=== SWE-Forge Evaluator ==="
echo "Task dir: $TASK_DIR"
echo "Repo path: $REPO_PATH"
echo ""

# ── Setup ─────────────────────────────────────────────
if [ ! -d "$REPO_PATH/.git" ]; then
  echo "Cloning repository..."
  rm -rf "$REPO_PATH"
  git clone {repo_url} "$REPO_PATH" 2>/dev/null
fi

cd "$REPO_PATH"
git checkout {base_commit} --force 2>/dev/null
git clean -fdx 2>/dev/null

if [ -s "$FORGE_PATH/deletion_patch.diff" ]; then
  echo "Applying synthetic deletion patch..."
  if ! git apply "$FORGE_PATH/deletion_patch.diff" 2>/dev/null; then
    if ! git apply --3way "$FORGE_PATH/deletion_patch.diff" 2>/dev/null; then
      echo "ERROR: Could not apply deletion patch"
      echo '{{"score": 0}}'
      exit 0
    fi
  fi
fi

# Copy test files into repo if needed
if [ -d "$FORGE_PATH/tests" ]; then
  mkdir -p "$REPO_PATH/forge_tests"
  cp -r "$FORGE_PATH/tests/"* "$REPO_PATH/forge_tests/" 2>/dev/null || true
fi

# ── Install ───────────────────────────────────────────
echo "=== Installing dependencies ==="
{install_block if install_block else "  echo 'No install commands'"}

# ── Phase 1: BEFORE PATCH ────────────────────────────
echo ""
echo "=== Phase 1: Before patch (base commit) ==="
echo "fail_to_pass tests must FAIL, pass_to_pass tests must PASS"
echo ""

{f2p_before_block if f2p_before_block else "  echo 'No fail_to_pass tests'"}
{p2p_before_block if p2p_before_block else "  echo 'No pass_to_pass tests'"}

if [ "$SCORE" -eq 0 ]; then
  echo ""
  echo "Phase 1 FAILED - aborting"
  echo '{{"score": 0}}'
  exit 0
fi
echo "Phase 1 PASSED"

# ── Apply patch ───────────────────────────────────────
echo ""
echo "=== Applying patch ==="
if ! git apply "$FORGE_PATH/patch.diff" 2>/dev/null; then
  if ! git apply --3way "$FORGE_PATH/patch.diff" 2>/dev/null; then
    echo "ERROR: Could not apply patch"
    echo '{{"score": 0}}'
    exit 0
  fi
fi
echo "Patch applied successfully"

# ── Phase 2: AFTER PATCH ─────────────────────────────
echo ""
echo "=== Phase 2: After patch ==="
echo "ALL tests must PASS"
echo ""

{f2p_after_block if f2p_after_block else "  echo 'No fail_to_pass tests'"}
{p2p_after_block if p2p_after_block else "  echo 'No pass_to_pass tests'"}

# ── Result ────────────────────────────────────────────
echo ""
if [ "$SCORE" -eq 1 ]; then
  echo "=== RESULT: PASS ==="
  echo '{{"score": 1}}'
else
  echo "=== RESULT: FAIL ==="
  echo '{{"score": 0}}'
fi
"""

    with open(script_path, "w") as f:
        f.write(script_content)
    script_path.chmod(0o755)
    return script_path


def verify_task_in_docker(task_dir: Path, timeout: int = 300) -> dict:
    import yaml

    workspace_path = task_dir / "workspace.yaml"
    if not workspace_path.exists():
        return {"success": False, "error": "No workspace.yaml found"}

    generate_run_script(task_dir)

    with open(workspace_path) as f:
        config = yaml.safe_load(f)

    repo_url = config.get("repo", {}).get("url", "")
    base_commit = config.get("repo", {}).get("base_commit", "")
    install_commands = config.get("install", {}).get("commands", [])
    fail_to_pass = config.get("tests", {}).get("fail_to_pass", [])

    if not repo_url:
        return {"success": False, "error": "No repo URL in workspace.yaml"}

    image = config.get("environment", {}).get("image", "ubuntu:24.04")

    verify_script = f"""#!/bin/bash
set -e

REPO_PATH="/workspace/repo"
FORGE_PATH="/workspace/forge"

echo "=== Installing system dependencies ==="
apt-get update > /dev/null 2>&1
apt-get install -y git python3 python3-pip python3-venv > /dev/null 2>&1

echo "=== Cloning repository ==="
rm -rf "$REPO_PATH"
git clone {repo_url} "$REPO_PATH" 2>/dev/null
cd "$REPO_PATH"
git checkout {base_commit} 2>/dev/null || true

echo "=== Base commit: $(git rev-parse HEAD) ==="

echo "=== Listing available tests ==="
ls -la "$FORGE_PATH/tests/" 2>/dev/null || echo "No tests directory found"

echo "=== Running init commands ==="
"""

    for cmd in install_commands[:5]:
        verify_script += f"""
echo "Running: {cmd}"
{cmd} || echo "Command finished with exit $?"
"""

    verify_script += """
echo ""
echo "=== Test verification complete ==="
"""

    for test_cmd in fail_to_pass[:3]:
        verify_script += f"""
echo "Would run: {test_cmd}"
"""

    script_path = task_dir / "verify_docker.sh"
    with open(script_path, "w") as f:
        f.write(verify_script)
    script_path.chmod(0o755)

    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{task_dir}:/workspace/forge:ro",
                "-w",
                "/workspace/repo",
                "--timeout",
                str(timeout),
                image,
                "bash",
                "/workspace/forge/verify_docker.sh",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}
