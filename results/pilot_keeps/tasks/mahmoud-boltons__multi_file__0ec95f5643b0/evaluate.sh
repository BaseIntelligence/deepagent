#!/bin/bash
# evaluate.sh - SWE-Forge task evaluator (synthetic FAIL->PASS).
# Emits {"score": 1} iff: the broken (mutation) tree FAILS the full hidden suite
# and PASSES regression, and the gold-patched tree PASSES the full hidden suite
# AND regression. Otherwise {"score": 0}.
set -o pipefail
export PYTHONDONTWRITEBYTECODE=1

TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_PATH="${1:-/workspace/repo}"
FORGE_PATH="$TASK_DIR"
SCORE=1

echo "=== SWE-Forge Evaluator ==="
echo "Task dir: $TASK_DIR"
echo "Repo path: $REPO_PATH"

# -- Setup: clone + checkout the base commit (skip when already present) -----
if [ ! -d "$REPO_PATH/.git" ]; then
  echo "Cloning repository..."
  rm -rf "$REPO_PATH"
  git clone https://github.com/mahmoud/boltons.git "$REPO_PATH" 2>/dev/null
fi
cd "$REPO_PATH" || { echo '{"score": 0}'; exit 0; }
git checkout 979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe --force 2>/dev/null || true
git clean -fdx 2>/dev/null || true

# -- Apply the mutation/deletion patch (robust: straight apply -> --3way) ----
if [ -s "$FORGE_PATH/deletion_patch.diff" ]; then
  echo "Applying mutation/deletion patch..."
  if ! git apply --whitespace=nowarn "$FORGE_PATH/deletion_patch.diff" 2>/dev/null; then
    if ! git apply --3way --whitespace=nowarn "$FORGE_PATH/deletion_patch.diff" 2>/dev/null; then
      echo "ERROR: could not apply deletion patch"
      echo '{"score": 0}'
      exit 0
    fi
  fi
fi

# -- Install dependencies -----------------------------------------------------
echo "=== Installing dependencies ==="
  pip install -e . || true
  pip install pytest || true

# -- Phase 1: broken tree (hidden suite FAILs, regression PASSes) -------------
echo "=== Phase 1: before gold patch ==="
  find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; find . -name "*.pyc" -delete 2>/dev/null; true
  cp "$FORGE_PATH/tests/test_regression_bug.py" "$REPO_PATH/test_regression_bug.py"
  if python -m pytest test_regression_bug.py; then
    echo "FAIL: hidden test should FAIL before patch: python -m pytest test_regression_bug.py"
    SCORE=0
  fi
  rm -f "$REPO_PATH/test_regression_bug.py"
  find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; find . -name "*.pyc" -delete 2>/dev/null; true
  if ! python -m pytest -k 'not (test_get_fstr_args)'; then
    echo "FAIL: pass_to_pass should PASS: python -m pytest -k 'not (test_get_fstr_args)'"
    SCORE=0
  fi
if [ "$SCORE" -eq 0 ]; then
  echo "Phase 1 FAILED - aborting"
  echo '{"score": 0}'
  exit 0
fi
echo "Phase 1 PASSED"

# -- Apply the gold patch (robust: straight apply -> --3way) -----------------
echo "=== Applying gold patch ==="
if ! git apply --whitespace=nowarn "$FORGE_PATH/patch.diff" 2>/dev/null; then
  if ! git apply --3way --whitespace=nowarn "$FORGE_PATH/patch.diff" 2>/dev/null; then
    echo "ERROR: could not apply gold patch"
    echo '{"score": 0}'
    exit 0
  fi
fi
echo "Gold patch applied"

# -- Phase 2: gold tree (hidden suite PASSes, regression PASSes) -------------
echo "=== Phase 2: after gold patch ==="
  find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; find . -name "*.pyc" -delete 2>/dev/null; true
  cp "$FORGE_PATH/tests/test_regression_bug.py" "$REPO_PATH/test_regression_bug.py"
  if ! python -m pytest test_regression_bug.py; then
    echo "FAIL: hidden test should PASS after patch: python -m pytest test_regression_bug.py"
    SCORE=0
  fi
  rm -f "$REPO_PATH/test_regression_bug.py"
  find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; find . -name "*.pyc" -delete 2>/dev/null; true
  if ! python -m pytest -k 'not (test_get_fstr_args)'; then
    echo "FAIL: pass_to_pass should PASS: python -m pytest -k 'not (test_get_fstr_args)'"
    SCORE=0
  fi

echo ""
if [ "$SCORE" -eq 1 ]; then
  echo "=== RESULT: PASS ==="
  echo '{"score": 1}'
else
  echo "=== RESULT: FAIL ==="
  echo '{"score": 0}'
fi
