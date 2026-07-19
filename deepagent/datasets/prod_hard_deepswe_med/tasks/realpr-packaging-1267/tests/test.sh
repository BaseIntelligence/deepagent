#!/bin/bash
# Verifier entrypoint. Patching/grading live in tests/grader.py.
set -uo pipefail
trap 'if [ ! -f /logs/verifier/reward.json ] && [ ! -f /logs/verifier/reward.txt ]; then mkdir -p /logs/verifier; echo -1 > /logs/verifier/reward.txt; fi' EXIT
log() { echo "[verifier] $*"; }
cd /app || { mkdir -p /logs/verifier; exit 6; }

python3 /tests/grader.py prepare || exit $?
[ -f /logs/verifier/reward.json ] && exit 0

export RUN_LOG=/logs/verifier/run.log
: > "$RUN_LOG" 2>/dev/null || true

# Resolve pytest targets from dual-run suite_paths when present.
mapfile -t PYTEST_TARGETS < <(python3 - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path('/tests/config.json').read_text())
paths = cfg.get('suite_paths') or cfg.get('pytest_paths') or []
out = []
for p in paths:
    s = str(p).strip()
    if not s:
        continue
    # Accept both repo-relative files and dotted suite prefixes.
    if Path(s).exists():
        out.append(s)
    elif Path(s.replace('.', '/') + '.py').exists():
        out.append(s.replace('.', '/') + '.py')
if not out:
    out = ['tests/']
print('\n'.join(out))
PY
)
if [ ${#PYTEST_TARGETS[@]} -eq 0 ]; then PYTEST_TARGETS=(tests/); fi
log "pytest targets: ${PYTEST_TARGETS[*]}"

set +e
python -m pytest "${PYTEST_TARGETS[@]}" -v -p no:cacheprovider -o filterwarnings=default --junitxml=/logs/verifier/new.xml > /logs/verifier/new.log 2>&1
new_rc=$?
# Optional base/regression report when present
if [ -f tests/test_ok.py ] || [ -f tests/test_regression.py ]; then
  python -m pytest tests/test_ok.py tests/test_regression.py -v -p no:cacheprovider --junitxml=/logs/verifier/base.xml > /logs/verifier/base.log 2>&1
  base_rc=$?
else
  : > /logs/verifier/base.xml
  base_rc=0
fi
set -e
log "base pytest rc=${base_rc:-0}; new pytest rc=$new_rc"

set -e
python3 /tests/grader.py grade || exit $?
