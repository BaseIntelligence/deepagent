#!/usr/bin/env bash
set -euo pipefail
cd /projects/Agent-SWE/deepagent
# Parse token from .env without sourcing proxy vars
export GITHUB_TOKEN="$(python3 - <<'PY'
from pathlib import Path
import re
t = Path('.env').read_text()
m = re.search(r'^GITHUB_TOKEN=(.+)$', t, re.M)
print((m.group(1).strip().strip('"').strip("'")) if m else '')
PY
)"
export GH_TOKEN="${GITHUB_TOKEN}"
unset ALL_PROXY all_proxy HTTPS_PROXY HTTP_PROXY https_proxy http_proxy OXYLABS_PROXY_URL || true
export PYTHONUNBUFFERED=1
mkdir -p datasets/_m28b_evidence
exec .venv/bin/python scripts/m28b_retry_smoke_gen.py >> datasets/_m28b_evidence/retry_gen_console3.log 2>&1
