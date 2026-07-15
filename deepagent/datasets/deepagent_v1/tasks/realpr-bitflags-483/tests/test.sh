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

# Rust dual-run: cargo test → junit node ids (unit + doctest names)
set +e
python3 - <<'PY'
import json, re, subprocess, xml.etree.ElementTree as ET
from pathlib import Path
cfg = json.loads(Path('/tests/config.json').read_text())
cmd = (cfg.get('suite_command') or 'cargo test -- --nocapture --test-threads=1').strip()
proc = subprocess.run(cmd, shell=True, cwd='/app', capture_output=True, text=True, check=False)
text = (proc.stdout or '') + '\n' + (proc.stderr or '')
# If online resolution fails but Cooked registry missing, retry is best-effort no-op.
Path('/logs/verifier/new.log').write_text(text)
Path('/logs/verifier/run.log').write_text(text[-20000:])
passed=[]; failed=[]
try:
    from swe_factory.producers.suite_reporters import parse_with_reporter
    out = parse_with_reporter('rust', text, returncode=proc.returncode)
    passed=list(out.passed); failed=list(out.failed)
except Exception:
    for m in re.finditer(r'^test\s+(.+?)\s+\.\.\.\s+(ok|FAILED|ignored|error)\b', text, re.M):
        name=m.group(1).strip(); st=m.group(2)
        if not name or name.startswith('result') or 'result:' in name: continue
        if st=='ok': passed.append(name)
        elif st in {'FAILED','error'}: failed.append(name)
root=ET.Element('testsuite', name='cargo', tests=str(len(passed)+len(failed)))
for n in dict.fromkeys(passed):
    ET.SubElement(root,'testcase', name=n)
for n in dict.fromkeys(failed):
    tc=ET.SubElement(root,'testcase', name=n)
    ET.SubElement(tc,'failure', message='failed')
Path('/logs/verifier/new.xml').write_text(ET.tostring(root, encoding='unicode'))
Path('/logs/verifier/base.xml').write_text('<testsuite tests="0"/>')
print('[verifier] cargo nodes pass=%d fail=%d rc=%s' % (len(passed),len(failed),proc.returncode))
PY
new_rc=$?
set -e
log "cargo suite converter rc=$new_rc"

set -e
python3 /tests/grader.py grade || exit $?
