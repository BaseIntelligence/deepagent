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

# JS/TS dual-run suite: local bins on PATH, parse to junit node ids
export PATH="/app/node_modules/.bin:${PATH:-}"
if [ -f package.json ] && [ ! -d node_modules ]; then
  npm install --no-audit --no-fund >>"$RUN_LOG" 2>&1 || true
  export PATH="/app/node_modules/.bin:${PATH:-}"
fi
set +e
python3 - <<'PY'
import json, os, re, subprocess, xml.etree.ElementTree as ET
from pathlib import Path
cfg = json.loads(Path('/tests/config.json').read_text())
cmd = (cfg.get('suite_command') or 'npm test').strip()
rep = (cfg.get('suite_reporter') or {}).get('tool_label') or ''
pkg = {}
if Path('package.json').is_file():
    try:
        pkg = json.loads(Path('package.json').read_text())
    except Exception:
        pkg = {}
scripts = {str(k): str(v) for k, v in (pkg.get('scripts') or {}).items()}
deps = {}
for key in ('dependencies','devDependencies','peerDependencies'):
    blob = pkg.get(key) or {}
    if isinstance(blob, dict):
        deps.update({str(k).lower(): str(v) for k, v in blob.items()})
combo = f"{(scripts.get('test') or '')} {(scripts.get('tests-only') or '')}".lower()
env = dict(os.environ)
binp = str(Path('node_modules/.bin').resolve())
if Path(binp).is_dir():
    env['PATH'] = binp + os.pathsep + env.get('PATH','')
argv = None
if 'ava' in deps or re.search(r'(^|[\s&|;])ava([\s&|;]|$)', combo):
    ava = Path('node_modules/.bin/ava')
    argv = [str(ava)] if ava.is_file() else ['npx','--no-install','ava']
elif 'tape' in deps or 'tape' in combo:
    if scripts.get('tests-only'):
        argv = ['npm','run','tests-only','--silent']
    else:
        tbin = Path('node_modules/.bin/tape')
        argv = [str(tbin),'test/**/*.js'] if tbin.is_file() else ['npx','--no-install','tape','test/**/*.js']
elif 'jest' in deps or 'jest' in combo:
    argv = ['npm','test','--','--json','--runInBand']
else:
    argv = ['npm','test']
proc = subprocess.run(argv, cwd='/app', capture_output=True, text=True, env=env, check=False)
text = (proc.stdout or '') + '\n' + (proc.stderr or '')
Path('/logs/verifier/new.log').write_text(text)
Path('/logs/verifier/run.log').write_text(text[-20000:])
passed=[]; failed=[]
# Prefer suite_reporters multi-framework parse when importable
try:
    from swe_factory.producers.suite_reporters import parse_with_reporter
    out = parse_with_reporter('javascript', text, returncode=proc.returncode)
    passed=list(out.passed); failed=list(out.failed)
except Exception:
    for m in re.finditer(r'[✓✔]\s+(.+)', text):
        passed.append(m.group(1).strip())
    for m in re.finditer(r'[✕✖×]\s+(.+)', text):
        failed.append(m.group(1).strip())
    for m in re.finditer(r'^ok\s+\d+\s+(.+)$', text, re.M):
        passed.append(m.group(1).strip())
    for m in re.finditer(r'^not ok\s+\d+\s+(.+)$', text, re.M):
        failed.append(m.group(1).strip())
# Also include config-labeled node rows (bridge missing parse titles)
want=set(map(str, (cfg.get('f2p_node_ids') or []) + (cfg.get('p2p_node_ids') or [])))
seen=set(passed)|set(failed)
for n in want:
    if n not in seen:
        # If suite exit 0 and no explicit fail, treat as missing not auto-pass
        pass
root=ET.Element('testsuite', name='js', tests=str(len(passed)+len(failed)))
for n in dict.fromkeys(passed):
    ET.SubElement(root,'testcase', name=n)
for n in dict.fromkeys(failed):
    tc=ET.SubElement(root,'testcase', name=n)
    ET.SubElement(tc,'failure', message='failed')
Path('/logs/verifier/new.xml').write_text(ET.tostring(root, encoding='unicode'))
Path('/logs/verifier/base.xml').write_text('<testsuite tests="0"/>')
print('[verifier] js suite nodes pass=%d fail=%d rc=%s' % (len(passed),len(failed),proc.returncode))
PY
new_rc=$?
set -e
log "js suite converter rc=$new_rc"

set -e
python3 /tests/grader.py grade || exit $?
