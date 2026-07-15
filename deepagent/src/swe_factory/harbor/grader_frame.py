"""DeepAgent-compatible grader frame (prepare / grade / patch-paths).

Embedded as tests/grader.py inside each Harbor pack. Matches the DeepAgent v1.1
contract: apply model.patch + held-out test.patch on prepare; whitelist node-id
grading into /logs/verifier/reward.json.
"""

from __future__ import annotations

# Shared verbatim-style grade frame. Keep in sync with DeepAgent tools/verifier/grader.py
# intent: offline-exportable pure sibling, no external package imports.
GRADER_PY_SOURCE = r'''#!/usr/bin/env python3
"""DeepAgent v1.1 task verifier frame — prepare / grade / patch-paths.

Per-task data lives in tests/config.json next to this file:
  base_commit, f2p_node_ids, p2p_node_ids, grade.{format,reports,tool_label,...}
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
VERIFIER_DIR = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/logs/artifacts"))
RANK = {"passed": 0, "skipped": 1, "failed": 2}


def log(msg: str) -> None:
    print(f"[verifier] {msg}", flush=True)


def load_config() -> dict:
    return json.loads((TESTS_DIR / "config.json").read_text())


def patch_paths(text: str) -> list[str]:
    """Unique file paths a unified diff touches, in order of appearance."""
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        path = None
        m = re.match(r'^diff --git (?:"?a/(.*?)"?) (?:"?b/(.*?)"?)$', line)
        if m:
            path = m.group(2)
        elif line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("--- a/"):
            path = line[6:]
        if path and path != "/dev/null" and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def read_patch(path: Path | str) -> str:
    p = Path(path)
    return p.read_text(errors="replace") if p.exists() else ""


def git(*args: str, **kw: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=APP_DIR, **kw)  # type: ignore[arg-type]


def git_commit_exists(ref: str) -> bool:
    if not ref or not str(ref).strip():
        return False
    r = git(
        "cat-file",
        "-e",
        f"{ref.strip()}^{{commit}}",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0


def resolve_base_commit(configured: str | None = None) -> str:
    """Resolve the workspace base ref for prepare-time path resets.

    Prefer config/marker only when the object exists in the agent git history.
    Offline motors create a real ``git commit -m base`` SHA at image build time;
    synthetic placeholders like a1000… are not ancestors and must not be used.
    """
    marker = APP_DIR / ".harbor_base_commit"
    if marker.is_file():
        file_ref = marker.read_text(errors="replace").strip()
        if git_commit_exists(file_ref):
            return file_ref
    env_ref = (os.environ.get("BASE_COMMIT") or "").strip()
    if git_commit_exists(env_ref):
        return env_ref
    cfg = (configured or "").strip()
    if git_commit_exists(cfg):
        return cfg
    try:
        root = git(
            "rev-list",
            "--max-parents=0",
            "HEAD",
            capture_output=True,
            text=True,
            check=False,
        )
        line = (root.stdout or "").strip().splitlines()
        if line and git_commit_exists(line[0].strip()):
            return line[0].strip()
    except OSError:
        pass
    return cfg or "HEAD"


def reset_paths(paths: list[str], ref: str) -> None:
    for f in paths:
        if not f:
            continue
        rc = git("checkout", "-q", ref, "--", f, stderr=subprocess.DEVNULL).returncode
        if rc != 0 and ref == "HEAD" and (APP_DIR / f).exists():
            subprocess.run(["rm", "-rf", "--", f], cwd=APP_DIR)


def cmd_prepare(argv: list[str]) -> None:
    del argv  # unused; interface parity with DeepAgent
    if not APP_DIR.is_dir():
        VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
        sys.exit(6)
    os.chdir(APP_DIR)
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", str(APP_DIR)],
        stderr=subprocess.DEVNULL,
    )
    configured = str(load_config().get("base_commit") or "")
    base = resolve_base_commit(configured)
    if base != configured:
        log(f"base_commit {configured!r} not in workspace git; using {base!r}")
    model_patch = ARTIFACTS_DIR / "model.patch"
    if model_patch.exists() and model_patch.stat().st_size > 0:
        reset_paths(patch_paths(read_patch(model_patch)), base)
        rc = git("apply", "--whitespace=nowarn", str(model_patch)).returncode
        if rc != 0:
            log("ERROR: submitted model.patch failed to apply")
            cmd_grade(["--apply-failed"])
            sys.exit(0)
        log(f"model.patch applied ({model_patch.stat().st_size} bytes)")
    else:
        log("no model.patch submitted — grading pristine base state")

    test_patch = TESTS_DIR / "test.patch"
    log("Resetting files touched by test.patch")
    reset_paths(patch_paths(read_patch(test_patch)), "HEAD")
    log("Applying test.patch")
    r = git(
        "apply",
        "--whitespace=nowarn",
        "--allow-empty",
        str(test_patch),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        log("ERROR: test.patch failed to apply")
        sys.stderr.write((r.stdout or "") + (r.stderr or ""))
        sys.exit(r.returncode)
    try:
        inner = APP_DIR / "test.sh"
        inner.chmod(inner.stat().st_mode | 0o111)
    except OSError:
        pass


def norm_status(raw: object) -> str:
    raw_s = str(raw or "").strip().lower()
    if raw_s == "passed":
        return "passed"
    if raw_s in ("skipped", "pending", "other"):
        return "skipped"
    return "failed"


def add(res: dict[str, tuple[str, str]], nid: str, st: str, msg: str = "") -> None:
    cur = res.get(nid)
    msg = msg or ""
    if cur is None or RANK[st] > RANK[cur[0]]:
        res[nid] = (st, msg if st != "passed" else "")
    elif RANK[st] == RANK[cur[0]] and st != "passed" and not cur[1] and msg:
        res[nid] = (st, msg)


def parse_ctrf(path: str | Path, cfg: dict) -> dict[str, tuple[str, str]]:
    res: dict[str, tuple[str, str]] = {}
    try:
        doc = json.loads(Path(path).read_text())
        tests = (doc.get("results") or {}).get("tests") or []
        if not isinstance(tests, list):
            return res
    except Exception:
        return res
    for tc in tests:
        if not isinstance(tc, dict):
            continue
        nm = str(tc.get("name") or "").strip()
        if not nm:
            continue
        su_raw = tc.get("suite")
        if isinstance(su_raw, list) and su_raw:
            su = str(su_raw[0]).strip()
        elif isinstance(su_raw, str):
            su = su_raw.strip()
        else:
            su = ""
        nid = f"{su}.{nm}" if (cfg.get("node_id") == "suite.name" and su) else nm
        st = norm_status(tc.get("status"))
        msg = ""
        if st != "passed":
            msg = str(tc.get("message") or tc.get("trace") or "").strip()
        add(res, nid, st, msg)
    return res


def junit_status_msg(tc: ET.Element) -> tuple[str, str]:
    st, msg = "passed", ""
    for ch in tc:
        tag = ch.tag.rsplit("}", 1)[-1]
        if tag in ("failure", "error"):
            parts = [(ch.get("message") or "").strip(), (ch.text or "").strip()]
            return "failed", "\n".join(p for p in parts if p).strip()
        if tag == "skipped":
            st = "skipped"
    return st, msg


def parse_junit(path: str | Path, cfg: dict) -> dict[str, tuple[str, str]]:
    del cfg
    res: dict[str, tuple[str, str]] = {}
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return res
    for tc in root.iter("testcase"):
        cn = (tc.attrib.get("classname", "") or "").strip()
        nm = (tc.attrib.get("name", "") or "").strip()
        if not nm:
            continue
        nid = f"{cn}.{nm}" if cn else nm
        st, msg = junit_status_msg(tc)
        add(res, nid, st, msg)
    return res


PARSERS = {"ctrf": parse_ctrf, "junit": parse_junit}


def cmd_grade(argv: list[str]) -> None:
    full = load_config()
    cfg = full.get("grade", {})
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)

    def load_ids(key: str) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for line in full.get(key, []):
            s = str(line).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            ids.append(s)
        return ids

    p2p = load_ids("p2p_node_ids")
    f2p = load_ids("f2p_node_ids")

    def stats(fp: int, pp: int) -> dict:
        total = len(f2p) + len(p2p)
        return {
            "f2p_total": len(f2p),
            "f2p_passed": fp,
            "p2p_total": len(p2p),
            "p2p_passed": pp,
            "f2p": fp / len(f2p) if f2p else 0.0,
            "p2p": pp / len(p2p) if p2p else 1.0,
            "partial": (fp + pp) / total if total else 0.0,
        }

    if "--apply-failed" in argv:
        out = {"reward": 0, **stats(0, 0), "apply_failed": 1}
        (VERIFIER_DIR / "reward.json").write_text(json.dumps(out))
        print(f"[grade] model.patch failed to apply; reward.json={json.dumps(out)}")
        return

    parse = PARSERS[cfg.get("format", "ctrf")]
    seen: dict[str, tuple[str, str]] = {}
    for rep in cfg.get("reports") or []:
        for k, (st, msg) in parse(rep, cfg).items():
            add(seen, k, st, msg)

    def bucket(ids: list[str]) -> tuple[int, int, list[dict]]:
        p = f = 0
        rows: list[dict] = []
        for nid in ids:
            entry = seen.get(nid)
            if entry is None:
                rows.append(
                    {
                        "name": nid,
                        "status": "failed",
                        "message": "missing from report (test did not run)",
                    }
                )
                f += 1
            elif entry[0] == "passed":
                rows.append({"name": nid, "status": "passed"})
                p += 1
            else:
                rows.append({"name": nid, "status": entry[0], "message": entry[1]})
                f += 1
        return p, f, rows

    pp, pf, pr = bucket(p2p)
    fp, ff, fr = bucket(f2p)
    binary = 1 if (len(f2p) > 0 and ff == 0 and pf == 0) else 0

    def ctrf_test(t: dict, b: str) -> dict:
        d: dict = {"name": f"[{b}] {t['name']}", "status": t["status"]}
        if t.get("message"):
            d["message"] = t["message"]
        return d

    ctrf = {
        "reportFormat": "CTRF",
        "specVersion": "1.0.0",
        "results": {
            "tool": {"name": cfg.get("tool_label", "unknown")},
            "summary": {
                "tests": len(p2p) + len(f2p),
                "passed": pp + fp,
                "failed": pf + ff,
                "skipped": 0,
                "pending": 0,
                "other": 0,
            },
            "tests": [ctrf_test(t, "p2p") for t in pr]
            + [ctrf_test(t, "f2p") for t in fr],
        },
    }
    (VERIFIER_DIR / "ctrf.json").write_text(json.dumps(ctrf, indent=2))
    out = {"reward": binary, **stats(fp, pp)}
    (VERIFIER_DIR / "reward.json").write_text(json.dumps(out))
    print(
        f"P2P {pp}/{len(p2p)} pass {pf} fail; F2P {fp}/{len(f2p)} pass {ff} fail; "
        f"PARTIAL {out['partial']}; BINARY {binary}"
    )


def cmd_patch_paths(argv: list[str]) -> None:
    for path in patch_paths(read_patch(argv[0])):
        print(path)


def main() -> None:
    cmds = {"prepare": cmd_prepare, "grade": cmd_grade, "patch-paths": cmd_patch_paths}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(f"usage: grader.py {{{'|'.join(cmds)}}} [args]", file=sys.stderr)
        sys.exit(2)
    cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
'''


def render_grader_py() -> str:
    """Return the shared DeepAgent-compatible grader.py body."""
    return GRADER_PY_SOURCE


def default_test_sh(*, language: str = "python") -> str:
    # ruff: noqa: E501 — embedded multi-lang verifier shell strings
    """Default tests/test.sh frame for dual-run F2P/P2P verifier suites.

    - python: pytest (scoped by suite_paths when present)
    - javascript/typescript: npm/ava/tape/jest with node_modules/.bin on PATH;
      suite output converted to junit for the shared grader
    - rust: cargo test converted to junit for shared grader
    - go: go test converted to junit

    Junit case ``name`` = dual-run node id so grade.py buckets match product
    labels (tape titles, cargo names, ava titles, go TestXxx).
    """
    lang = (language or "python").strip().lower()
    header = (
        "#!/bin/bash\n"
        "# Verifier entrypoint. Patching/grading live in tests/grader.py.\n"
        "set -uo pipefail\n"
        "trap '"
        "if [ ! -f /logs/verifier/reward.json ] "
        "&& [ ! -f /logs/verifier/reward.txt ]; "
        "then mkdir -p /logs/verifier; "
        "echo -1 > /logs/verifier/reward.txt; fi' EXIT\n"
        'log() { echo "[verifier] $*"; }\n'
        "cd /app || { mkdir -p /logs/verifier; exit 6; }\n"
        "\n"
        "python3 /tests/grader.py prepare || exit $?\n"
        "[ -f /logs/verifier/reward.json ] && exit 0\n"
        "\n"
        "export RUN_LOG=/logs/verifier/run.log\n"
        ': > "$RUN_LOG" 2>/dev/null || true\n'
        "\n"
    )
    # Shared: write raw suite log + junit from reporter node ids
    footer = "set -e\npython3 /tests/grader.py grade || exit $?\n"
    if lang in {"javascript", "js", "typescript", "ts"}:
        return (
            header + "# JS/TS dual-run suite: local bins on PATH, parse to junit node ids\n"
            'export PATH="/app/node_modules/.bin:${PATH:-}"\n'
            "if [ -f package.json ] && [ ! -d node_modules ]; then\n"
            '  npm install --no-audit --no-fund >>"$RUN_LOG" 2>&1 || true\n'
            '  export PATH="/app/node_modules/.bin:${PATH:-}"\n'
            "fi\n"
            "set +e\n"
            "python3 - <<'PY'\n"
            "import json, os, re, subprocess, xml.etree.ElementTree as ET\n"
            "from pathlib import Path\n"
            "cfg = json.loads(Path('/tests/config.json').read_text())\n"
            "cmd = (cfg.get('suite_command') or 'npm test').strip()\n"
            "rep = (cfg.get('suite_reporter') or {}).get('tool_label') or ''\n"
            "pkg = {}\n"
            "if Path('package.json').is_file():\n"
            "    try:\n"
            "        pkg = json.loads(Path('package.json').read_text())\n"
            "    except Exception:\n"
            "        pkg = {}\n"
            "scripts = {str(k): str(v) for k, v in (pkg.get('scripts') or {}).items()}\n"
            "deps = {}\n"
            "for key in ('dependencies','devDependencies','peerDependencies'):\n"
            "    blob = pkg.get(key) or {}\n"
            "    if isinstance(blob, dict):\n"
            "        deps.update({str(k).lower(): str(v) for k, v in blob.items()})\n"
            "combo = f\"{(scripts.get('test') or '')} {(scripts.get('tests-only') or '')}\".lower()\n"  # noqa: E501
            "env = dict(os.environ)\n"
            "binp = str(Path('node_modules/.bin').resolve())\n"
            "if Path(binp).is_dir():\n"
            "    env['PATH'] = binp + os.pathsep + env.get('PATH','')\n"
            "argv = None\n"
            "if 'ava' in deps or re.search(r'(^|[\\s&|;])ava([\\s&|;]|$)', combo):\n"
            "    ava = Path('node_modules/.bin/ava')\n"
            "    argv = [str(ava)] if ava.is_file() else ['npx','--no-install','ava']\n"
            "elif 'tape' in deps or 'tape' in combo:\n"
            "    if scripts.get('tests-only'):\n"
            "        argv = ['npm','run','tests-only','--silent']\n"
            "    else:\n"
            "        tbin = Path('node_modules/.bin/tape')\n"
            "        argv = [str(tbin),'test/**/*.js'] if tbin.is_file() else ['npx','--no-install','tape','test/**/*.js']\n"  # noqa: E501
            "elif 'jest' in deps or 'jest' in combo:\n"
            "    argv = ['npm','test','--','--json','--runInBand']\n"
            "else:\n"
            "    argv = ['npm','test']\n"
            "proc = subprocess.run(argv, cwd='/app', capture_output=True, text=True, env=env, check=False)\n"  # noqa: E501
            "text = (proc.stdout or '') + '\\n' + (proc.stderr or '')\n"
            "Path('/logs/verifier/new.log').write_text(text)\n"
            "Path('/logs/verifier/run.log').write_text(text[-20000:])\n"
            "passed=[]; failed=[]\n"
            "# Prefer suite_reporters multi-framework parse when importable\n"
            "try:\n"
            "    from swe_factory.producers.suite_reporters import parse_with_reporter\n"
            "    out = parse_with_reporter('javascript', text, returncode=proc.returncode)\n"
            "    passed=list(out.passed); failed=list(out.failed)\n"
            "except Exception:\n"
            "    for m in re.finditer(r'[✓✔]\\s+(.+)', text):\n"
            "        passed.append(m.group(1).strip())\n"
            "    for m in re.finditer(r'[✕✖×]\\s+(.+)', text):\n"
            "        failed.append(m.group(1).strip())\n"
            "    for m in re.finditer(r'^ok\\s+\\d+\\s+(.+)$', text, re.M):\n"
            "        passed.append(m.group(1).strip())\n"
            "    for m in re.finditer(r'^not ok\\s+\\d+\\s+(.+)$', text, re.M):\n"
            "        failed.append(m.group(1).strip())\n"
            "# Also include config-labeled node rows (bridge missing parse titles)\n"
            "want=set(map(str, (cfg.get('f2p_node_ids') or []) + (cfg.get('p2p_node_ids') or [])))\n"  # noqa: E501
            "seen=set(passed)|set(failed)\n"
            "for n in want:\n"
            "    if n not in seen:\n"
            "        # If suite exit 0 and no explicit fail, treat as missing not auto-pass\n"
            "        pass\n"
            "root=ET.Element('testsuite', name='js', tests=str(len(passed)+len(failed)))\n"
            "for n in dict.fromkeys(passed):\n"
            "    ET.SubElement(root,'testcase', name=n)\n"
            "for n in dict.fromkeys(failed):\n"
            "    tc=ET.SubElement(root,'testcase', name=n)\n"
            "    ET.SubElement(tc,'failure', message='failed')\n"
            "Path('/logs/verifier/new.xml').write_text(ET.tostring(root, encoding='unicode'))\n"
            "Path('/logs/verifier/base.xml').write_text('<testsuite tests=\"0\"/>')\n"
            "print('[verifier] js suite nodes pass=%d fail=%d rc=%s' % (len(passed),len(failed),proc.returncode))\n"  # noqa: E501
            "PY\n"
            "new_rc=$?\n"
            "set -e\n"
            'log "js suite converter rc=$new_rc"\n'
            "\n" + footer
        )
    if lang in {"rust", "rs"}:
        return (
            header + "# Rust dual-run: cargo test → junit node ids (unit + doctest names)\n"
            "set +e\n"
            "python3 - <<'PY'\n"
            "import json, re, subprocess, xml.etree.ElementTree as ET\n"
            "from pathlib import Path\n"
            "cfg = json.loads(Path('/tests/config.json').read_text())\n"
            "cmd = (cfg.get('suite_command') or 'cargo test -- --nocapture --test-threads=1').strip()\n"  # noqa: E501
            "proc = subprocess.run(cmd, shell=True, cwd='/app', capture_output=True, text=True, check=False)\n"  # noqa: E501
            "text = (proc.stdout or '') + '\\n' + (proc.stderr or '')\n"
            "# If online resolution fails but Cooked registry missing, retry is best-effort no-op.\n"
            "Path('/logs/verifier/new.log').write_text(text)\n"
            "Path('/logs/verifier/run.log').write_text(text[-20000:])\n"
            "passed=[]; failed=[]\n"
            "try:\n"
            "    from swe_factory.producers.suite_reporters import parse_with_reporter\n"
            "    out = parse_with_reporter('rust', text, returncode=proc.returncode)\n"
            "    passed=list(out.passed); failed=list(out.failed)\n"
            "except Exception:\n"
            "    for m in re.finditer(r'^test\\s+(.+?)\\s+\\.\\.\\.\\s+(ok|FAILED|ignored|error)\\b', text, re.M):\n"  # noqa: E501
            "        name=m.group(1).strip(); st=m.group(2)\n"
            "        if not name or name.startswith('result') or 'result:' in name: continue\n"
            "        if st=='ok': passed.append(name)\n"
            "        elif st in {'FAILED','error'}: failed.append(name)\n"
            "root=ET.Element('testsuite', name='cargo', tests=str(len(passed)+len(failed)))\n"
            "for n in dict.fromkeys(passed):\n"
            "    ET.SubElement(root,'testcase', name=n)\n"
            "for n in dict.fromkeys(failed):\n"
            "    tc=ET.SubElement(root,'testcase', name=n)\n"
            "    ET.SubElement(tc,'failure', message='failed')\n"
            "Path('/logs/verifier/new.xml').write_text(ET.tostring(root, encoding='unicode'))\n"
            "Path('/logs/verifier/base.xml').write_text('<testsuite tests=\"0\"/>')\n"
            "print('[verifier] cargo nodes pass=%d fail=%d rc=%s' % (len(passed),len(failed),proc.returncode))\n"  # noqa: E501
            "PY\n"
            "new_rc=$?\n"
            "set -e\n"
            'log "cargo suite converter rc=$new_rc"\n'
            "\n" + footer
        )
    if lang in {"go", "golang"}:
        return (
            header + "# Go dual-run: go test → junit TestXxx node ids\n"
            "set +e\n"
            "python3 - <<'PY'\n"
            "import re, subprocess, xml.etree.ElementTree as ET\n"
            "from pathlib import Path\n"
            "proc = subprocess.run(['go','test','./...','-count=1','-v'], cwd='/app', capture_output=True, text=True, check=False)\n"  # noqa: E501
            "text = (proc.stdout or '') + '\\n' + (proc.stderr or '')\n"
            "Path('/logs/verifier/new.log').write_text(text)\n"
            "passed=list(dict.fromkeys(re.findall(r'--- PASS: (\\w+)', text)))\n"
            "failed=list(dict.fromkeys(re.findall(r'--- FAIL: (\\w+)', text)))\n"
            "root=ET.Element('testsuite', name='go', tests=str(len(passed)+len(failed)))\n"
            "for n in passed: ET.SubElement(root,'testcase', name=n)\n"
            "for n in failed:\n"
            "    tc=ET.SubElement(root,'testcase', name=n)\n"
            "    ET.SubElement(tc,'failure', message='failed')\n"
            "Path('/logs/verifier/new.xml').write_text(ET.tostring(root, encoding='unicode'))\n"
            "Path('/logs/verifier/base.xml').write_text('<testsuite tests=\"0\"/>')\n"
            "print('[verifier] go nodes pass=%d fail=%d rc=%s' % (len(passed),len(failed),proc.returncode))\n"  # noqa: E501
            "PY\n"
            "new_rc=$?\n"
            "set -e\n"
            'log "go suite converter rc=$new_rc"\n'
            "\n" + footer
        )
    # Default python path
    return (
        header + "# Resolve pytest targets from dual-run suite_paths when present.\n"
        "mapfile -t PYTEST_TARGETS < <(python3 - <<'PY'\n"
        "import json\n"
        "from pathlib import Path\n"
        "cfg = json.loads(Path('/tests/config.json').read_text())\n"
        "paths = cfg.get('suite_paths') or cfg.get('pytest_paths') or []\n"
        "out = []\n"
        "for p in paths:\n"
        "    s = str(p).strip()\n"
        "    if not s:\n"
        "        continue\n"
        "    # Accept both repo-relative files and dotted suite prefixes.\n"
        "    if Path(s).exists():\n"
        "        out.append(s)\n"
        "    elif Path(s.replace('.', '/') + '.py').exists():\n"
        "        out.append(s.replace('.', '/') + '.py')\n"
        "if not out:\n"
        "    out = ['tests/']\n"
        "print('\\n'.join(out))\n"
        "PY\n"
        ")\n"
        "if [ ${#PYTEST_TARGETS[@]} -eq 0 ]; then PYTEST_TARGETS=(tests/); fi\n"
        'log "pytest targets: ${PYTEST_TARGETS[*]}"\n'
        "\n"
        "set +e\n"
        'python -m pytest "${PYTEST_TARGETS[@]}" -v -p no:cacheprovider '
        "-o filterwarnings=default "
        "--junitxml=/logs/verifier/new.xml "
        "> /logs/verifier/new.log 2>&1\n"
        "new_rc=$?\n"
        "# Optional base/regression report when present\n"
        "if [ -f tests/test_ok.py ] || [ -f tests/test_regression.py ]; then\n"
        "  python -m pytest tests/test_ok.py tests/test_regression.py "
        "-v -p no:cacheprovider "
        "--junitxml=/logs/verifier/base.xml "
        "> /logs/verifier/base.log 2>&1\n"
        "  base_rc=$?\n"
        "else\n"
        "  : > /logs/verifier/base.xml\n"
        "  base_rc=0\n"
        "fi\n"
        "set -e\n"
        'log "base pytest rc=${base_rc:-0}; new pytest rc=$new_rc"\n'
        "\n" + footer
    )


def default_solve_sh() -> str:
    return (
        "#!/bin/bash\n"
        "cd /app\n"
        "# Apply the solution patch\n"
        "git apply --whitespace=nowarn /solution/solution.patch\n"
        "# Commit the solution like a normal submission.\n"
        "git checkout -b feature/solution 2>/dev/null || true\n"
        "git add -A\n"
        'git -c user.name="oracle" -c user.email="oracle@local" '
        'commit -q --no-verify -m "Apply reference solution" || true\n'
    )


def default_environment_dockerfile(
    *,
    base_image: str = "python:3.12-slim",
    repo_url: str,
    base_commit: str,
    install_cmd: str = "pip install -e . || pip install pytest",
) -> str:
    """Agent environment Dockerfile (real clone @ base_commit for product packs).

    VAL-RCLN-001..004: materializes the public repository at the pinned SHA via
    git clone/checkout — never a motor-only ``COPY repo/`` layout when a real
    HTTPS remote is claimed. Runtime ``allow_internet=false`` is documented;
    deps are installed only at image *build* time.
    """
    # Prefer envbuild product recipe when URL/SHA are real; keep a thin local
    # shell for offline/edge callers that still pass stub URLs.
    try:
        from swe_factory.envbuild.agent_recipe import (
            is_full_base_sha,
            is_public_git_https,
            render_real_pr_agent_dockerfile,
        )

        if is_public_git_https(repo_url) and is_full_base_sha(base_commit):
            # Map single install shell string → one RUN command list.
            installs = [install_cmd] if install_cmd else None
            return render_real_pr_agent_dockerfile(
                repository_url=repo_url,
                base_commit=base_commit,
                base_image=base_image,
                install_commands=installs,
                workspace_dir="/app",
            )
    except Exception:  # noqa: BLE001 — fall through to inline template
        pass

    return f"""\
FROM {base_image}

# Runtime offline contract (VAL-RCLN-004)
LABEL harbor.allow_internet="false"
LABEL swe_factory.base_commit="{base_commit}"
ENV HARBOR_ALLOW_INTERNET=false
ENV SWE_FACTORY_ALLOW_INTERNET=false
ENV BASE_COMMIT={base_commit}

RUN apt-get update && apt-get install -y --no-install-recommends \\
    git build-essential ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG BASE_SHA={base_commit}
# Real-PR authority: git clone of repository_url at base_commit (VAL-RCLN-001).
# Product path forbids motor-only COPY repo/ hybrid (VAL-RCLN-002).
RUN set -eux; \\
    if [ -n "{repo_url}" ] && echo "{repo_url}" | grep -qE '^https?://'; then \\
      git clone --filter=blob:none "{repo_url}" .; \\
      git fetch --depth 1 origin "$BASE_SHA" || git fetch origin "$BASE_SHA"; \\
      git checkout --force "$BASE_SHA"; \\
      test "$(git rev-parse HEAD)" = "$BASE_SHA" \\
        || test "$(git rev-parse HEAD)" = "$(git rev-parse "$BASE_SHA")"; \\
      git rev-parse --verify "$BASE_SHA^{{commit}}"; \\
      git remote remove origin || true; \\
      git reflog expire --expire=now --all || true; \\
      git gc --prune=now || true; \\
      git rev-parse HEAD > /app/.harbor_base_commit; \\
    else \\
      echo "FATAL: product agent Dockerfile requires HTTPS repository_url + clone@SHA" >&2; \\
      exit 42; \\
    fi

# Build-time dependency install only — runtime allow_internet=false
RUN {install_cmd} || true

ENV PYTHONUNBUFFERED=1
RUN cd /app && git config core.hooksPath /dev/null || true
CMD ["/bin/bash"]
"""


def offline_environment_dockerfile() -> str:
    """Fixture-friendly agent Dockerfile: copies /context repo into /app."""
    return """\
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    git \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Offline fixture: build context is environment/ with repo/ subtree optional.
# The image packages a minimal broken workspace for structural load smoke.
# After creating the real base commit, bake its SHA into .harbor_base_commit so
# pre_artifacts/grader can resolve it even when pack metadata used a placeholder.
COPY repo/ /app/
RUN git init \\
 && git config user.email "fixture@local" \\
 && git config user.name "fixture" \\
 && git add -A \\
 && git commit -q -m "base" \\
 && git checkout -B main \\
 && git rev-parse HEAD > /app/.harbor_base_commit \\
 && printf '%s\\n' '.harbor_base_commit' >> /app/.git/info/exclude \\
 && true

RUN pip install --no-cache-dir pytest
ENV PYTHONUNBUFFERED=1
RUN git config core.hooksPath /dev/null || true
CMD ["/bin/bash"]
"""


def default_tests_dockerfile(*, agent_image_ref: str = "harbor-sdf-agent:local") -> str:
    """Separate verifier Dockerfile: agent image + held-out tests baked in."""
    return f"""\
# Verifier image: agent base with hidden tests baked in.
# tests/ is the build context; the agent never sees this container.
FROM {agent_image_ref}

COPY test.sh /tests/test.sh
COPY test.patch /tests/test.patch
COPY grader.py /tests/grader.py
COPY config.json /tests/config.json
RUN chmod +x /tests/test.sh
"""


__all__ = [
    "GRADER_PY_SOURCE",
    "default_environment_dockerfile",
    "default_solve_sh",
    "default_test_sh",
    "default_tests_dockerfile",
    "offline_environment_dockerfile",
    "render_grader_py",
]
