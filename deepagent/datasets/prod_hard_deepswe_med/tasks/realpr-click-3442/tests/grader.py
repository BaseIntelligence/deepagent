#!/usr/bin/env python3
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
