#!/usr/bin/env python3
"""m28b retry round-2 dual-smoke after relative nodeid + host-venv SUT-shadow fixes."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

for k in list(os.environ):
    ku = k.upper()
    if ku in {"ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "OXYLABS_PROXY_URL"} or (
        ku.endswith("_PROXY") and not ku.startswith("GITHUB")
    ):
        os.environ.pop(k, None)

from swe_factory.pipeline.hardness_floors import (  # noqa: E402
    PRODUCT_MIN_ADDED_LINES,
    PRODUCT_SOURCE_HUNK_FLOOR,
    count_gold_added_lines,
    multi_file_floor_ok,
)
from swe_factory.pipeline.ship_real_pr import (  # noqa: E402
    _materialize_base_worktree,
    _prepare_host_suite_env,
    load_real_pr_materials,
)
from swe_factory.producers.hard_filter import count_unified_diff_hunks  # noqa: E402
from swe_factory.producers.real_dual_run import label_real_pr_dual_run  # noqa: E402
from swe_factory.sources.clone_cache import CloneCache  # noqa: E402

EV = Path("datasets/_m28b_evidence2")
EV.mkdir(parents=True, exist_ok=True)
SURV_DIR = Path("datasets/live_materials_m28b_retry2")
SURV_DIR.mkdir(parents=True, exist_ok=True)

HAVE = {
    "realpr-itemadapter-101",
    "realpr-packaging-1120",
    "realpr-rich-3930",
    "realpr-wtforms-923",
    "realpr-werkzeug-2637",
    "realpr-werkzeug-3116",
}

CANDS = [
    "realpr-oauthlib-889",
    "realpr-oauthlib-723",
    "realpr-oauthlib-416",
    "realpr-oauthlib-525",
    "realpr-click-3442",
    "realpr-boltons-362",
    "realpr-jinja-634",
    "realpr-jinja-637",
    "realpr-httpcore-353",
    "realpr-httpcore-420",
    "realpr-httpcore-880",
    "realpr-httpx-3068",
    "realpr-attrs-660",
    "realpr-attrs-392",
    "realpr-flask-4692",
    "realpr-flask-5812",
    "realpr-paramiko-2166",
    "realpr-quart-386",
    "realpr-scrapy-7524",
    "realpr-tornado-3596",
    "realpr-marshmallow-2733",
    "realpr-packaging-1267",
    "realpr-packaging-200",
    "realpr-rq-2386",
    "realpr-rq-2385",
    "realpr-rq-2350",
    "realpr-rq-1874",
    "realpr-charset-normalizer-715",
]


def log(msg: str) -> None:
    print(msg, flush=True)
    with (EV / "dual_smoke2.log").open("a") as fh:
        fh.write(msg + "\n")


def find(tid: str) -> Path | None:
    prefer = [
        "live_materials_m28b_retry2",
        "live_materials_m28b_cert",
        "live_materials_m27c_densify6",
        "live_materials_m27c_densify5",
        "live_materials_m27c_densify3",
        "live_materials_m27c_densify2",
        "live_materials_m27c_densify",
        "live_materials_m27c_cert2",
        "live_materials_m27_priority",
        "live_materials_m27_wave3",
        "live_materials_m27_pass4",
        "live_materials_m27",
        "live_materials_m22",
        "live_materials",
    ]
    for name in prefer:
        p = Path("datasets") / name / tid
        if (
            (p / "solution.patch").exists()
            and (p / "test.patch").exists()
            and (p / "meta.json").exists()
        ):
            return p
    for root in sorted(Path("datasets").glob("live_materials*")):
        p = root / tid
        if (
            (p / "solution.patch").exists()
            and (p / "test.patch").exists()
            and (p / "meta.json").exists()
        ):
            return p
    return None


def src_count(text: str) -> int:
    files: list[str] = []
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3][2:] if parts[3].startswith("b/") else parts[3]
                files.append(path)
    src: list[str] = []
    for path in files:
        low = path.lower()
        base = Path(path).name.lower()
        if "/tests/" in f"/{low}/" or "/test/" in f"/{low}/":
            continue
        if base.startswith("test_") or base.endswith("_test.py"):
            continue
        src.append(path)
    return len(set(src) or set(files))


def structure_ok(text: str) -> tuple[bool, int, int, int]:
    n = src_count(text)
    h = count_unified_diff_hunks(text)
    a = count_gold_added_lines(text)
    ok = (
        multi_file_floor_ok(source_files=n, added_lines=a, hunks=h)
        and h >= PRODUCT_SOURCE_HUNK_FLOOR
        and a >= PRODUCT_MIN_ADDED_LINES
    )
    return ok, n, h, a


def main() -> None:
    cache = CloneCache(root=Path("datasets/_clone_cache"))
    results: list[dict] = []
    survivors: list[str] = []
    if (EV / "survivors.json").exists():
        try:
            survivors = list(json.loads((EV / "survivors.json").read_text()))
        except Exception:
            survivors = []

    # auto-add untried floor-ok non-werkzeug mats on disk
    seen = set(CANDS) | set(HAVE)
    extra: list[str] = []
    for root in sorted(Path("datasets").glob("live_materials*")):
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if not d.is_dir() or not d.name.startswith("realpr-"):
                continue
            if d.name in seen or "werkzeug" in d.name:
                continue
            if not (
                (d / "solution.patch").exists()
                and (d / "test.patch").exists()
                and (d / "meta.json").exists()
            ):
                continue
            text = (d / "solution.patch").read_text(errors="replace")
            ok, _, _, _ = structure_ok(text)
            if ok:
                extra.append(d.name)
                seen.add(d.name)
    cands = list(dict.fromkeys(CANDS + extra))
    log(f"START2 {datetime.now(timezone.utc).isoformat()} cand={len(cands)} extra={extra[:25]}")

    for tid in cands:
        if tid in HAVE:
            continue
        src = find(tid)
        if not src:
            continue
        text = (src / "solution.patch").read_text(errors="replace")
        ok, n, h, a = structure_ok(text)
        if not ok:
            continue
        log(f"==== {tid} f={n} h={h} a={a} <- {src} ====")
        mr = Path(tempfile.mkdtemp(prefix="m2_", dir="/tmp"))
        work = Path(tempfile.mkdtemp(prefix=f"m2d_{tid}_", dir="/tmp"))
        try:
            shutil.copytree(src, mr / tid)
            if tid == "realpr-click-3442":
                meta = json.loads((mr / tid / "meta.json").read_text())
                tests = [
                    t
                    for t in (meta.get("tests") or meta.get("test_files") or [])
                    if "test_options" not in str(t)
                ]
                meta["tests"] = tests
                meta["test_files"] = tests
                (mr / tid / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            m = load_real_pr_materials(mr, rebuild_inventory=True)[0]
            base = _materialize_base_worktree(m, work=work, clone_cache=cache)
            host = _prepare_host_suite_env(
                base,
                language=m.language or "python",
                work_root=work / "host",
                task_id=tid,
            )
            tfiles = [
                t
                for t in (m.test_files or [])
                if str(t).endswith(".py")
                and "termui" not in str(t)
                and "test_options" not in str(t)
                and not str(t).endswith((".coveragerc", ".cfg", ".ini"))
            ]
            res = label_real_pr_dual_run(
                language=m.language or "python",
                base_repo=base,
                solution_patch=m.solution_patch,
                test_patch=m.test_patch,
                base_commit=m.base_commit,
                work_root=work / "dual",
                require_nonempty_f2p=True,
                allow_green_flake=True,
                min_f2p_nodes=5,
                python_executable=host.python if host.isolated else None,
                held_out_relative_paths=tfiles,
            )
            f2p = list(res.f2p_node_ids)
            p2p = list(res.p2p_node_ids)
            dirty = sum(
                1
                for node in f2p
                if "datasets." in node or "/tmp/" in node or "dual_run" in node
            )
            row = {
                "task_id": tid,
                "ok": True,
                "f2p": len(f2p),
                "p2p": len(p2p),
                "dirty": dirty,
                "sample": f2p[:3],
                "files": n,
                "hunks": h,
                "added": a,
                "src": str(src),
            }
            log(f"  OK f2p={len(f2p)} dirty={dirty} sample={f2p[:2]}")
            results.append(row)
            if len(f2p) >= 5 and dirty == 0:
                if tid not in survivors:
                    survivors.append(tid)
                dest = SURV_DIR / tid
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
                if tid == "realpr-click-3442":
                    meta = json.loads((dest / "meta.json").read_text())
                    meta["tests"] = tfiles
                    meta["test_files"] = tfiles
                    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
                (EV / f"{tid}.dual.json").write_text(
                    json.dumps({"f2p": f2p, "p2p": p2p[:40]}, indent=2)
                )
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            log(f"  FAIL {err[:300]}")
            results.append({"task_id": tid, "ok": False, "error": err[:600]})
            traceback.print_exc()
        finally:
            shutil.rmtree(work, ignore_errors=True)
            shutil.rmtree(mr, ignore_errors=True)
            (EV / "dual_smoke2.json").write_text(json.dumps(results, indent=2))
            (EV / "survivors2.json").write_text(json.dumps(survivors, indent=2))
            time.sleep(0.05)

    log(f"DONE survivors={survivors} n={len(survivors)}")
    print(json.dumps({"survivors": survivors, "n": len(survivors)}, indent=2))


if __name__ == "__main__":
    main()
