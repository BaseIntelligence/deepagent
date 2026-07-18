#!/usr/bin/env python3
"""Smoke dual-run after host-env fix on promising M27c family packs."""
from __future__ import annotations

import json
import shutil
import tempfile
import traceback
from pathlib import Path

from swe_factory.pipeline.ship_real_pr import (
    _materialize_base_worktree,
    _prepare_host_suite_env,
    load_real_pr_materials,
)
from swe_factory.producers.harbor_labeling import run_python_suite
from swe_factory.producers.real_dual_run import label_real_pr_dual_run
from swe_factory.sources.clone_cache import CloneCache

MAT = Path("datasets/live_materials_m27c_smoke")
OUT = Path("datasets/real_pr_pool_m27c_smoke_dual.json")

CANDS = [
    "realpr-marshmallow-2733",
    "realpr-werkzeug-2641",
    "realpr-werkzeug-2637",
    "realpr-werkzeug-2608",
    "realpr-flask-5812",
    "realpr-attrs-660",
    "realpr-rq-1874",
    "realpr-tornado-3596",
    "realpr-itemadapter-101",
]


def find_src(tid: str) -> Path | None:
    prefer = [
        Path("datasets/live_materials_m27c_recent"),
        Path("datasets/live_materials_m27c_priority_cert"),
        Path("datasets/live_materials_m27c_cert2"),
        Path("datasets/live_materials_m27_wave3"),
        Path("datasets/live_materials_m27_pass4"),
        Path("datasets/live_materials_m27_priority"),
        Path("datasets/live_materials_m27"),
        Path("datasets/live_materials_m22"),
        Path("datasets/live_materials"),
    ]
    for root in prefer:
        p = root / tid
        if p.is_dir() and (p / "solution.patch").exists() and (p / "test.patch").exists():
            return p
    return None


def main() -> None:
    if MAT.exists():
        shutil.rmtree(MAT)
    MAT.mkdir(parents=True)
    for tid in CANDS:
        src = find_src(tid)
        if not src:
            print("missing", tid, flush=True)
            continue
        shutil.copytree(src, MAT / tid)
        print("copy", tid, "from", src, flush=True)

    mats = load_real_pr_materials(MAT, rebuild_inventory=True)
    by = {m.task_id: m for m in mats}
    cache = CloneCache(root=Path("datasets/_clone_cache"))
    out: list[dict] = []

    for tid in CANDS:
        m = by.get(tid)
        if not m:
            print("skip missing mat", tid, flush=True)
            continue
        print(f"\n==== {tid} ====", flush=True)
        work = Path(tempfile.mkdtemp(prefix=f"smoke_{tid}_", dir="/tmp"))
        try:
            base = _materialize_base_worktree(m, work=work, clone_cache=cache)
            _prepare_host_suite_env(base, language=m.language or "python")
            tfiles = list(getattr(m, "test_files", None) or [])
            try:
                green = run_python_suite(base, test_paths=tfiles or None)
                print(
                    "GREEN pass",
                    len(green.passed),
                    "fail",
                    len(green.failed),
                    "err",
                    len(green.errors),
                    "rc",
                    green.returncode,
                    flush=True,
                )
                if len(green.passed) == 0:
                    print("tail", green.raw_tail[-700:], flush=True)
                    out.append(
                        {
                            "task_id": tid,
                            "ok": False,
                            "stage": "green0",
                            "error": green.raw_tail[-500:],
                        }
                    )
                    continue
            except Exception as e:  # noqa: BLE001
                print("GREEN ERR", type(e).__name__, e, flush=True)
                out.append(
                    {
                        "task_id": tid,
                        "ok": False,
                        "stage": "green",
                        "error": f"{type(e).__name__}: {e}"[:350],
                    }
                )
                continue
            try:
                res = label_real_pr_dual_run(
                    language=m.language or "python",
                    base_repo=base,
                    solution_patch=m.solution_patch,
                    test_patch=m.test_patch,
                    base_commit=m.base_commit,
                    work_root=work / "dual",
                    require_nonempty_f2p=True,
                )
                f2p = list(
                    getattr(res, "f2p_node_ids", None)
                    or getattr(res, "fail_to_pass", None)
                    or []
                )
                print("DUAL OK f2p", len(f2p), "sample", f2p[:3], flush=True)
                out.append({"task_id": tid, "ok": True, "f2p": len(f2p), "f2p_sample": f2p[:5]})
            except Exception as e:  # noqa: BLE001
                print("DUAL FAIL", type(e).__name__, e, flush=True)
                out.append(
                    {
                        "task_id": tid,
                        "ok": False,
                        "stage": "dual",
                        "error": f"{type(e).__name__}: {e}"[:350],
                    }
                )
        except Exception as e:  # noqa: BLE001
            print("SETUP", type(e).__name__, e, flush=True)
            traceback.print_exc()
            out.append(
                {
                    "task_id": tid,
                    "ok": False,
                    "stage": "setup",
                    "error": f"{type(e).__name__}: {e}"[:350],
                }
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
