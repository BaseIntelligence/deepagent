#!/usr/bin/env python3
"""Smoke dual-run on werkzeug/flask family after pytest-xprocess install."""
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

CANDS = [
    "realpr-werkzeug-2641",
    "realpr-werkzeug-2637",
    "realpr-werkzeug-2608",
    "realpr-flask-5812",
    "realpr-werkzeug-3116",
]
MAT = Path("datasets/live_materials_m27c_smoke_wk")
OUT = Path("datasets/real_pr_pool_m27c_smoke_wk.json")


def find_src(tid: str) -> Path | None:
    for root in [
        Path("datasets/live_materials_m27c_priority_cert"),
        Path("datasets/live_materials_m27_wave3"),
        Path("datasets/live_materials_m27_pass4"),
        Path("datasets/live_materials_m27_priority"),
        Path("datasets/live_materials_m27"),
        Path("datasets/live_materials_m22"),
        Path("datasets/live_materials"),
    ]:
        p = root / tid
        if p.is_dir() and (p / "solution.patch").exists():
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
            continue
        print(f"\n==== {tid} ====", flush=True)
        work = Path(tempfile.mkdtemp(prefix=f"smk_{tid}_", dir="/tmp"))
        try:
            base = _materialize_base_worktree(m, work=work, clone_cache=cache)
            _prepare_host_suite_env(base, language=m.language or "python")
            tfiles = list(getattr(m, "test_files", None) or [])
            green = run_python_suite(base, test_paths=tfiles or None)
            print(
                "GREEN",
                len(green.passed),
                len(green.failed),
                len(green.errors),
                green.returncode,
                flush=True,
            )
            if not green.passed:
                print(green.raw_tail[-500:], flush=True)
                out.append({"task_id": tid, "ok": False, "stage": "green0"})
                continue
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
            print("DUAL OK f2p", len(f2p), flush=True)
            out.append({"task_id": tid, "ok": True, "f2p": len(f2p)})
        except Exception as e:  # noqa: BLE001
            print("FAIL", type(e).__name__, e, flush=True)
            traceback.print_exc()
            out.append(
                {
                    "task_id": tid,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}"[:400],
                }
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
