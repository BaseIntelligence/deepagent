#!/usr/bin/env python3
"""Write coverage_stats.json for current prod_hard_deepswe_med (honest N)."""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from swe_factory.pipeline.hardness_floors import count_gold_added_lines
from swe_factory.pipeline.repo_diversity import normalize_upstream_repo
from swe_factory.producers.hard_filter import count_unified_diff_hunks

PRODUCT = Path("datasets/prod_hard_deepswe_med")


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


def med(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def main() -> int:
    summary = json.loads((PRODUCT / "gate_audit_summary.json").read_text())
    by = {r["task_id"]: r for r in summary.get("rows") or [] if r.get("task_id")}
    rows = []
    for pack in sorted((PRODUCT / "tasks").iterdir()):
        if not pack.is_dir() or not pack.name.startswith("realpr-"):
            continue
        text = (pack / "solution" / "solution.patch").read_text(errors="replace")
        n = src_count(text)
        h = count_unified_diff_hunks(text)
        a = count_gold_added_lines(text)
        fields = (by.get(pack.name) or {}).get("fields") or {}
        f2p = int(fields.get("f2p_count") or 0)
        url = fields.get("repository_url") or ""
        if not url:
            tt = (pack / "task.toml").read_text(errors="replace")
            m = re.search(r'repository_url\s*=\s*"([^"]+)"', tt)
            url = m.group(1) if m else ""
        rows.append(
            {
                "task_id": pack.name,
                "source_files": n,
                "source_hunks": h,
                "gold_added_lines": a,
                "f2p_nodes": f2p,
                "repository_url": url,
            }
        )
        print(pack.name, "f", n, "h", h, "a", a, "f2p", f2p)

    n_count = len(rows)
    p50 = {
        "source_files": med([float(r["source_files"]) for r in rows]),
        "source_hunks": med([float(r["source_hunks"]) for r in rows]),
        "gold_added_lines": med([float(r["gold_added_lines"]) for r in rows]),
        "f2p_nodes": med([float(r["f2p_nodes"]) for r in rows]),
    }
    repo = Counter(
        normalize_upstream_repo(r["repository_url"]) or r["repository_url"] for r in rows
    )
    max_per = max(repo.values()) if repo else 0
    coverage = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "product_root": "datasets/prod_hard_deepswe_med",
        "N": n_count,
        "unique_repos": len(repo),
        "max_packs_per_repo": max_per,
        "packs_per_repo": dict(sorted(repo.items())),
        "langs": {"python": n_count},
        "p50": p50,
        "p50_vs_deepswe": {
            "files_delta_vs_6": p50["source_files"] - 6.0,
            "hunks_delta_vs_14": p50["source_hunks"] - 14.0,
            "added_delta_vs_640": p50["gold_added_lines"] - 640.0,
            "p50_added_meets_400": p50["gold_added_lines"] >= 400.0,
        },
        "floors_band": "deepswe_median_m27",
        "diversity_policy": {"max_packs_per_repo": 2},
        "target_n": 15,
        "min_success_n": 12,
        "hard_fail_n_lt": 8,
        "fixture_pad": False,
        "ok_unique_repos_ge_6": len(repo) >= 6,
        "ok_max_packs_le_2": max_per <= 2,
        "ok_n_ge_12": n_count >= 12,
        "ok_n_ge_8": n_count >= 8,
        "accepted_ids": sorted(r["task_id"] for r in rows),
        "keeps_detail": rows,
        "m28b_funnel_note": (
            "LIVE densify via SOCKS list_pulls + Search + historical struct-ok materials "
            "dual-smoke under M27f isolation+green-flake yielded zero F2P>=5 dual-truth "
            "keeps beyond baseline. No fixture pad. Product remains N=5 dual-truth. "
            "Fail-closed vs N>=8 hard densify bar."
        ),
        "evidence_paths": [
            "datasets/_m28b_evidence/",
            "datasets/_m28b_evidence/fresh_smoke.json",
            "datasets/_m28b_evidence/retry_gen_smoke.json",
            "datasets/_m28b_evidence/hits.json",
        ],
    }
    (PRODUCT / "coverage_stats.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print("wrote coverage_stats N=", n_count, "p50_f2p=", p50["f2p_nodes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
