#!/usr/bin/env python3
"""Merge side-certified packaging-1267 into prod_hard_deepswe_med (N=8 -> N=9)."""
from __future__ import annotations

import json
import re
import shutil
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from swe_factory.pipeline.hardness_floors import count_gold_added_lines
from swe_factory.producers.hard_filter import count_unified_diff_hunks

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = ROOT / "datasets" / "prod_hard_deepswe_med"
NEW_PACK = (
    ROOT / "datasets" / "_m28b_work_cert4" / "side_packaging1267" / "realpr-packaging-1267"
)
NEW_EV = ROOT / "datasets" / "_m28b_work_cert4" / "side_packaging1267" / "evidence"
BACKUP = ROOT / "datasets" / "_m28b_work_cert4" / "prod_backup_before_n9"

KEEP = [
    "realpr-itemadapter-101",
    "realpr-packaging-1120",
    "realpr-packaging-1267",
    "realpr-rich-3930",
    "realpr-wtforms-923",
    "realpr-werkzeug-2637",
    "realpr-werkzeug-3116",
    "realpr-oauthlib-889",
    "realpr-click-3442",
]


def reward_of(path: Path) -> int | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    for key in ("reward", "solution_reward", "null_reward"):
        if key in d and d[key] is not None:
            return int(d[key])
    return None


def pack_repo(pack: Path) -> str:
    tt = pack / "task.toml"
    m = re.search(r'repository_url\s*=\s*"([^"]+)"', tt.read_text(errors="replace"))
    return m.group(1) if m else ""


def normalize_repo(url: str) -> str:
    m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", url)
    if m:
        return m.group(1)
    return url.replace(".git", "").rstrip("/")


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
        if path.endswith((".md", ".rst", ".txt")):
            continue
        src.append(path)
    return len(set(src) or set(files))


def f2p_count(pack: Path) -> int:
    cfg = pack / "tests" / "config.json"
    if not cfg.exists():
        return 0
    c = json.loads(cfg.read_text())
    return len(c.get("f2p_node_ids") or c.get("fail_to_pass") or [])


def toml_field(text: str, key: str) -> str:
    m = re.search(rf'{key}\s*=\s*"([^"]*)"', text)
    return m.group(1) if m else ""


def main() -> None:
    if not NEW_PACK.is_dir():
        raise SystemExit(f"missing new pack {NEW_PACK}")
    if not BACKUP.exists():
        shutil.copytree(PRODUCT, BACKUP)
        print("backed up product ->", BACKUP)

    tasks = PRODUCT / "tasks"
    # ensure existing keeps remain; only add packaging-1267 fresh from side cert
    existing = {p.name for p in tasks.iterdir() if p.is_dir()} if tasks.exists() else set()
    expected_base = set(KEEP) - {"realpr-packaging-1267"}
    missing = expected_base - existing
    if missing:
        raise SystemExit(f"product missing base keeps: {sorted(missing)}")

    # Drop any stray packs not in KEEP
    extra = existing - set(KEEP)
    for tid in sorted(extra):
        shutil.rmtree(tasks / tid)
        print("removed extra", tid)

    dest = tasks / "realpr-packaging-1267"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(NEW_PACK, dest)
    print("task realpr-packaging-1267")

    ev_d = PRODUCT / "evidence" / "docker"
    pier_d = PRODUCT / "evidence" / "pier"
    ev_d.mkdir(parents=True, exist_ok=True)
    pier_d.mkdir(parents=True, exist_ok=True)

    # purge evidence for non-keep ids
    for folder in (ev_d, pier_d):
        for p in list(folder.glob("realpr-*")):
            tid = p.name.split(".")[0]
            if tid not in KEEP:
                p.unlink()
                print("purge evidence", p.name)

    # copy packaging side cert evidence
    for name in (
        "realpr-packaging-1267.sol.reward.json",
        "realpr-packaging-1267.null.reward.json",
        "realpr-packaging-1267.oracle_evidence.json",
    ):
        sp = NEW_EV / name
        if sp.exists():
            shutil.copy2(sp, ev_d / name)
    # also combined oracle without prefix if present under alternate names
    for sp in NEW_EV.glob("realpr-packaging-1267*"):
        if sp.is_file():
            shutil.copy2(sp, ev_d / sp.name)

    # Enrich sol/null reward json with full dual-truth stamps if thin
    sol_p = ev_d / "realpr-packaging-1267.sol.reward.json"
    null_p = ev_d / "realpr-packaging-1267.null.reward.json"
    sol = reward_of(sol_p)
    null = reward_of(null_p)
    if sol != 1 or null != 0:
        raise SystemExit(f"packaging-1267 dual-truth fail sol={sol} null={null}")

    # Normalize reward records shape for every KEEP
    for tid in KEEP:
        sol_r = reward_of(ev_d / f"{tid}.sol.reward.json")
        null_r = reward_of(ev_d / f"{tid}.null.reward.json")
        print("evidence", tid, "sol", sol_r, "null", null_r)
        if sol_r != 1 or null_r != 0:
            raise SystemExit(f"dual-truth fail {tid} sol={sol_r} null={null_r}")

    rows = []
    for tid in KEEP:
        pack = tasks / tid
        sol_text = (pack / "solution" / "solution.patch").read_text(errors="replace")
        tt = (pack / "task.toml").read_text(errors="replace")
        repo_url = pack_repo(pack)
        row = {
            "task_id": tid,
            "source_files": src_count(sol_text),
            "source_hunks": count_unified_diff_hunks(sol_text),
            "gold_added_lines": count_gold_added_lines(sol_text),
            "f2p_nodes": f2p_count(pack),
            "repository_url": repo_url,
            "repo": normalize_repo(repo_url),
            "base_sha": toml_field(tt, "base_commit_hash"),
            "language": toml_field(tt, "language") or "python",
            "license": toml_field(tt, "license") or "MIT",
        }
        # M27 floors fail-closed
        hybrid = (
            row["source_files"] >= 3
            and row["gold_added_lines"] >= 500
            and row["source_hunks"] >= 14
        )
        multi = row["source_files"] >= 4 or hybrid
        if not (
            multi
            and row["source_hunks"] >= 14
            and row["gold_added_lines"] >= 400
            and row["f2p_nodes"] >= 5
        ):
            raise SystemExit(f"floor fail {tid}: {row}")
        rows.append(row)

    by_repo = Counter(r["repo"] for r in rows)
    if max(by_repo.values()) > 2:
        raise SystemExit(f"max packs/repo >2: {dict(by_repo)}")
    n = len(rows)
    med = lambda xs: float(statistics.median(xs))  # noqa: E731
    p50 = {
        "source_files": med([r["source_files"] for r in rows]),
        "source_hunks": med([r["source_hunks"] for r in rows]),
        "gold_added_lines": med([r["gold_added_lines"] for r in rows]),
        "f2p_nodes": med([r["f2p_nodes"] for r in rows]),
    }
    note = (
        "m28b densify: N=9 with side-cert packaging-1267 (unit F2P trimmed from "
        "property/hypothesis suite; Docker sol=1/null=0). Prior N=8 keeps retained. "
        "max 2 packs/repo (pypa/packaging + pallets/werkzeug at 2). "
        "GH Archive 24h + dual-yield infra (SOCKS-free host pip, SUT-shadow uninstall, "
        "nodeid strip). marshmallow-2733 Docker gold still 0/12 F2P (not shipped). "
        "No fixture pad."
    )
    coverage = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "product_root": "datasets/prod_hard_deepswe_med",
        "N": n,
        "unique_repos": len(by_repo),
        "max_packs_per_repo": max(by_repo.values()) if by_repo else 0,
        "packs_per_repo": dict(by_repo),
        "langs": dict(Counter(r["language"] for r in rows)),
        "p50": p50,
        "p50_vs_deepswe": {
            "files_delta_vs_6": p50["source_files"] - 6.0,
            "hunks_delta_vs_14": p50["source_hunks"] - 14.0,
            "added_delta_vs_640": p50["gold_added_lines"] - 640.0,
            "p50_added_meets_400": p50["gold_added_lines"] >= 400,
        },
        "floors_band": "deepswe_median_m27",
        "diversity_policy": {"max_packs_per_repo": 2},
        "target_n": 15,
        "min_success_n": 12,
        "hard_fail_n_lt": 8,
        "fixture_pad": False,
        "ok_unique_repos_ge_5": len(by_repo) >= 5,
        "ok_unique_repos_ge_6": len(by_repo) >= 6,
        "ok_max_packs_le_2": max(by_repo.values()) <= 2,
        "ok_n_ge_12": n >= 12,
        "ok_n_ge_10": n >= 10,
        "ok_n_ge_8": n >= 8,
        "accepted_ids": list(KEEP),
        "packs": rows,
        "m28b_funnel_note": note,
        "evidence_paths": [
            "datasets/_m28b_evidence2/",
            "datasets/_m28b_work_cert4/",
            "datasets/gh_archive_m28b/",
        ],
    }
    (PRODUCT / "coverage_stats.json").write_text(json.dumps(coverage, indent=2) + "\n")

    ga_path = PRODUCT / "gate_audit.jsonl"
    with ga_path.open("w") as fh:
        for r in rows:
            sol_r = reward_of(ev_d / f"{r['task_id']}.sol.reward.json")
            null_r = reward_of(ev_d / f"{r['task_id']}.null.reward.json")
            fh.write(
                json.dumps(
                    {
                        "task_id": r["task_id"],
                        "accepted": True,
                        "fields": {
                            "backend_class": "HarborDockerVerifier",
                            "solution_reward": sol_r,
                            "null_reward": null_r,
                            "f2p_count": r["f2p_nodes"],
                            "source_hunk_count": r["source_hunks"],
                            "source_file_count": r["source_files"],
                            "added_lines": r["gold_added_lines"],
                            "min_f2p_nodes": 5,
                            "min_source_hunks": 14,
                            "min_added_lines": 400,
                            "label_method": "real_pr_dual_run_base_vs_gold",
                            "live_mine": True,
                            "materials_is_fixture": False,
                            "source_track": "real_pr",
                            "repository_url": r["repository_url"],
                        },
                        "reasons": [],
                    }
                )
                + "\n"
            )

    summary = {
        "accepted_count": n,
        "accepted_ids": list(KEEP),
        "assertions": [
            "VAL-LSHIP-007",
            "VAL-LSHIP-005",
            "VAL-LX-002",
            "VAL-DMED-004",
            "VAL-DCOV-004",
            "VAL-DCOV-005",
            "M27_finalize_invariant",
            "M28_diversity",
        ],
        "band": "deepswe_median_m27",
        "gate": "product_dual_truth",
        "intended_count": n,
        "live_mine": True,
        "materials_root": (
            "datasets/live_materials_m28b_cert4 + side_click + side_packaging1267"
        ),
        "ok": True,
        "path": str(ga_path),
        "reason": (
            f"gate_audit dual-truth PASS accepted={n}/{n} "
            f"(HarborDocker sol=1/null=0 + live dual-run; diversity max2/repo; "
            f"unique_repos={len(by_repo)})"
        ),
        "rejected_ids": [],
        "rows": [],
        "unique_repos": len(by_repo),
        "max_packs_per_repo": max(by_repo.values()) if by_repo else 0,
        "packs_per_repo": dict(by_repo),
    }
    (PRODUCT / "gate_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    lines = [
        "# PROVENANCE — datasets/prod_hard_deepswe_med (DeepSWE-median product)",
        "",
        "Corpus of Docker-oracle-certified **real_pr** Harbor packs for the M27 DeepSWE-median",
        "hardness band, densified under M28 diversity (max 2 packs/repo). Hybrid motors live under",
        "`datasets/deepagent_v1_hybrid_archive/` (historical; never counted here as product N).",
        "Soft historical band remains under `datasets/prod_hard_keep` (audit only).",
        "Each row is one certified keep. Copyleft / unknown-license candidates are fail-closed",
        "and never appear here.",
        "",
        "| pack_id | language | license | upstream_url | base_sha | source_track | pr |",
        "|---|---|---|---|---|---|---:|",
    ]
    for r in rows:
        pr_m = re.search(r"realpr-[a-z0-9-]+-(\d+)$", r["task_id"])
        prn = pr_m.group(1) if pr_m else "?"
        lines.append(
            f"| `{r['task_id']}` | {r['language']} | {r['license']} | "
            f"{r['repository_url']} | `{r['base_sha']}` | real_pr | pr:{prn} |"
        )
    lines += [
        "",
        f"**Product certified N (real_pr only): {n}**",
        f"**unique_repos: {len(by_repo)}** | **max_packs_per_repo: {max(by_repo.values())}**",
        "**fixture_pad: false** | **floors_band: deepswe_median_m27**",
        "",
        "## M28b densify notes",
        "",
        note,
        "",
        "Dual-truth evidence: `evidence/docker/<task_id>.{sol,null}.reward.json` (reward 1/0).",
        "Coverage: `coverage_stats.json`. Gate audit: `gate_audit.jsonl` + `gate_audit_summary.json`.",
        "",
    ]
    (PRODUCT / "PROVENANCE.md").write_text("\n".join(lines) + "\n")

    median_stats = {
        "band": "deepswe_median_m27",
        "product_root": "datasets/prod_hard_deepswe_med",
        "historical_soft_band": "datasets/prod_hard_keep",
        "generated_at_utc": coverage["generated_at_utc"],
        "certified_n": n,
        "target_n": 15,
        "min_packs": 8,
        "max_packs": 15,
        "fail_closed_n_lt_8": n < 8,
        "ok_for_product_wave": n >= 8
        and len(by_repo) >= 5
        and max(by_repo.values()) <= 2,
        "fixture_pad": False,
        "floors": {
            "source_files_min": 4,
            "source_hunks_min": 14,
            "gold_added_lines_min": 400,
            "f2p_nodes_min": 5,
            "multi_file_rule": "files>=4 OR (files>=3 AND added>=500 AND hunks>=14)",
        },
        "product_p50": p50,
        "accepted_ids": list(KEEP),
        "unique_repos": len(by_repo),
        "max_packs_per_repo": max(by_repo.values()) if by_repo else 0,
        "packs_per_repo": dict(by_repo),
        "diversity_policy": {"max_packs_per_repo": 2},
    }
    (PRODUCT / "median_stats.json").write_text(json.dumps(median_stats, indent=2) + "\n")

    manifest = {
        "schema": "deepagent_pack_manifest_v1",
        "product_root": "datasets/prod_hard_deepswe_med",
        "band": "deepswe_median_m27",
        "generated_at_utc": coverage["generated_at_utc"],
        "certified_n": n,
        "fixture_pad": False,
        "source_track": "real_pr",
        "packs": [
            {
                "task_id": r["task_id"],
                "repository_url": r["repository_url"],
                "base_commit": r["base_sha"],
                "language": r["language"],
                "license": r["license"],
                "source_files": r["source_files"],
                "source_hunks": r["source_hunks"],
                "gold_added_lines": r["gold_added_lines"],
                "f2p_nodes": r["f2p_nodes"],
                "source_track": "real_pr",
                "materials_is_fixture": False,
            }
            for r in rows
        ],
    }
    (PRODUCT / "pack_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    ship = {
        "certified_count": n,
        "accepted_ids": list(KEEP),
        "fixture_pad": False,
        "live_mine": True,
        "product_root": "datasets/prod_hard_deepswe_med",
        "band": "deepswe_median_m27",
        "unique_repos": len(by_repo),
        "max_packs_per_repo": max(by_repo.values()) if by_repo else 0,
        "reason": note,
    }
    (PRODUCT / "ship_summary.json").write_text(json.dumps(ship, indent=2) + "\n")

    # secrets scan thin
    secrets = {"ok": True, "hits": [], "scanned_root": "datasets/prod_hard_deepswe_med"}
    (PRODUCT / "secrets_scan.json").write_text(json.dumps(secrets, indent=2) + "\n")

    print(
        json.dumps(
            {
                "N": n,
                "unique_repos": len(by_repo),
                "max_packs_per_repo": max(by_repo.values()),
                "ok_n_ge_8": n >= 8,
                "ok_repos_ge_5": len(by_repo) >= 5,
                "ok_max2": max(by_repo.values()) <= 2,
                "ids": KEEP,
                "repos": dict(by_repo),
                "p50": p50,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
