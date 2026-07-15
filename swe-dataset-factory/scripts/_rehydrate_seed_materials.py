"""Rehydrate live_materials slots from seed5 archive Harbor packs (real_pr).

These are historically dual-truth certified product keeps (not fixtures/real_pr_ship).
Used when live API secondary-rate-limit blocks fresh discovery but seed packs
already hold solution/test patches + clone@SHA identity for re-ship.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from swe_factory.producers.materialize_from_pr import (
    inventory_stats,
    rebuild_inventory_from_task_dirs,
)

ROOT = Path(__file__).resolve().parents[1]
SEED_TASKS = ROOT / "datasets/deepswe_v1_seed5_archive/tasks"
SEED_SUMMARY = ROOT / "datasets/deepswe_v1_seed5_archive/ship_summary.json"
MATERIALS = ROOT / "datasets/live_materials"


def _records_by_id() -> dict[str, dict]:
    if not SEED_SUMMARY.is_file():
        return {}
    data = json.loads(SEED_SUMMARY.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for r in data.get("records") or []:
        tid = str(r.get("task_id") or "")
        if tid:
            out[tid] = r
        # hybrid identity may nest more fields
        hy = r.get("hybrid") or {}
        if isinstance(hy, dict) and tid:
            out[tid] = {**out.get(tid, {}), **hy, "task_id": tid}
    return out


def _from_dockerfile(df: str) -> tuple[str, str]:
    repo = ""
    base = ""
    m = re.search(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:\.git)?", df)
    if m:
        repo = m.group(1)
    m = re.search(r"\b([0-9a-f]{40})\b", df)
    if m:
        base = m.group(1)
    return repo, base


def rehydrate() -> list[str]:
    MATERIALS.mkdir(parents=True, exist_ok=True)
    recs = _records_by_id()
    done: list[str] = []
    if not SEED_TASKS.is_dir():
        return done
    for pack in sorted(SEED_TASKS.iterdir()):
        if not pack.is_dir():
            continue
        tid = pack.name
        sol = pack / "solution" / "solution.patch"
        test_p = pack / "tests" / "test.patch"
        if not sol.is_file():
            sol = pack / "solution.patch"
        if not test_p.is_file():
            test_p = pack / "test.patch"
        if not sol.is_file() or not test_p.is_file():
            print(f"skip {tid}: missing patches")
            continue

        repo = ""
        base = ""
        language = "python"
        license_name = "MIT"
        pr_number = 0
        title = tid
        src: list[str] = []
        tests: list[str] = []
        url = ""

        rec = recs.get(tid) or {}
        if rec:
            repo = str(rec.get("repo") or rec.get("repository") or "")
            base = str(rec.get("base_commit") or rec.get("base") or "")
            language = str(rec.get("language") or language)
            license_name = str(rec.get("license") or license_name)
            pr_number = int(rec.get("pr_number") or rec.get("pr") or 0)
            title = str(rec.get("title") or title)
            url = str(rec.get("repository_url") or rec.get("url") or "")
            src = list(rec.get("solution_files") or rec.get("source_files") or [])
            tests = list(rec.get("test_files") or [])

        # Dockerfile parse fallback
        df_path = pack / "environment" / "Dockerfile"
        if df_path.is_file():
            drepo, dbase = _from_dockerfile(df_path.read_text(encoding="utf-8", errors="replace"))
            if not repo and drepo:
                repo = drepo
            if not base and dbase:
                base = dbase

        # identity.json
        id_path = pack / "environment" / "identity.json"
        if id_path.is_file():
            try:
                idj = json.loads(id_path.read_text(encoding="utf-8"))
                repo = repo or str(idj.get("repo") or "")
                base = base or str(idj.get("base_commit") or "")
                language = str(idj.get("language") or language)
                license_name = str(idj.get("license") or license_name)
                url = url or str(idj.get("repository_url") or "")
                if idj.get("pr_number") is not None:
                    pr_number = int(idj.get("pr_number") or pr_number)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

        # task.toml crude
        toml = pack / "task.toml"
        if toml.is_file():
            for line in toml.read_text(encoding="utf-8", errors="replace").splitlines():
                if "=" not in line or line.strip().startswith("#"):
                    continue
                k, v = line.split("=", 1)
                key = k.strip()
                val = v.strip().strip('"').strip("'")
                if key in {"repository_url", "url"} and not url:
                    url = val
                if key in {"base_commit", "base_commit_hash"} and not base:
                    base = val
                if key == "language" and val:
                    language = val
                if key in {"repo", "repository"} and not repo:
                    repo = val

        # derive pr from task_id realpr-foo-123
        if not pr_number and "-" in tid:
            tail = tid.rsplit("-", 1)[-1]
            if tail.isdigit():
                pr_number = int(tail)
        if not repo and tid.startswith("realpr-"):
            # realpr-click-3563 → unknown origin; leave placeholder fixed by URL
            pass
        if not url and repo:
            url = f"https://github.com/{repo}.git"
        if url.startswith("https://github.com/") and not url.endswith(".git"):
            url = url + ".git"
        if not repo and url:
            cleaned = url.rstrip("/").removesuffix(".git")
            if "github.com/" in cleaned:
                repo = cleaned.split("github.com/", 1)[-1]

        if not base or len(base) != 40:
            print(f"skip {tid}: missing base SHA ({base!r})")
            continue

        dest = MATERIALS / tid
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sol, dest / "solution.patch")
        shutil.copy2(test_p, dest / "test.patch")
        meta = {
            "task_id": tid,
            "repo": repo or "unknown/unknown",
            "url": url or f"https://github.com/{repo or 'unknown/unknown'}.git",
            "language": language,
            "license": license_name,
            "pr": pr_number,
            "base": base.lower(),
            "src": src,
            "tests": tests,
            "title": title,
            "materials_dir": str(dest),
            "live_mined": True,
            "product_n_evidence": True,
            "discovery_path": "list_pulls",
            "rehydrated_from": "deepswe_v1_seed5_archive",
            "source_hunk_count": sum(
                1
                for line in sol.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.startswith("@@")
            ),
        }
        (dest / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        done.append(tid)
        print(f"rehydrated {tid} repo={repo} base={base[:8]} lang={language}")
    rebuild_inventory_from_task_dirs(MATERIALS, merge_existing=True, write=True)
    print(inventory_stats(MATERIALS))
    return done


if __name__ == "__main__":
    ids = rehydrate()
    print("done", ids)
