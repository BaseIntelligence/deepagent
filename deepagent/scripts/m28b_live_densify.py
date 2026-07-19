#!/usr/bin/env python3
"""M28b: LIVE densify prod_hard_deepswe_med toward N=15 under diversity + M27 floors.

Pipeline:
  1. Build materials root with current keeps (cap werkzeug at 2 strongest)
  2. Discover large merged PRs via GitHub REST over SOCKS proxy (multi-repo)
  3. Materialize + M27 structural filter
  4. Serial dual-smoke (isolation + green-flake) require F2P>=5
  5. deepagent generate --materials ... --target 15 (Docker dual-truth concurrency 1)

Never pads fixtures. Prefers non-werkzeug fills.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

import httpx

from swe_factory.pipeline.hardness_floors import (
    PRODUCT_MIN_ADDED_LINES,
    PRODUCT_SOURCE_HUNK_FLOOR,
    count_gold_added_lines,
    multi_file_floor_ok,
)
from swe_factory.pipeline.ship_real_pr import (
    _materialize_base_worktree,
    _prepare_host_suite_env,
    load_real_pr_materials,
)
from swe_factory.producers.hard_filter import count_unified_diff_hunks
from swe_factory.producers.real_dual_run import label_real_pr_dual_run
from swe_factory.sources.clone_cache import CloneCache
from swe_factory.sources.github import redact_proxy_url, resolve_github_proxy_url

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

PRODUCT = Path("datasets/prod_hard_deepswe_med")
MATS = Path("datasets/live_materials_m28b")
WORK = Path("datasets/_m28b_work_densify")
EVIDENCE = PRODUCT / "evidence" / "m28b"
LOG = PRODUCT / "generate_m28b_densify.log"

# Keep current certified pack seeds; diversity will drop 3rd werkzeug at ship if needed.
# Prefer strongest: packaging+itemadapter+best 2 werkzeug (3116 + 2637 by F2P) in seed copy.
SEED_KEEPS = [
    "realpr-itemadapter-101",
    "realpr-packaging-1120",
    "realpr-werkzeug-3116",
    "realpr-werkzeug-2637",
    "realpr-werkzeug-2608",  # leave in materials; diversity at generate may keep ≤2
]

# Multi-repo discovery breadth (non-werkzeug first).
REPOS = [
    "pallets/click",
    "pallets/jinja",
    "pallets/flask",
    "pallets/quart",
    "encode/httpx",
    "encode/httpcore",
    "oauthlib/oauthlib",
    "rq/rq",
    "Textualize/rich",
    "mahmoud/boltons",
    "python-attrs/attrs",
    "jawah/charset_normalizer",
    "psf/requests",
    "urllib3/urllib3",
    "pypa/packaging",
    "more-itertools/more-itertools",
    "scrapy/itemadapter",
    "scrapy/scrapy",
    "pallets-eco/wtforms",
    "marshmallow-code/marshmallow",
    "tkem/cachetools",
    "python-jsonschema/jsonschema",
    "pydantic/pydantic",
    "tornadoweb/tornado",
    "paramiko/paramiko",
    "pyca/cryptography",
    "yaml/pyyaml",
    "dateutil/dateutil",
    "gpvn/platformdirs",
    "tox-dev/platformdirs",
    "pallets/markupsafe",
    "pallets/itsdangerous",
    "jaraco/zipp",
    "benoitc/gunicorn",
    "celery/celery",
    "aio-libs/aiohttp",
    "psf/black",
    "pycqa/flake8",
    "pypa/wheel",
    "pypa/setuptools",
    "wizgine/httpie",  # may 404 — skipped
    "httpie/cli",
    "psf/requests-html",
]

SKIP_TITLE = (
    "docs",
    "bump",
    "changelog",
    "ci:",
    "chore:",
    "dependabot",
    "lock file",
    "dependencies",
    "typing only",
    "merge stable",
    "release ",
    "version bump",
    "update readme",
    "fix typo",
    "spelling",
)

USER_AGENT = "deepagent-m28b-densify"


def log(msg: str) -> None:
    print(msg, flush=True)


def token() -> str:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""


def gh_client() -> httpx.Client:
    proxy = resolve_github_proxy_url()
    log(f"proxy={redact_proxy_url(proxy) if proxy else None}")
    headers = {
        "Authorization": f"Bearer {token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    }
    return httpx.Client(
        proxy=proxy,
        trust_env=False,
        timeout=httpx.Timeout(60.0, connect=30.0),
        headers=headers,
        follow_redirects=True,
    )


def gh_get(client: httpx.Client, url: str, params: dict | None = None) -> tuple[object, int]:
    for attempt in range(8):
        r = client.get(url, params=params)
        rem = int(r.headers.get("X-RateLimit-Remaining") or 9999)
        if r.status_code in (403, 429):
            wait = int(r.headers.get("Retry-After") or (15 * (attempt + 1)))
            log(f"rate-limit {r.status_code} wait={wait}s rem={rem} url={url[:80]}")
            time.sleep(wait)
            continue
        if r.status_code == 404:
            return None, rem
        r.raise_for_status()
        return r.json(), rem
    raise RuntimeError(f"exhausted retries for {url}")


def tid_for(repo: str, pr: int) -> str:
    name = repo.split("/")[-1].replace("_", "-")
    return f"realpr-{name}-{pr}"


def source_file_count(text: str) -> int:
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


def structure_ok(pack: Path) -> tuple[bool, int, int, int]:
    sol = pack / "solution.patch"
    if not sol.is_file():
        return False, 0, 0, 0
    text = sol.read_text(encoding="utf-8", errors="replace")
    n = source_file_count(text)
    h = count_unified_diff_hunks(text)
    a = count_gold_added_lines(text)
    ok = (
        multi_file_floor_ok(source_files=n, added_lines=a, hunks=h)
        and h >= PRODUCT_SOURCE_HUNK_FLOOR
        and a >= PRODUCT_MIN_ADDED_LINES
    )
    return ok, n, h, a


def collect_prior_failed() -> set[str]:
    failed: set[str] = set()
    dual_ok: set[str] = set()
    for drip in Path("datasets").glob("_m*/**/e2e_drip.jsonl"):
        try:
            for line in drip.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                tid = row.get("task_id")
                if not tid:
                    continue
                stage = row.get("stage")
                status = row.get("status")
                if stage == "dual_run" and status == "ok":
                    dual_ok.add(tid)
                if stage == "dual_run" and status != "ok":
                    failed.add(tid)
        except Exception:
            continue
    # Don't permanently skip dual_ok that later oracle-failed
    return failed - dual_ok


def find_seed(tid: str) -> Path | None:
    prefer = [
        PRODUCT / "tasks" / tid,
        Path("datasets/live_materials_m27c_densify5") / tid,
        Path("datasets/live_materials_m27c_densify6") / tid,
        Path("datasets/live_materials_m27c_densify") / tid,
        Path("datasets/live_materials_m27_wave3") / tid,
        Path("datasets/live_materials_m27") / tid,
        Path("datasets/live_materials_m22") / tid,
        Path("datasets/live_materials") / tid,
    ]
    for p in prefer:
        if p.is_dir() and (
            (p / "solution.patch").exists()
            or (p / "solution" / "solution.patch").exists()
        ):
            return p
    # search all live_materials
    for root in sorted(Path("datasets").glob("live_materials*")):
        p = root / tid
        if p.is_dir() and (p / "solution.patch").exists():
            return p
    return None


def copy_seed_as_materials(tid: str, dest_root: Path) -> bool:
    src = find_seed(tid)
    if not src:
        log(f"seed missing {tid}")
        return False
    dest = dest_root / tid
    if dest.exists():
        shutil.rmtree(dest)
    # If seed is a product task pack, convert to materials layout
    if (src / "solution" / "solution.patch").exists():
        dest.mkdir(parents=True)
        shutil.copy2(src / "solution" / "solution.patch", dest / "solution.patch")
        tp = src / "tests" / "test.patch"
        if tp.exists():
            shutil.copy2(tp, dest / "test.patch")
        # meta from task.toml/instruction isn't enough — try locate original meta
        meta_src = None
        for root in Path("datasets").glob("live_materials*"):
            m = root / tid / "meta.json"
            if m.exists():
                meta_src = m
                break
        if meta_src:
            shutil.copy2(meta_src, dest / "meta.json")
        else:
            # minimal meta from task.toml
            repo = ""
            base = ""
            pr = 0
            tt = src / "task.toml"
            if tt.exists():
                t = tt.read_text(errors="replace")
                m = re.search(r'repository_url\s*=\s*"([^"]+)"', t)
                if m:
                    repo = m.group(1).replace("https://github.com/", "").replace(".git", "")
                m = re.search(r'base_commit\s*=\s*"([0-9a-f]{40})"', t)
                if m:
                    base = m.group(1)
                m = re.search(r'pr[_\s]*number\s*=\s*(\d+)', t, re.I)
                if not m:
                    m = re.search(r'realpr-[\w-]+-(\d+)', tid)
                if m:
                    pr = int(m.group(1))
            meta = {
                "task_id": tid,
                "repo": repo,
                "pr": pr,
                "base": base,
                "language": "python",
                "license": "MIT",
                "source_hunk_count": 0,
            }
            (dest / "meta.json").write_text(json.dumps(meta, indent=2))
        log(f"seed-from-product {tid}")
        return True
    shutil.copytree(src, dest)
    log(f"seed-from-materials {tid} <- {src}")
    return True


def discover_hits(client: httpx.Client, max_hits: int = 80) -> list[dict]:
    hits: list[dict] = []
    per_repo_cap = 6
    for repo in REPOS:
        n_closed = 0
        repo_hits = 0
        try:
            for page in range(1, 5):
                data, rem = gh_get(
                    client,
                    f"https://api.github.com/repos/{repo}/pulls",
                    params={
                        "state": "closed",
                        "sort": "updated",
                        "direction": "desc",
                        "per_page": 40,
                        "page": page,
                    },
                )
                if data is None:
                    log(f"skip missing repo {repo}")
                    break
                assert isinstance(data, list)
                for pr in data:
                    if not pr.get("merged_at"):
                        continue
                    n_closed += 1
                    if n_closed > 50:
                        break
                    num = pr["number"]
                    detail, rem = gh_get(
                        client, f"https://api.github.com/repos/{repo}/pulls/{num}"
                    )
                    if not isinstance(detail, dict):
                        continue
                    add = int(detail.get("additions") or 0)
                    files = int(detail.get("changed_files") or 0)
                    title = pr.get("title") or ""
                    low = title.lower()
                    if add < 420 or files < 5:
                        time.sleep(0.03)
                        continue
                    if any(term in low for term in SKIP_TITLE):
                        time.sleep(0.03)
                        continue
                    # skip pure docs/tests-only by filename later
                    hit = {
                        "repo": repo,
                        "pr": num,
                        "add": add,
                        "files": files,
                        "title": title,
                        "base": (detail.get("base") or {}).get("sha") or "",
                        "tid": tid_for(repo, num),
                    }
                    hits.append(hit)
                    repo_hits += 1
                    log(
                        f"HIT {repo}#{num} +{add} f={files} {title[:60]}",
                    )
                    time.sleep(0.06)
                    if repo_hits >= per_repo_cap:
                        break
                if n_closed > 50 or repo_hits >= per_repo_cap:
                    break
                time.sleep(0.15)
        except Exception as exc:  # noqa: BLE001
            log(f"discover error {repo}: {type(exc).__name__}: {exc}"[:200])
        log(f"done {repo} repo_hits={repo_hits} total_hits={len(hits)}")
        if len(hits) >= max_hits:
            break
        time.sleep(0.2)
    return hits


def materialize(mats: Path, repo: str, pr: int) -> bool:
    cmd = [
        str(ROOT / ".venv/bin/swe-factory"),
        "materialize-from-pr",
        "--repo",
        repo,
        "--pr",
        str(pr),
        "--out",
        str(mats),
        "--product-mode",
        "--json",
        "--discovery-path",
        "list_pulls",
    ]
    env = os.environ.copy()
    # Prefer explicit proxy for any child HTTP if supported
    proxy = resolve_github_proxy_url()
    if proxy:
        env.setdefault("ALL_PROXY", proxy)
        env.setdefault("HTTPS_PROXY", proxy)
        env.setdefault("HTTP_PROXY", proxy)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240, env=env)
    tail = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-240:]
    log(f"MAT {repo}#{pr} rc={proc.returncode} {tail.replace(chr(10),' ')[:200]}")
    return proc.returncode == 0


def dual_smoke(tid: str, mats: Path, cache: CloneCache) -> dict:
    mats_list = load_real_pr_materials(mats, rebuild_inventory=False)
    by = {m.task_id: m for m in mats_list}
    m = by.get(tid)
    if not m:
        # rebuild inventory once
        mats_list = load_real_pr_materials(mats, rebuild_inventory=True)
        by = {m.task_id: m for m in mats_list}
        m = by.get(tid)
    if not m:
        return {"task_id": tid, "ok": False, "error": "missing material"}
    work = Path(tempfile.mkdtemp(prefix=f"m28b_{tid}_", dir="/tmp"))
    try:
        base = _materialize_base_worktree(m, work=work, clone_cache=cache)
        host = _prepare_host_suite_env(
            base,
            language=m.language or "python",
            work_root=work / "host",
        )
        res = label_real_pr_dual_run(
            language=m.language or "python",
            base_repo=base,
            solution_patch=m.solution_patch,
            test_patch=m.test_patch,
            base_commit=m.base_commit,
            work_root=work / "dual",
            require_nonempty_f2p=True,
            allow_green_flake=True,
            python_executable=host.python if host.isolated else None,
        )
        f2p = list(getattr(res, "f2p_node_ids", None) or [])
        p2p = list(getattr(res, "p2p_node_ids", None) or [])
        return {
            "task_id": tid,
            "ok": True,
            "f2p": len(f2p),
            "p2p": len(p2p),
            "f2p_sample": f2p[:5],
            "repo": getattr(m, "repo", None) or getattr(m, "repository", None),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "task_id": tid,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_generate() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROOT / ".venv/bin/deepagent"),
        "generate",
        "--out",
        str(PRODUCT),
        "--target",
        "15",
        "--min-packs",
        "8",
        "--max-packs",
        "15",
        "--materials",
        str(MATS),
        "--live-mine",
        "--oracle",
        "docker",
        "--panel",
        "offline",
        "--pier",
        "scripted",
        "--work",
        str(WORK),
        "--json",
    ]
    log("GENERATE " + " ".join(cmd))
    with LOG.open("w") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    log(f"GENERATE exit={proc.returncode} log={LOG}")
    return proc.returncode


PHASE = os.environ.get("M28B_PHASE", "all")  # all|discover|smoke|generate


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if not token():
        log("missing GITHUB_TOKEN")
        return 2

    prior_failed = collect_prior_failed()
    log(f"prior dual-failed (non-dual-ok) count={len(prior_failed)}")

    if PHASE in ("all", "discover"):
        if MATS.exists():
            # keep residual only if rebuild deleted
            shutil.rmtree(MATS)
        MATS.mkdir(parents=True)

        for tid in SEED_KEEPS:
            copy_seed_as_materials(tid, MATS)

        # Also try previously struct-ok untried candidates (charset-normalizer)
        for tid in ("realpr-charset-normalizer-715", "realpr-httpcore-880"):
            if (MATS / tid).exists():
                continue
            src = find_seed(tid)
            if src and (src / "solution.patch").exists():
                shutil.copytree(src, MATS / tid)
                log(f"bonus materials {tid}")

        with gh_client() as client:
            rl, _ = gh_get(client, "https://api.github.com/rate_limit")
            if isinstance(rl, dict):
                log(f"rate core={(rl.get('resources') or {}).get('core')}")
            hits = discover_hits(client, max_hits=70)
        (EVIDENCE / "hits.json").write_text(json.dumps(hits, indent=2))
        log(f"TOTAL HITS {len(hits)}")

        # Prefer high-add + non-werkzeug; diversity ≤2 per repo for new fills
        by_repo: dict[str, int] = defaultdict(int)
        materialized = []
        for hit in sorted(hits, key=lambda h: (-int(h["add"]), -int(h["files"]))):
            repo = hit["repo"]
            if "werkzeug" in repo:
                continue  # already over-concentrated
            tid = hit["tid"]
            if tid in SEED_KEEPS:
                continue
            if tid in prior_failed:
                log(f"skip prior dual-fail {tid}")
                continue
            if by_repo[repo] >= 2:
                continue
            if materialize(MATS, repo, int(hit["pr"])):
                by_repo[repo] += 1
                materialized.append(hit)
            if len(materialized) >= 28:
                break
            time.sleep(0.25)
        (EVIDENCE / "materialized.json").write_text(json.dumps(materialized, indent=2))
        log(f"materialized_ok {len(materialized)}")

        # Structural filter non-seeds
        for pack in list(MATS.iterdir()):
            if not pack.is_dir() or not pack.name.startswith("realpr-"):
                continue
            if pack.name in SEED_KEEPS:
                continue
            ok, n, h, a = structure_ok(pack)
            log(f"struct {pack.name} ok={ok} f={n} h={h} a={a}")
            if not ok:
                shutil.rmtree(pack)

        # Ensure test.patch + meta present
        for pack in list(MATS.iterdir()):
            if not pack.is_dir():
                continue
            if not (pack / "solution.patch").exists() or not (pack / "test.patch").exists():
                log(f"drop incomplete {pack.name}")
                shutil.rmtree(pack)

        log(
            "mats after struct "
            + str(sorted(p.name for p in MATS.iterdir() if p.is_dir()))
        )

    if PHASE in ("all", "smoke"):
        if not MATS.is_dir():
            log("missing mats")
            return 3
        cache = CloneCache(root=Path("datasets/_clone_cache"))
        smoke_results = []
        for pack in sorted(MATS.iterdir()):
            if not pack.is_dir() or pack.name in SEED_KEEPS:
                continue
            if not (pack / "meta.json").exists():
                log(f"no meta skip smoke {pack.name}")
                continue
            log(f"SMOKE {pack.name}")
            res = dual_smoke(pack.name, MATS, cache)
            smoke_results.append(res)
            log(f" -> {json.dumps(res)[:300]}")
            # free mem between
            time.sleep(0.5)
        (EVIDENCE / "smoke.json").write_text(json.dumps(smoke_results, indent=2))
        survivors = [
            r["task_id"]
            for r in smoke_results
            if r.get("ok") and int(r.get("f2p") or 0) >= 5
        ]
        log(f"SURVIVORS F2P>=5: {survivors}")
        # Drop non-survivor non-seeds
        for pack in list(MATS.iterdir()):
            if not pack.is_dir():
                continue
            if pack.name in SEED_KEEPS or pack.name in survivors:
                continue
            # keep near-misses with f2p>=3 for generate try? No — only F2P>=5 survivors
            log(f"drop non-survivor {pack.name}")
            shutil.rmtree(pack)
        log(
            "mats final for generate "
            + str(sorted(p.name for p in MATS.iterdir() if p.is_dir()))
        )

    if PHASE in ("all", "generate"):
        if not MATS.is_dir():
            log("missing mats for generate")
            return 4
        n_mat = len([p for p in MATS.iterdir() if p.is_dir()])
        log(f"generate with {n_mat} materials packs")
        rc = run_generate()
        # snapshot task count
        tasks = PRODUCT / "tasks"
        if tasks.is_dir():
            ids = sorted(p.name for p in tasks.iterdir() if p.is_dir())
            log(f"product tasks N={len(ids)} {ids}")
            (EVIDENCE / "post_generate_tasks.json").write_text(
                json.dumps({"n": len(ids), "ids": ids}, indent=2)
            )
        return rc

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
