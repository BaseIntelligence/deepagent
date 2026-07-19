#!/usr/bin/env python3
"""M28b fresh wave: SOCKS materialize -> no-proxy dual-smoke (pip install .) -> side generate -> merge.

Preserves product seeds until dual-truth side packs merge under diversity max 2/repo.
Evidence under datasets/_m28b_evidence (outside product tree).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from swe_factory.pipeline.hardness_floors import (
    PRODUCT_MIN_ADDED_LINES,
    PRODUCT_SOURCE_HUNK_FLOOR,
    count_gold_added_lines,
    multi_file_floor_ok,
)
from swe_factory.pipeline.repo_diversity import (
    DEFAULT_MAX_PACKS_PER_REPO,
    apply_max_packs_per_repo,
    normalize_upstream_repo,
)
from swe_factory.pipeline.ship_real_pr import (
    HOST_SUITE_COMMON_DEPS,
    _materialize_base_worktree,
    _prepare_host_suite_env,
    build_host_suite_pip_command,
    load_real_pr_materials,
)
from swe_factory.producers.hard_filter import count_unified_diff_hunks
from swe_factory.producers.real_dual_run import label_real_pr_dual_run
from swe_factory.sources.clone_cache import CloneCache
from swe_factory.sources.github import redact_proxy_url, resolve_github_proxy_url

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

PRODUCT = Path("datasets/prod_hard_deepswe_med")
MAT = Path("datasets/live_materials_m28b_fresh")
WORK = Path("datasets/_m28b_work_fresh")
OUT_SIDE = WORK / "product_out"
EV = Path("datasets/_m28b_evidence")

PRODUCT_SEEDS = {
    "realpr-itemadapter-101",
    "realpr-packaging-1120",
    "realpr-werkzeug-2608",
    "realpr-werkzeug-2637",
    "realpr-werkzeug-3116",
}

# Backup PR list if hits.json missing. Non-werkzeug first.
CAND_PRS: list[tuple[str, int]] = [
    ("pypa/packaging", 1267),
    ("pypa/packaging", 1298),
    ("scrapy/itemadapter", 91),
    ("pallets/click", 2980),
    ("pallets/jinja", 1412),
    ("pallets/flask", 4692),
    ("encode/httpx", 3672),
    ("encode/httpx", 3319),
    ("encode/httpcore", 880),
    ("oauthlib/oauthlib", 889),
    ("oauthlib/oauthlib", 881),
    ("rq/rq", 2420),
    ("rq/rq", 2406),
    ("rq/rq", 2351),
    ("Textualize/rich", 2977),
    ("Textualize/rich", 2567),
    ("python-attrs/attrs", 1454),
    ("python-attrs/attrs", 1457),
    ("jawah/charset_normalizer", 715),
    ("mahmoud/boltons", 362),
    ("more-itertools/more-itertools", 1136),
    ("pallets-eco/wtforms", 921),
    ("marshmallow-code/marshmallow", 2733),
    ("tkem/cachetools", 385),
    ("tox-dev/platformdirs", 491),
    ("benoitc/gunicorn", 3614),
    ("chardet/chardet", 350),
    ("chardet/chardet", 352),
    ("pypa/wheel", 655),
    ("scrapy/scrapy", 6546),
    ("urllib3/urllib3", 3132),
    ("idna/idna", 246),
    ("hynek/structlog", 285),
    ("psf/requests", 6706),
    ("pallets/markupsafe", 400),
    ("jaraco/zipp", 122),
]


def log(msg: str) -> None:
    print(msg, flush=True)
    EV.mkdir(parents=True, exist_ok=True)
    with (EV / "fresh_wave.log").open("a") as fh:
        fh.write(msg + "\n")


def load_token() -> str:
    env_tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if env_tok:
        return env_tok
    t = Path(".env").read_text()
    m = re.search(r"^GITHUB_TOKEN=(.+)$", t, re.M)
    return m.group(1).strip().strip('"').strip("'") if m else ""


def noproxy_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    for k in list(env):
        ku = k.upper()
        if ku.endswith("_PROXY") or ku == "OXYLABS_PROXY_URL":
            env.pop(k, None)
    return env


def socks_env(token: str) -> dict[str, str]:
    env = noproxy_env()
    env["GITHUB_TOKEN"] = token
    env["GH_TOKEN"] = token
    proxy = resolve_github_proxy_url()
    if proxy:
        # Prefer OXYLABS only for github client inside swe-factory (may also read ALL_PROXY)
        env["OXYLABS_PROXY_URL"] = proxy
        env["ALL_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
        env["HTTP_PROXY"] = proxy
    return env


def tid_for(repo: str, pr: int) -> str:
    return f"realpr-{repo.split('/')[-1].replace('_', '-')}-{pr}"


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


def structure_ok_text(text: str) -> tuple[bool, int, int, int]:
    n = src_count(text)
    h = count_unified_diff_hunks(text)
    a = count_gold_added_lines(text)
    ok = (
        multi_file_floor_ok(source_files=n, added_lines=a, hunks=h)
        and h >= PRODUCT_SOURCE_HUNK_FLOOR
        and a >= PRODUCT_MIN_ADDED_LINES
    )
    return ok, n, h, a


def materialize_one(repo: str, pr: int, token: str) -> Path | None:
    before = {p.name for p in MAT.iterdir() if p.is_dir()} if MAT.exists() else set()
    cmd = [
        str(ROOT / ".venv/bin/swe-factory"),
        "materialize-from-pr",
        "--repo",
        repo,
        "--pr",
        str(pr),
        "--out",
        str(MAT),
        "--product-mode",
        "--json",
        "--discovery-path",
        "list_pulls",
    ]
    env = socks_env(token)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240, env=env)
    tail = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-220:].replace("\n", " ")
    log(f"MAT {repo}#{pr} rc={proc.returncode} {tail[:180]}")
    if proc.returncode != 0:
        return None
    after = {p.name for p in MAT.iterdir() if p.is_dir()}
    new = after - before
    if not new:
        # maybe same name rewritten
        guess = MAT / tid_for(repo, pr)
        return guess if guess.exists() else None
    # pick first new dir
    name = sorted(new)[0]
    return MAT / name


def dual_smoke(tid: str, cache: CloneCache) -> dict:
    clean = noproxy_env()
    mats_list = load_real_pr_materials(MAT, rebuild_inventory=True)
    by = {m.task_id: m for m in mats_list}
    m = by.get(tid)
    if not m:
        return {"task_id": tid, "ok": False, "error": "missing"}
    work = Path(tempfile.mkdtemp(prefix=f"m28bf_{tid}_", dir="/tmp"))
    saved_proxy = {
        k: os.environ.get(k)
        for k in list(os.environ)
        if k.upper().endswith("_PROXY") or k.upper() == "OXYLABS_PROXY_URL"
    }
    try:
        for k in list(saved_proxy):
            os.environ.pop(k, None)
        base = _materialize_base_worktree(m, work=work, clone_cache=cache)
        host = _prepare_host_suite_env(
            base, language=m.language or "python", work_root=work / "host"
        )
        # force common deps (no socksvia)
        pip_common = build_host_suite_pip_command(host.python, packages=HOST_SUITE_COMMON_DEPS)
        subprocess.run(
            pip_common, capture_output=True, text=True, env=clean, cwd=str(base), check=False
        )
        # install package so green suite imports resolve even when PYTHONPATH alone fails
        subprocess.run(
            [
                host.python,
                "-m",
                "pip",
                "install",
                "--upgrade-strategy",
                "only-if-needed",
                "-q",
                ".",
            ],
            capture_output=True,
            text=True,
            env=clean,
            cwd=str(base),
            check=False,
        )
        chk = subprocess.run(
            [host.python, "-c", "import pytest; print(pytest.__version__)"],
            capture_output=True,
            text=True,
            env=clean,
        )
        if chk.returncode != 0:
            return {
                "task_id": tid,
                "ok": False,
                "error": f"pytest missing after install: {chk.stderr[:200]}",
            }
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
            "sample": f2p[:5],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "task_id": tid,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
    finally:
        for k, v in saved_proxy.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(work, ignore_errors=True)


def f2p_from_pack(pack: Path) -> int:
    cfg = pack / "tests" / "config.json"
    if cfg.exists():
        try:
            j = json.loads(cfg.read_text())
            f2p = j.get("fail_to_pass") or j.get("f2p") or []
            if isinstance(f2p, list):
                return len(f2p)
            if isinstance(f2p, int):
                return f2p
        except Exception:
            pass
    tt = pack / "task.toml"
    if tt.exists():
        t = tt.read_text(errors="replace")
        m = re.search(r"fail_to_pass\s*=\s*\[(.*?)\]", t, re.S)
        if m:
            return len(re.findall(r'"[^"]+"', m.group(1)))
    return 0


def pack_repo_url(pack: Path) -> str:
    tt = pack / "task.toml"
    if tt.exists():
        m = re.search(r'repository_url\s*=\s*"([^"]+)"', tt.read_text(errors="replace"))
        if m:
            return m.group(1)
    return ""


def load_retry_finalize():
    path = ROOT / "scripts" / "m28b_retry_smoke_gen.py"
    spec = importlib.util.spec_from_file_location("m28b_retry_smoke_gen", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge_product(side_root: Path) -> dict:
    cand: list[dict] = []
    for origin, root, score0 in (
        ("product", PRODUCT / "tasks", 2000.0),
        ("side", side_root / "tasks", 500.0),
    ):
        if not root.is_dir():
            continue
        for pack in sorted(root.iterdir()):
            if not pack.is_dir() or not pack.name.startswith("realpr-"):
                continue
            sol = pack / "solution" / "solution.patch"
            if not sol.exists():
                continue
            text = sol.read_text(errors="replace")
            ok, n, h, a = structure_ok_text(text)
            f2p = f2p_from_pack(pack)
            repo_url = pack_repo_url(pack)
            score = score0 + f2p + h * 0.1 + a * 0.01
            if origin == "side":
                ev = side_root / "evidence" / "docker" / f"{pack.name}.sol.reward.json"
                if not ev.exists():
                    log(f"skip side no sol {pack.name}")
                    continue
                try:
                    reward = json.loads(ev.read_text()).get("reward")
                except Exception:
                    reward = None
                if reward != 1:
                    log(f"skip side reward {pack.name}={reward}")
                    continue
                null_p = side_root / "evidence" / "docker" / f"{pack.name}.null.reward.json"
                null_r = 0
                if null_p.exists():
                    try:
                        null_r = int(json.loads(null_p.read_text()).get("reward") or 0)
                    except Exception:
                        null_r = 1
                if null_r != 0:
                    log(f"skip side null {pack.name}")
                    continue
                if not ok or f2p < 5:
                    log(f"skip side floors {pack.name} ok={ok} f2p={f2p}")
                    continue
            cand.append(
                {
                    "task_id": pack.name,
                    "pack_id": pack.name,
                    "path": str(pack),
                    "repo": normalize_upstream_repo(repo_url) or repo_url,
                    "repository_url": repo_url,
                    "score": score,
                    "from": origin,
                    "source_files": n,
                    "source_hunks": h,
                    "gold_added_lines": a,
                    "f2p_nodes": f2p,
                    "struct_ok": ok,
                }
            )

    by_id: dict[str, dict] = {}
    for c in cand:
        prev = by_id.get(c["task_id"])
        if prev is None or c["score"] > prev["score"]:
            by_id[c["task_id"]] = c
    kept, dropped = apply_max_packs_per_repo(
        list(by_id.values()), max_packs=DEFAULT_MAX_PACKS_PER_REPO, score_key="score"
    )
    kept_sorted = sorted(kept, key=lambda x: -float(x.get("score") or 0))[:15]
    log(
        f"diversity kept={[k['task_id'] for k in kept_sorted]} "
        f"dropped={[d.get('task_id') for d in dropped]}"
    )

    staging = WORK / "merge_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for k in kept_sorted:
        shutil.copytree(Path(k["path"]), staging / k["task_id"])

    tasks_dir = PRODUCT / "tasks"
    backup = WORK / "tasks_backup"
    if backup.exists():
        shutil.rmtree(backup)
    if tasks_dir.exists():
        shutil.copytree(tasks_dir, backup)
        shutil.rmtree(tasks_dir)
    shutil.copytree(staging, tasks_dir)

    ev_docker = PRODUCT / "evidence" / "docker"
    ev_docker.mkdir(parents=True, exist_ok=True)
    for k in kept_sorted:
        if k["from"] != "side":
            continue
        tid = k["task_id"]
        src_ev = side_root / "evidence" / "docker"
        for name in (
            f"{tid}.json",
            f"{tid}.sol.reward.json",
            f"{tid}.null.reward.json",
            f"{tid}.oracle_evidence.json",
        ):
            s = src_ev / name
            if s.exists():
                shutil.copy2(s, ev_docker / name)

    mod = load_retry_finalize()
    result = mod.finalize(kept_sorted)
    return result


def main() -> int:
    EV.mkdir(parents=True, exist_ok=True)
    (EV / "fresh_wave.log").write_text(
        f"start {datetime.now(timezone.utc).isoformat()}\n"
    )
    token = load_token()
    if not token:
        log("missing token")
        return 2
    proxy = resolve_github_proxy_url()
    log(f"proxy={redact_proxy_url(proxy) if proxy else None}")

    seeds = sorted(p.name for p in (PRODUCT / "tasks").iterdir() if p.is_dir())
    log(f"product seeds {seeds}")
    if len(seeds) < 5:
        log("WARNING product N<5 before wave; abort to avoid further damage")
        return 9

    if MAT.exists():
        shutil.rmtree(MAT)
    MAT.mkdir(parents=True)

    pairs: list[tuple[str, int]] = []
    hits_path = EV / "hits.json"
    if hits_path.exists():
        try:
            hits = json.loads(hits_path.read_text())
            for h in sorted(
                hits, key=lambda x: (-int(x.get("add") or 0), -int(x.get("files") or 0))
            ):
                repo = h["repo"]
                if "werkzeug" in repo:
                    continue
                pairs.append((repo, int(h["pr"])))
        except Exception as exc:  # noqa: BLE001
            log(f"hits load err {exc}")
    for repo, pr in CAND_PRS:
        if (repo, pr) not in pairs:
            pairs.append((repo, pr))

    by_repo: dict[str, int] = defaultdict(int)
    mat_ok: list[str] = []
    for repo, pr in pairs:
        if by_repo[repo] >= 2:
            continue
        tid_guess = tid_for(repo, pr)
        if tid_guess in PRODUCT_SEEDS:
            continue
        pack = materialize_one(repo, pr, token)
        if not pack or not pack.exists():
            continue
        if not (pack / "solution.patch").exists() or not (pack / "test.patch").exists():
            shutil.rmtree(pack, ignore_errors=True)
            continue
        text = (pack / "solution.patch").read_text(errors="replace")
        ok, n, h, a = structure_ok_text(text)
        log(f"struct {pack.name} ok={ok} f={n} h={h} a={a}")
        if not ok:
            shutil.rmtree(pack, ignore_errors=True)
            continue
        by_repo[repo] += 1
        mat_ok.append(pack.name)
        if len(mat_ok) >= 16:
            break
        time.sleep(0.15)

    log(f"struct-ok mats ({len(mat_ok)}): {mat_ok}")
    for p in list(MAT.iterdir()):
        if p.is_dir() and p.name.startswith("realpr-") and p.name not in mat_ok:
            shutil.rmtree(p)

    if not mat_ok:
        log("no struct-ok materials")
        return 3

    cache = CloneCache(root=Path("datasets/_clone_cache"))
    smoke: list[dict] = []
    for tid in mat_ok:
        log(f"SMOKE {tid}")
        row = dual_smoke(tid, cache)
        smoke.append(row)
        log(f" -> {json.dumps(row)[:320]}")
        (EV / "fresh_smoke.json").write_text(json.dumps(smoke, indent=2))
        time.sleep(0.2)

    survivors = [
        r["task_id"]
        for r in smoke
        if r.get("ok") and int(r.get("f2p") or 0) >= 5
    ]
    log(f"SURVIVORS F2P>=5: {survivors}")
    # also note dual-ok thin F2P for diagnostics
    dual_ok_any = [r for r in smoke if r.get("ok")]
    log(f"dual_ok_any: {[(r['task_id'], r.get('f2p')) for r in dual_ok_any]}")
    (EV / "fresh_survivors.json").write_text(json.dumps(survivors, indent=2))

    for p in list(MAT.iterdir()):
        if p.is_dir() and p.name.startswith("realpr-") and p.name not in survivors:
            shutil.rmtree(p)

    if not survivors:
        log("zero F2P>=5 survivors; leave product at N=5")
        return 4

    # side generate under no-proxy (host dual-run + docker)
    if OUT_SIDE.exists():
        shutil.rmtree(OUT_SIDE)
    OUT_SIDE.mkdir(parents=True)
    WORK.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROOT / ".venv/bin/deepagent"),
        "generate",
        "--out",
        str(OUT_SIDE),
        "--target",
        "12",
        "--min-packs",
        "1",
        "--max-packs",
        "12",
        "--materials",
        str(MAT),
        "--live-mine",
        "--oracle",
        "docker",
        "--panel",
        "offline",
        "--pier",
        "scripted",
        "--work",
        str(WORK / "gen_work"),
        "--json",
    ]
    log("GENERATE " + " ".join(cmd))
    gen_env = noproxy_env()
    gen_env["GITHUB_TOKEN"] = token
    gen_env["GH_TOKEN"] = token
    with (EV / "fresh_generate.log").open("w") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, env=gen_env
        )
    log(f"GENERATE exit={proc.returncode}")
    if (OUT_SIDE / "tasks").is_dir():
        log(
            "side tasks "
            + str(sorted(p.name for p in (OUT_SIDE / "tasks").iterdir() if p.is_dir()))
        )

    result = merge_product(OUT_SIDE)
    (EV / "fresh_merge.json").write_text(json.dumps(result, indent=2, default=str))
    n = int(result.get("n") or 0)
    log(f"MERGE N={n} ids={result.get('ids')}")
    if n < 8:
        log(f"FAIL hard N={n}<8")
        return 10
    if n < 12:
        log(f"PARTIAL N={n}<12 preferred")
        return 0
    log(f"SUCCESS N={n}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise
