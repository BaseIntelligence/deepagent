"""M9 funnel hardening: skip docs, monorepo skip, clone cache, parallel envbuild caps."""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path

import pytest

from swe_factory.envbuild.hygiene import (
    MAX_CONCURRENT_ENVBUILD_JOBS,
    HygieneError,
)
from swe_factory.envbuild.models import EnvRecipe
from swe_factory.envbuild.parallel import (
    DEFAULT_ENVBUILD_WORKERS,
    HARD_MAX_ENVBUILD_WORKERS,
    ParallelEnvJob,
    clamp_envbuild_workers,
    parallel_envbuild,
    parallel_envbuild_recipes,
    peak_envbuild_jobs,
    reset_global_envbuild_semaphore_for_tests,
)
from swe_factory.producers.harbor_labeling import SuiteOutcome, detect_dual_run_flake
from swe_factory.sources.allowlist import SeedRepo
from swe_factory.sources.clone_cache import CloneCache, cache_key_for
from swe_factory.sources.funnel import (
    apply_flake_gate,
    apply_monorepo_skip,
    default_funnel_config_for_scale,
    document_all_skip_reasons,
    gold_signature,
    make_scale_funnel_report,
)
from swe_factory.sources.monorepo import (
    evaluate_monorepo_gate,
    paths_look_monorepo,
    scan_monorepo_signals,
)
from swe_factory.sources.skip_reasons import (
    SKIP_FLAKE_GATE,
    SKIP_MONOREPO,
    SKIP_OFF_LIMITS_DOCKER,
    SKIP_ORACLE_FLAKE,
    SKIP_REASON_DOCS,
    SKIP_REPO_TOO_LARGE,
    SkipReason,
    describe_skip_reason,
    normalize_reason_code,
    tally_skip_reasons,
)

# ---------------------------------------------------------------------------
# Documented skip reasons
# ---------------------------------------------------------------------------


def test_skip_reason_catalog_documents_monorepo_flake_and_parallel() -> None:
    catalog = document_all_skip_reasons()
    codes = {row["code"] for row in catalog}
    assert SKIP_MONOREPO in codes
    assert SKIP_FLAKE_GATE in codes
    assert SKIP_ORACLE_FLAKE in codes
    assert SKIP_OFF_LIMITS_DOCKER in codes
    assert SKIP_REPO_TOO_LARGE in codes
    for row in catalog:
        assert row["documentation"].strip()
    assert describe_skip_reason(SKIP_MONOREPO)
    assert "monorepo" in describe_skip_reason(SKIP_MONOREPO).lower()


def test_normalize_reason_aliases_flake_codes() -> None:
    assert normalize_reason_code("G2_FLAKE") == SKIP_ORACLE_FLAKE
    assert normalize_reason_code("FLAKE_REJECT") == SKIP_ORACLE_FLAKE
    assert normalize_reason_code("monorepo") == SKIP_MONOREPO
    assert normalize_reason_code("unknown_custom_x") == "unknown_custom_x"


def test_funnel_report_includes_catalog_and_bound_parallelism(tmp_path: Path) -> None:
    report = make_scale_funnel_report(default_funnel_config_for_scale(70))
    report.add_skip(
        SkipReason(code=SKIP_MONOREPO, detail="workspace turbo", stage="mine", repo="org/big")
    )
    path = report.write_json(tmp_path / "funnel_report.json")
    assert path.is_file()
    payload = report.to_dict()
    assert payload["parallelism_bounded"] is True
    assert payload["milestone"] == "m9-scale-70"
    assert "mission-test-pg" in payload["off_limits_docker_policy"]
    assert payload["counters"]["monorepo_skipped"] == 1
    assert SKIP_MONOREPO in payload["skip_reasons_tallied"]
    assert len(payload["documented_skip_catalog"]) == len(SKIP_REASON_DOCS)


# ---------------------------------------------------------------------------
# Monorepo skip
# ---------------------------------------------------------------------------


def test_scan_and_skip_pnpm_workspace(tmp_path: Path) -> None:
    root = tmp_path / "mega"
    root.mkdir()
    (root / "pnpm-workspace.yaml").write_text("packages:\n  - 'packages/*'\n", encoding="utf-8")
    (root / "package.json").write_text(
        '{"name":"root","workspaces":["packages/*"]}', encoding="utf-8"
    )
    for i in range(3):
        pkg = root / "packages" / f"p{i}"
        pkg.mkdir(parents=True)
        (pkg / "package.json").write_text(f'{{"name":"p{i}"}}', encoding="utf-8")
    signal = scan_monorepo_signals(root)
    assert signal.has_explicit_workspace
    assert "pnpm-workspace.yaml" in signal.markers_found
    decision = evaluate_monorepo_gate(root)
    assert decision.skip is True
    assert decision.reason_code == SKIP_MONOREPO
    should, reason, _ = apply_monorepo_skip(root=root, repo="org/mega")
    assert should is True
    assert reason is not None
    assert reason.code == SKIP_MONOREPO
    assert reason.stage == "mine"


def test_paths_look_monorepo_on_lerna_and_many_packages() -> None:
    decision = paths_look_monorepo(
        [
            "lerna.json",
            "packages/a/package.json",
            "packages/b/package.json",
            "packages/c/package.json",
        ]
    )
    assert decision.skip is True
    assert decision.reason_code == SKIP_MONOREPO

    small = paths_look_monorepo(["src/a.py", "src/b.py", "tests/test_a.py"])
    assert small.skip is False


def test_repo_too_large_gate() -> None:
    sig_ok = type(scan_monorepo_signals)  # just for type comfort  # noqa: F841
    from swe_factory.sources.monorepo import MonorepoSignal

    big = MonorepoSignal(tracked_file_estimate=100_000, tree_bytes_estimate=10)
    decision = evaluate_monorepo_gate(signal=big, max_tracked_files=25_000)
    assert decision.skip is True
    assert decision.reason_code == SKIP_REPO_TOO_LARGE


# ---------------------------------------------------------------------------
# Clone cache
# ---------------------------------------------------------------------------


def test_clone_cache_reuse_local_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture_repo"
    fixture.mkdir()
    (fixture / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    seed = SeedRepo(
        seed_id="cache_demo",
        language="python",
        repo="owner/cache-demo",
        base_commit="a" * 40,
        license="MIT",
        local_fixture=None,
    )
    # Point resolve_local_path via monkeypatch-style substitute: use prefer_local
    # with a hand-built local by writing resolve through SeedRepo local_fixture
    # Using CloneCache prefer_local with a real resolve path:
    cache = CloneCache(root=tmp_path / "cache", depth=10)

    # Manually plant a fake "local" by calling ensure_seed with prefer after
    # patching resolve — simpler: inject by materializing once then second hit.
    # Direct tree copy path: create seed with local_fixture relative is hard;
    # use two-step: write index via ensure that fails network-less → prefer simulate.
    # Test only the key + hit counter when directory pre-seeded:
    key = cache_key_for(seed)
    dest = cache.entry_path(key)
    dest.mkdir(parents=True)
    (dest / ".git").mkdir()
    (dest / "README").write_text("x", encoding="utf-8")
    entry1 = cache.ensure_seed(seed, refresh=False)
    assert entry1.source == "reused"
    assert cache.stats.hits >= 1
    entry2 = cache.ensure_seed(seed, refresh=False)
    assert entry2.hits >= 2
    assert cache.stats.hits >= 2
    stats = cache.stats_dict()
    assert stats["entry_count"] >= 1
    assert Path(stats["root"]) == tmp_path / "cache"


def test_gold_signature_stable() -> None:
    a = gold_signature(
        repo="o/r", base_commit="a" * 40, gold_patch="diff a\n", source_files=["a.py", "b.py"]
    )
    b = gold_signature(
        repo="o/r", base_commit="a" * 40, gold_patch="diff a\n", source_files=["b.py", "a.py"]
    )
    c = gold_signature(
        repo="o/r", base_commit="a" * 40, gold_patch="diff b\n", source_files=["a.py", "b.py"]
    )
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# Flake gate
# ---------------------------------------------------------------------------


def test_apply_flake_gate_from_dual_run_detector() -> None:
    run_a = SuiteOutcome(language="python", passed=("t1",), failed=("t2",), returncode=1)
    run_b = SuiteOutcome(language="python", passed=("t1", "t2"), failed=(), returncode=0)
    is_flake, codes, details = detect_dual_run_flake([run_a, run_b], phase="gold")
    assert is_flake is True
    skip, reason = apply_flake_gate(
        is_flake=is_flake,
        reason_codes=codes,
        details=details,
        repo="o/r",
        candidate_id="c1",
    )
    assert skip is True
    assert reason is not None
    assert reason.code == SKIP_ORACLE_FLAKE
    # non-flake dual agree
    ok_a = SuiteOutcome(language="python", passed=("t1",), failed=("t2",))
    ok_b = SuiteOutcome(language="python", passed=("t1",), failed=("t2",))
    is_ok, _, _ = detect_dual_run_flake([ok_a, ok_b])
    skip2, reason2 = apply_flake_gate(is_flake=is_ok, reason_codes=[])
    assert skip2 is False
    assert reason2 is None


# ---------------------------------------------------------------------------
# Parallel envbuild caps + no off-limits damage
# ---------------------------------------------------------------------------


def test_clamp_envbuild_workers_bounds() -> None:
    assert clamp_envbuild_workers(None) == DEFAULT_ENVBUILD_WORKERS
    assert clamp_envbuild_workers(0) == 1
    assert clamp_envbuild_workers(-5) == 1
    assert clamp_envbuild_workers(999) == HARD_MAX_ENVBUILD_WORKERS
    assert clamp_envbuild_workers(16) == 16
    assert 16 <= MAX_CONCURRENT_ENVBUILD_JOBS <= HARD_MAX_ENVBUILD_WORKERS


def test_parallel_envbuild_peak_never_exceeds_cap() -> None:
    reset_global_envbuild_semaphore_for_tests()
    max_workers = 3
    gate = threading.Barrier(max_workers)
    active = {"n": 0, "peak": 0}
    lock = threading.Lock()

    def build_fn(recipe: EnvRecipe) -> dict[str, str]:
        with lock:
            active["n"] += 1
            active["peak"] = max(active["peak"], active["n"])
        try:
            # Rendezvous so all workers hold slots together
            with contextlib.suppress(threading.BrokenBarrierError):
                gate.wait(timeout=5)
            time.sleep(0.05)
            return {"ok": recipe.repo_id}
        finally:
            with lock:
                active["n"] -= 1

    recipes = [
        EnvRecipe(repo_id=f"org/r{i}", base_commit="a" * 40, language="python") for i in range(8)
    ]
    report = parallel_envbuild_recipes(
        recipes,
        build_fn,
        max_workers=max_workers,
        acquire_timeout_s=10.0,
    )
    assert report.max_workers == max_workers
    assert report.job_count == 8
    assert report.ok_count == 8
    assert active["peak"] <= max_workers
    assert peak_envbuild_jobs() <= max_workers
    assert report.to_dict()["parallelism_bounded"] is True
    reset_global_envbuild_semaphore_for_tests()


def test_parallel_envbuild_refuses_off_limits_names() -> None:
    reset_global_envbuild_semaphore_for_tests()
    called: list[str] = []

    def build_fn(recipe: EnvRecipe) -> dict[str, str]:
        called.append(recipe.repo_id)
        return {"ok": "yes"}

    jobs = [
        ParallelEnvJob(
            job_id="safe",
            recipe=EnvRecipe(repo_id="org/safe", base_commit="a" * 40),
            container_name="sdf-envbuild-safe",
        ),
        ParallelEnvJob(
            job_id="bad-pg",
            recipe=EnvRecipe(repo_id="org/bad", base_commit="a" * 40),
            container_name="mission-test-pg",
        ),
        ParallelEnvJob(
            job_id="bad-prism",
            recipe=EnvRecipe(repo_id="org/prism", base_commit="a" * 40),
            container_name="challenge-prism-side",
        ),
        ParallelEnvJob(
            job_id="bad-proxy",
            recipe=EnvRecipe(repo_id="org/proxy", base_commit="a" * 40),
            container_name="acproxy",
        ),
    ]
    report = parallel_envbuild(jobs, build_fn, max_workers=4)
    assert "org/safe" in called
    assert "org/bad" not in called
    assert "org/prism" not in called
    assert "org/proxy" not in called
    assert report.ok_count == 1
    assert report.skip_count >= 3
    codes = {r.skip_reason.code for r in report.results if r.skip_reason is not None}
    assert SKIP_OFF_LIMITS_DOCKER in codes
    reset_global_envbuild_semaphore_for_tests()


def test_assert_job_docker_safe_raises() -> None:
    from swe_factory.envbuild.parallel import assert_job_docker_safe

    with pytest.raises(HygieneError):
        assert_job_docker_safe("mission-test-pg")
    with pytest.raises(HygieneError):
        assert_job_docker_safe("challenge-prism-1")
    with pytest.raises(HygieneError):
        assert_job_docker_safe("acproxy")
    # owned is fine
    assert_job_docker_safe("sdf-envbuild-ok")
    assert_job_docker_safe("deepagent-env-ok")


def test_tally_skips_from_events() -> None:
    events = [
        SkipReason(code=SKIP_MONOREPO, detail="a"),
        SkipReason(code="monorepo", detail="b"),
        SkipReason(code=SKIP_FLAKE_GATE, detail="c"),
    ]
    tallies = tally_skip_reasons(events)
    assert tallies[SKIP_MONOREPO] == 2
    assert tallies[SKIP_FLAKE_GATE] == 1


def test_default_funnel_config_scale_70() -> None:
    cfg = default_funnel_config_for_scale(70)
    assert cfg.monorepo_enabled is True
    assert cfg.clone_cache_enabled is True
    assert cfg.flake_gate_enabled is True
    assert 1 <= cfg.clamped_workers() <= HARD_MAX_ENVBUILD_WORKERS
