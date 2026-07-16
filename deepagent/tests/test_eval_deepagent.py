"""Offline unit tests for DeepAgent-grade eval-deepagent CLI (VAL-DEVAL-001..007).

Mocks pier reward paths; never uses never-solve panel as DeepAgent fidelity.
"""

from __future__ import annotations

import json
import threading
import time
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_factory.accounting import BudgetLedger
from swe_factory.cli import app
from swe_factory.panel.eval_deepagent import (
    DEEPAGENT_EVAL_FIDELITY,
    DEEPAGENT_EVAL_MODELS,
    DEFAULT_HARD_STOP_USD,
    DeepAgentEvalError,
    PackPreflight,
    _default_live_miniswe_invoke,
    harvest_miniswe_cost_usd,
    load_product_packs,
    mocked_miniswe_invoker,
    normalize_model_id,
    openrouter_model_flag,
    preflight_pack_oracle_nop,
    resolve_eval_models,
    run_deepagent_eval,
    trajectory_backed_miniswe_invoker,
)
from swe_factory.panel.runner import REQUIRED_PANEL_MODELS

GROK = "x-ai/grok-4.5"
KIMI = "moonshotai/kimi-k2.6"
runner = CliRunner()


def _write_min_pack(root: Path, pack_id: str) -> Path:
    pack = root / "tasks" / pack_id
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "instruction.md").write_text(
        f"Fix bug in {pack_id}\n",
        encoding="utf-8",
    )
    (pack / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.1"',
                "[task]",
                f'name = "swe-factory/{pack_id}"',
                "[metadata]",
                f'task_id = "{pack_id}"',
                'language = "python"',
                'source_track = "real_pr"',
                'repository_url = "https://github.com/example/demo.git"',
                'base_commit_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"',
                'license = "MIT"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return pack


def test_help_lists_eval_deepagent() -> None:
    """VAL-DEVAL-007: CLI help surfaces eval-deepagent."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "eval-deepagent" in result.output


def test_eval_deepagent_help_exit_zero() -> None:
    """VAL-DEVAL-007: eval-deepagent --help works."""
    result = runner.invoke(app, ["eval-deepagent", "--help"])
    assert result.exit_code == 0, result.output
    text = result.output.lower()
    assert "hard-stop" in text or "hard_stop" in text or "300" in text
    assert "n-concurrent" in text or "concurrent" in text
    assert "pier" in text or "mini-swe" in text or "deepagent" in text


def test_models_exact_grok_kimi_pair() -> None:
    """VAL-DEVAL-003: exact OpenRouter model pair."""
    assert list(DEEPAGENT_EVAL_MODELS) == [GROK, KIMI]
    assert resolve_eval_models() == REQUIRED_PANEL_MODELS
    assert normalize_model_id("openrouter/x-ai/grok-4.5") == GROK
    assert openrouter_model_flag(GROK) == f"openrouter/{GROK}"
    with pytest.raises(DeepAgentEvalError):
        resolve_eval_models(["anthropic/claude-opus-4.8"])


def test_offline_mocked_pier_reward_paths(tmp_path: Path) -> None:
    """VAL-DEVAL-001/002/006: offline unit mocks pier reward.json harvest."""
    product = tmp_path / "deepagent_v1"
    for pid in ("realpr-itemadapter-101", "realpr-click-3645"):
        _write_min_pack(product, pid)

    matrix = {
        "realpr-itemadapter-101": {GROK: [True], KIMI: [False]},
        "realpr-click-3645": {GROK: [False], KIMI: [False]},
    }
    invoker = mocked_miniswe_invoker(matrix, job_root=tmp_path / "mock-jobs")
    out = tmp_path / "panel_deepagent_ut"
    jobs = tmp_path / "harbor-deepagent-jobs-ut"
    report = run_deepagent_eval(
        product_root=product,
        out_dir=out,
        max_packs=2,
        k=1,
        n_concurrent=1,
        hard_stop_usd=Decimal("300"),
        reserve_usd=Decimal("1.00"),
        jobs_dir=jobs,
        preflight=True,
        invoker=invoker,
        offline=True,
        reclaim=False,
    )

    assert report.fidelity == DEEPAGENT_EVAL_FIDELITY
    assert report.fidelity == "pier_miniswe_harbor"
    assert report.n_concurrent == 1
    assert list(report.models) == [GROK, KIMI]
    assert report.hard_stop_usd == Decimal("300")
    assert report.invented_rewards is False
    assert report.n_packs_scored == 2
    assert report.offline is True

    # Rewards harvested from written reward.json paths (not invented).
    first = report.pack_results[0]
    assert first.complete is True
    assert first.models[0].model == GROK
    assert first.models[0].pass_at_k == 1.0
    assert first.models[0].trials[0].reward == 1
    assert first.models[0].trials[0].reward_path is not None
    reward_path = Path(first.models[0].trials[0].reward_path)
    assert reward_path.is_file()
    raw = json.loads(reward_path.read_text(encoding="utf-8"))
    assert raw["reward"] == 1
    assert first.models[1].pass_at_k == 0.0

    second = report.pack_results[1]
    assert second.decision is not None
    assert second.decision.rule == "solve-none"

    report_path = out / "report.json"
    assert report_path.is_file()
    blob = json.loads(report_path.read_text(encoding="utf-8"))
    assert blob["fidelity"] == "pier_miniswe_harbor"
    assert blob["n_concurrent"] == 1
    assert blob["models"] == [GROK, KIMI]
    assert "spend_usd" in blob or "total_spend_usd" in blob
    assert blob.get("never_solve_panel") is False
    assert float(blob["hard_stop_usd"]) == 300.0


def test_n_concurrent_accepts_one_and_five(tmp_path: Path) -> None:
    """VAL-DBENCH-001: n_concurrent in 1..5 is accepted; report records value."""
    product = tmp_path / "deepagent_v1"
    _write_min_pack(product, "realpr-itemadapter-101")
    matrix = {"realpr-itemadapter-101": {GROK: [True], KIMI: [False]}}
    for n in (1, 5):
        report = run_deepagent_eval(
            product_root=product,
            out_dir=tmp_path / f"out_n{n}",
            max_packs=1,
            k=1,
            n_concurrent=n,
            hard_stop_usd=Decimal("300"),
            reserve_usd=Decimal("1.00"),
            jobs_dir=tmp_path / f"jobs_n{n}",
            invoker=mocked_miniswe_invoker(matrix),
            offline=True,
            reclaim=False,
        )
        assert report.n_concurrent == n
        blob = json.loads((tmp_path / f"out_n{n}" / "report.json").read_text(encoding="utf-8"))
        assert blob["n_concurrent"] == n


def test_n_concurrent_refuses_zero_and_above_five(tmp_path: Path) -> None:
    """VAL-DBENCH-001: n_concurrent <1 or >5 refuse fail-closed."""
    product = tmp_path / "deepagent_v1"
    _write_min_pack(product, "realpr-itemadapter-101")
    for bad in (0, 6):
        with pytest.raises(DeepAgentEvalError, match="n_concurrent"):
            run_deepagent_eval(
                product_root=product,
                out_dir=tmp_path / f"out_bad_{bad}",
                max_packs=1,
                n_concurrent=bad,
                invoker=mocked_miniswe_invoker({}),
                offline=True,
                reclaim=False,
            )


def test_budget_hard_stop(tmp_path: Path) -> None:
    """VAL-DEVAL-004: hard stop respects remaining reservation budget."""
    product = tmp_path / "deepagent_v1"
    for pid in ("p1", "p2", "p3"):
        _write_min_pack(product, pid)
    matrix = {pid: {GROK: [False], KIMI: [False]} for pid in ("p1", "p2", "p3")}
    # 2 models * $1 reserve each; hard stop $2 → first pack only, then stop.
    report = run_deepagent_eval(
        product_root=product,
        out_dir=tmp_path / "out",
        max_packs=3,
        k=1,
        n_concurrent=1,
        hard_stop_usd=Decimal("2.00"),
        reserve_usd=Decimal("1.00"),
        jobs_dir=tmp_path / "jobs",
        invoker=mocked_miniswe_invoker(matrix),
        offline=True,
        reclaim=False,
    )
    assert report.budget_stop is True
    assert report.n_packs_scored == 1
    assert report.total_spend_usd <= Decimal("2.00")
    assert any(p.budget_stop for p in report.pack_results)


def test_load_product_packs_preferred_order(tmp_path: Path) -> None:
    product = tmp_path / "deepagent_v1"
    for pid in (
        "realpr-packaging-1120",
        "realpr-itemadapter-101",
        "realpr-zzz-999",
        "realpr-click-3645",
    ):
        _write_min_pack(product, pid)
    keeps = load_product_packs(product, max_packs=3)
    ids = [k["task_id"] for k in keeps]
    assert ids[0] == "realpr-itemadapter-101"
    assert ids[1] == "realpr-click-3645"
    assert ids[2] == "realpr-packaging-1120"


def test_offline_cli_writes_report(tmp_path: Path) -> None:
    """CLI offline path produces report schema with spend/fidelity/n_concurrent."""
    product = tmp_path / "deepagent_v1"
    _write_min_pack(product, "realpr-itemadapter-101")
    out = tmp_path / "panel_deepagent_cli"
    jobs = tmp_path / "jobs-cli"
    result = runner.invoke(
        app,
        [
            "eval-deepagent",
            "--product-root",
            str(product),
            "--out",
            str(out),
            "--max-packs",
            "1",
            "--k",
            "1",
            "--n-concurrent",
            "1",
            "--hard-stop-usd",
            "300",
            "--reserve-usd",
            "1",
            "--jobs-dir",
            str(jobs),
            "--offline",
            "--no-reclaim",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["fidelity"] == "pier_miniswe_harbor"
    assert payload["n_concurrent"] == 1
    assert payload["models"] == [GROK, KIMI]
    assert float(payload["hard_stop_usd"]) == 300.0
    assert payload["never_solve_panel"] is False
    report_path = Path(payload["report"])
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "spend_usd" in report
    assert report["k"] == 1
    assert report["agent"] == "mini-swe-agent"


def test_cli_rejects_n_concurrent_outside_range(tmp_path: Path) -> None:
    """VAL-DBENCH-001: swe-factory eval-deepagent refuses n_concurrent 0 and 6."""
    product = tmp_path / "deepagent_v1"
    _write_min_pack(product, "realpr-itemadapter-101")
    for bad in (0, 6):
        result = runner.invoke(
            app,
            [
                "eval-deepagent",
                "--product-root",
                str(product),
                "--out",
                str(tmp_path / f"out_bad_{bad}"),
                "--n-concurrent",
                str(bad),
                "--offline",
            ],
        )
        assert result.exit_code == 2, (bad, result.output)
        combined = result.output.lower()
        assert "n_concurrent" in combined or "1..5" in combined or "refuse" in combined


def test_cli_accepts_n_concurrent_five_offline(tmp_path: Path) -> None:
    """VAL-DBENCH-001: n_concurrent=5 offline path accepted and recorded."""
    product = tmp_path / "deepagent_v1"
    _write_min_pack(product, "realpr-itemadapter-101")
    out = tmp_path / "panel_n5"
    jobs = tmp_path / "jobs_n5"
    result = runner.invoke(
        app,
        [
            "eval-deepagent",
            "--product-root",
            str(product),
            "--out",
            str(out),
            "--max-packs",
            "1",
            "--k",
            "1",
            "--n-concurrent",
            "5",
            "--hard-stop-usd",
            "300",
            "--reserve-usd",
            "1",
            "--jobs-dir",
            str(jobs),
            "--offline",
            "--no-reclaim",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["n_concurrent"] == 5


def test_preflight_offline_stub(tmp_path: Path) -> None:
    pack = _write_min_pack(tmp_path / "deepagent_v1", "realpr-itemadapter-101")
    stub = PackPreflight(
        pack_id="realpr-itemadapter-101",
        pack_path=str(pack),
        ok=True,
        solution_reward=1,
        null_reward=0,
        mode="offline-stub",
    )
    result = preflight_pack_oracle_nop(pack, offline_stub=stub)
    assert result.ok is True
    assert result.solution_reward == 1
    assert result.null_reward == 0


def test_default_hard_stop_is_300() -> None:
    assert Decimal("300") == DEFAULT_HARD_STOP_USD


def test_harvest_miniswe_cost_usd_from_trajectory(tmp_path: Path) -> None:
    """Trajectory final_metrics.total_cost_usd is harvested; missing → 0."""
    job = tmp_path / "eval-job"
    agent = job / "trial" / "agent"
    agent.mkdir(parents=True)
    traj = agent / "trajectory.json"
    traj.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "final_metrics": {
                    "total_prompt_tokens": 100,
                    "total_completion_tokens": 10,
                    "total_cost_usd": 1.25,
                    "total_steps": 3,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert harvest_miniswe_cost_usd(job) == Decimal("1.25")
    assert harvest_miniswe_cost_usd(tmp_path / "missing") == Decimal("0")
    empty = tmp_path / "empty-job"
    empty.mkdir()
    assert harvest_miniswe_cost_usd(empty) == Decimal("0")


def test_live_invoker_returns_non_zero_cost_from_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VAL-DEVAL-004: live invoker cost_usd is populated from trajectory when present.

    Does not invent costs: only returns what harvest_miniswe_cost_usd reads from
    trajectory final_metrics. Unit with fake trajectory; phones home optional.
    """
    pier_bin = tmp_path / "fake-pier"
    pier_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    pier_bin.chmod(0o755)

    pack = _write_min_pack(tmp_path / "product", "realpr-itemadapter-101")
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    traj_cost = Decimal("1.75")

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        del kwargs
        # Mimic pier writing job dir named from --job-name.
        job_name = None
        for i, tok in enumerate(cmd):
            if tok == "--job-name" and i + 1 < len(cmd):
                job_name = cmd[i + 1]
                break
        assert job_name is not None
        job_dir = jobs_dir / job_name
        agent = job_dir / "trial" / "agent"
        verifier = job_dir / "trial" / "verifier"
        agent.mkdir(parents=True)
        verifier.mkdir(parents=True)
        (agent / "trajectory.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "final_metrics": {
                        "total_prompt_tokens": 50,
                        "total_completion_tokens": 5,
                        "total_cost_usd": float(traj_cost),
                        "total_steps": 2,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (verifier / "reward.json").write_text(json.dumps({"reward": 0}) + "\n", encoding="utf-8")
        return _Proc()

    monkeypatch.setattr(
        "swe_factory.panel.eval_deepagent.subprocess.run",
        _fake_run,
    )
    # Live pier path mints pack-scoped agent image before shelling pier; stub here.
    monkeypatch.setattr(
        "swe_factory.panel.eval_deepagent.ensure_eval_agent_mintag",
        lambda *a, **k: "deepagent-agent-deadbeef:local",
    )

    result = _default_live_miniswe_invoke(
        pack_path=pack,
        pack_id="realpr-itemadapter-101",
        model=GROK,
        jobs_dir=jobs_dir,
        index=0,
        timeout_s=30.0,
        pier_bin=pier_bin,
    )
    assert result["cost_usd"] == traj_cost
    assert result["cost_usd"] > 0
    assert result["reward"] == 0
    assert result["invented_reward"] is False
    assert result["job_dir"] is not None
    assert harvest_miniswe_cost_usd(result["job_dir"]) == traj_cost


def test_hard_stop_with_trajectory_costs_mid_wave(tmp_path: Path) -> None:
    """VAL-DEVAL-004: hard_stop stops mid-wave on accumulated trajectory costs.

    Uses trajectory_backed invoker (fake total_cost_usd) rather than offline $0.01
    mock, so unit does not rely only on the cheap offline stub for live-path truth.
    """
    product = tmp_path / "deepagent_v1"
    for pid in ("p1", "p2", "p3"):
        _write_min_pack(product, pid)
    matrix = {pid: {GROK: [False], KIMI: [False]} for pid in ("p1", "p2", "p3")}
    # Each trial settles $1.50 from trajectory; reserve $2 so settle fits.
    # hard_stop $4 → first pack burns 2 pages × $1.50 = $3 settled, still
    # remaining $1 for next pack need ($4) → budget_stop before p2.
    invoker = trajectory_backed_miniswe_invoker(
        matrix,
        cost_usd_per_trial=Decimal("1.50"),
        job_root=tmp_path / "traj-jobs",
    )
    report = run_deepagent_eval(
        product_root=product,
        out_dir=tmp_path / "out",
        max_packs=3,
        k=1,
        n_concurrent=1,
        hard_stop_usd=Decimal("4.00"),
        reserve_usd=Decimal("2.00"),
        jobs_dir=tmp_path / "jobs",
        invoker=invoker,
        offline=True,
        reclaim=False,
    )
    assert report.budget_stop is True
    assert report.n_packs_scored == 1
    # Settled from trajectory harvest, not 0 and not offline $0.01 alone.
    assert report.total_spend_usd == Decimal("3.00")
    assert report.total_spend_usd <= Decimal("4.00")
    for pack in report.pack_results:
        for model in pack.models:
            for trial in model.trials:
                assert trial.cost_usd == Decimal("1.50")
                assert trial.job_dir is not None
                # Prove harvest path, not invented inline: trajectory file present.
                harvested = harvest_miniswe_cost_usd(trial.job_dir)
                assert harvested == Decimal("1.50")


def test_trajectory_backed_invoker_zero_when_metrics_missing(tmp_path: Path) -> None:
    """Missing total_cost_usd must not invent spend (harvest → 0)."""
    product = tmp_path / "deepagent_v1"
    _write_min_pack(product, "p1")
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    # Direct harvest zero on empty job.
    empty = jobs / "empty"
    empty.mkdir()
    assert harvest_miniswe_cost_usd(empty) == Decimal("0")
    # Trajectory invoker still returns harvested value (== fixed when written).
    inv = trajectory_backed_miniswe_invoker(
        {"p1": {GROK: [True], KIMI: [False]}},
        cost_usd_per_trial=Decimal("0"),
        job_root=tmp_path / "traj0",
    )
    out = inv(
        pack_path=product / "tasks" / "p1",
        pack_id="p1",
        model=GROK,
        jobs_dir=jobs,
        index=0,
        timeout_s=1.0,
    )
    assert out["cost_usd"] == Decimal("0")
    assert out["reward"] == 1


def test_ensure_eval_agent_mintag_stage_only_paints_from(tmp_path: Path) -> None:
    """Eval mintag helper paints pack-scoped FROM without inventing global tag reuse."""
    from swe_factory.panel.eval_deepagent import ensure_eval_agent_mintag

    pack = tmp_path / "tasks" / "realpr-itemadapter-101"
    pack.mkdir(parents=True)
    env = pack / "environment"
    tests = pack / "tests"
    env.mkdir()
    tests.mkdir()
    (env / "Dockerfile").write_text(
        "FROM python:3.12-slim\nWORKDIR /app\n",
        encoding="utf-8",
    )
    (tests / "Dockerfile").write_text(
        "FROM deepagent-agent:local\nCOPY test.sh /tests/test.sh\n",
        encoding="utf-8",
    )
    (pack / "task.toml").write_text('schema_version = "1.1"\n', encoding="utf-8")
    (pack / "instruction.md").write_text("fix\n", encoding="utf-8")
    tag = ensure_eval_agent_mintag(
        pack,
        pack_id="realpr-itemadapter-101",
        jobs_dir=tmp_path / "jobs",
        stage_only=True,
    )
    assert tag.startswith("deepagent-agent-")
    assert tag.endswith(":local")
    assert tag != "deepagent-agent:local"
    painted = (tests / "Dockerfile").read_text(encoding="utf-8")
    assert painted.splitlines()[0] == f"FROM {tag}"


def test_live_invoker_fails_closed_on_mintag_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live mini-swe invoker must not invent reward when agent mintag ensure fails."""
    pier_bin = tmp_path / "fake-pier"
    pier_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    pier_bin.chmod(0o755)
    pack = _write_min_pack(tmp_path / "product", "realpr-itemadapter-101")
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    def _boom(*_a: object, **_k: object) -> str:
        raise RuntimeError("simulated mintag build failure")

    monkeypatch.setattr(
        "swe_factory.panel.eval_deepagent.ensure_eval_agent_mintag",
        _boom,
    )

    result = _default_live_miniswe_invoke(
        pack_path=pack,
        pack_id="realpr-itemadapter-101",
        model=GROK,
        jobs_dir=jobs_dir,
        index=0,
        timeout_s=30.0,
        pier_bin=pier_bin,
    )
    assert result["ok"] is False
    assert result["reward"] is None
    assert result["solved"] is False
    assert result["invented_reward"] is False
    assert any("mintag" in e.lower() for e in result["errors"])


def _sleeping_invoker(
    *,
    delay_s: float = 0.15,
    counter: dict[str, int] | None = None,
) -> object:
    """Mock pier invoker that sleeps to prove real thread-pool concurrency.

    Tracks peak concurrent invocations in ``counter['max_inflight']``.
    """
    lock = threading.Lock()
    state = counter if counter is not None else {"current": 0, "max_inflight": 0}
    state.setdefault("current", 0)
    state.setdefault("max_inflight", 0)
    state.setdefault("calls", 0)

    def _invoke(
        *,
        pack_path: Path,
        pack_id: str,
        model: str,
        jobs_dir: Path,
        index: int,
        timeout_s: float,
    ) -> dict[str, object]:
        del pack_path, timeout_s
        with lock:
            state["current"] += 1
            state["calls"] += 1
            if state["current"] > state["max_inflight"]:
                state["max_inflight"] = state["current"]
        try:
            time.sleep(delay_s)
            mid = normalize_model_id(model)
            jobs_dir.mkdir(parents=True, exist_ok=True)
            job_dir = jobs_dir / f"sleep-{pack_id}-{mid.replace('/', '_')}-k{index}"
            verifier = job_dir / "trial" / "verifier"
            verifier.mkdir(parents=True, exist_ok=True)
            reward_path = verifier / "reward.json"
            reward_path.write_text(json.dumps({"reward": 0}) + "\n", encoding="utf-8")
            return {
                "reward": 0,
                "solved": False,
                "job_dir": str(job_dir),
                "reward_path": str(reward_path),
                "exit_code": 0,
                "errors": (),
                "cost_usd": Decimal("0.01"),
                "ok": True,
                "invented_reward": False,
            }
        finally:
            with lock:
                state["current"] = max(0, state["current"] - 1)

    return _invoke


def test_pool_max_inflight_ge_2_when_n_concurrent_2(tmp_path: Path) -> None:
    """VAL-DPOOL-001: N=2 launches concurrent workers; max_in_flight >= 2."""
    product = tmp_path / "deepagent_v1"
    # 2 packs × 2 models = 4 trials; with N=2 must overlap in-flight.
    for pid in ("p1", "p2"):
        _write_min_pack(product, pid)
    counter: dict[str, int] = {"current": 0, "max_inflight": 0, "calls": 0}
    invoker = _sleeping_invoker(delay_s=0.2, counter=counter)
    report = run_deepagent_eval(
        product_root=product,
        out_dir=tmp_path / "out_pool2",
        max_packs=2,
        k=1,
        n_concurrent=2,
        hard_stop_usd=Decimal("100"),
        reserve_usd=Decimal("1.00"),
        jobs_dir=tmp_path / "jobs_pool2",
        invoker=invoker,  # type: ignore[arg-type]
        offline=True,
        reclaim=False,
    )
    assert counter["calls"] == 4  # 2 packs × 2 models
    assert counter["max_inflight"] >= 2, counter
    assert report.actual_max_inflight is not None
    assert report.actual_max_inflight >= 2
    assert report.n_concurrent == 2
    assert report.pool_workers == 2
    blob = json.loads((tmp_path / "out_pool2" / "report.json").read_text(encoding="utf-8"))
    assert blob["actual_max_inflight"] >= 2
    assert blob["n_concurrent"] == 2
    assert blob["concurrent_pool"] is True
    assert report.invented_rewards is False
    assert report.n_packs_scored == 2


def test_pool_max_inflight_le_n_for_n_equals_3(tmp_path: Path) -> None:
    """VAL-DPOOL-002: N=3 with many trials keeps max in-flight ≤ 3."""
    product = tmp_path / "deepagent_v1"
    # 5 packs × 2 models = 10 trials; pool size 3.
    pack_ids = [f"p{i}" for i in range(5)]
    for pid in pack_ids:
        _write_min_pack(product, pid)
    counter: dict[str, int] = {"current": 0, "max_inflight": 0, "calls": 0}
    invoker = _sleeping_invoker(delay_s=0.12, counter=counter)
    report = run_deepagent_eval(
        product_root=product,
        out_dir=tmp_path / "out_pool3",
        max_packs=5,
        k=1,
        n_concurrent=3,
        hard_stop_usd=Decimal("100"),
        reserve_usd=Decimal("1.00"),
        jobs_dir=tmp_path / "jobs_pool3",
        invoker=invoker,  # type: ignore[arg-type]
        offline=True,
        reclaim=False,
    )
    assert counter["calls"] == 10
    assert counter["max_inflight"] <= 3, counter
    assert counter["max_inflight"] >= 2  # with 10 trials must reach higher than 1
    assert report.actual_max_inflight is not None
    assert report.actual_max_inflight <= 3
    assert report.actual_max_inflight >= 2
    assert report.n_concurrent == 3
    assert report.n_packs_scored == 5


def test_pool_n1_still_serial_equivalent(tmp_path: Path) -> None:
    """VAL-DPOOL default N=1: max_inflight ≤ 1 and full matrix scores."""
    product = tmp_path / "deepagent_v1"
    for pid in ("p1", "p2"):
        _write_min_pack(product, pid)
    counter: dict[str, int] = {"current": 0, "max_inflight": 0, "calls": 0}
    invoker = _sleeping_invoker(delay_s=0.05, counter=counter)
    report = run_deepagent_eval(
        product_root=product,
        out_dir=tmp_path / "out_pool1",
        max_packs=2,
        k=1,
        n_concurrent=1,
        hard_stop_usd=Decimal("100"),
        reserve_usd=Decimal("1.00"),
        jobs_dir=tmp_path / "jobs_pool1",
        invoker=invoker,  # type: ignore[arg-type]
        offline=True,
        reclaim=False,
    )
    assert counter["max_inflight"] <= 1, counter
    assert report.actual_max_inflight is not None
    assert report.actual_max_inflight <= 1
    assert report.n_concurrent == 1
    assert report.n_packs_scored == 2
    # Behavior parity: rewards/spend still present.
    assert report.fidelity == DEEPAGENT_EVAL_FIDELITY


def test_pool_ledger_race_safe_and_hard_stop(tmp_path: Path) -> None:
    """VAL-DPOOL-003: concurrent reserve/settle race-safe; hard-stop prevents overspend."""
    product = tmp_path / "deepagent_v1"
    for pid in ("p1", "p2", "p3", "p4", "p5"):
        _write_min_pack(product, pid)
    counter: dict[str, int] = {"current": 0, "max_inflight": 0, "calls": 0}
    invoker = _sleeping_invoker(delay_s=0.08, counter=counter)
    # 5 packs × 2 models need 10 × $1 = $10 worst-case; cap hard_stop at $4
    # so only a subset of trials may reserve.
    report = run_deepagent_eval(
        product_root=product,
        out_dir=tmp_path / "out_pool_hs",
        max_packs=5,
        k=1,
        n_concurrent=3,
        hard_stop_usd=Decimal("4.00"),
        reserve_usd=Decimal("1.00"),
        jobs_dir=tmp_path / "jobs_pool_hs",
        invoker=invoker,  # type: ignore[arg-type]
        offline=True,
        reclaim=False,
    )
    assert report.budget_stop is True
    assert report.total_spend_usd <= Decimal("4.00")
    # Ledger must settle without corruption under concurrency.
    assert report.ledger_path is not None
    ledger = BudgetLedger(report.ledger_path, cap_usd=Decimal("4.00"))
    summary = ledger.summary()
    assert summary.under_cap is True
    assert summary.has_unknown_billing is False
    assert summary.settled_call_count == counter["calls"]
    assert summary.open_call_count == 0
    assert summary.settled_exact_usd <= Decimal("4.00")
    # No invented rewards under concurrent path.
    assert report.invented_rewards is False
    for pack in report.pack_results:
        for model in pack.models:
            for trial in model.trials:
                assert trial.invented_reward is False
    # Mus have observed multi-worker peak before/around hard-stop.
    assert counter["max_inflight"] <= 3
