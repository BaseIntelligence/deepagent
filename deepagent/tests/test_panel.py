"""Offline unit tests for hardness band + panel runner (Real-PR pair).

Covers VAL-RPANEL-001..005 offline: two required models (Grok + Kimi), full-matrix
keep, hard budget stop at remaining $0, no invented rewards, Pier scaffold meta.
Also retains band rules (solve-all/none/out-of-band/disc).
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from swe_factory.accounting import BudgetLedger
from swe_factory.config import DEFAULT_PANEL_MODELS
from swe_factory.openrouter import ChatResult, ScriptedChatClient, TokenUsage
from swe_factory.panel.band import (
    DEFAULT_BAND_HIGH,
    DEFAULT_DISCRIMINATION_FLOOR,
    RULE_KEEP,
    RULE_LOW_DISCRIMINATION,
    RULE_OUT_OF_BAND,
    RULE_SOLVE_ALL,
    RULE_SOLVE_NONE,
    compute_discrimination,
    compute_pass_at_k,
    decision_from_model_stats,
)
from swe_factory.panel.pier_scaffold import (
    build_panel_job_spec,
    dry_run_panel_scaffold,
)
from swe_factory.panel.runner import (
    PANEL_SCAFFOLD_AGENT,
    PANEL_SCAFFOLD_NAME,
    PANEL_SCAFFOLD_RUNTIME,
    REAL_PR_PANEL_MODELS,
    REQUIRED_PANEL_MODELS,
    PanelRunnerError,
    canary_affordable,
    discover_real_pr_panel_keeps,
    full_panel_affordable,
    offline_panel_from_matrix,
    offline_tworollout_borderline_matrix,
    run_panel,
    run_panel_until_budget_zero,
)

GROK = "x-ai/grok-4.5"
KIMI = "moonshotai/kimi-k2.6"
OPUS = "anthropic/claude-opus-4.8"
PAIR = (GROK, KIMI)


def test_required_panel_models_are_grok_kimi_pair() -> None:
    """VAL-RPANEL-001: required models are exactly Grok 4.5 + Kimi K2.6."""
    assert REQUIRED_PANEL_MODELS == DEFAULT_PANEL_MODELS
    assert REQUIRED_PANEL_MODELS == REAL_PR_PANEL_MODELS
    assert list(REQUIRED_PANEL_MODELS) == [GROK, KIMI]
    assert OPUS not in REQUIRED_PANEL_MODELS


def test_pass_at_k_math() -> None:
    assert compute_pass_at_k(0, 4) == 0.0
    assert compute_pass_at_k(1, 4) == 0.25
    assert compute_pass_at_k(2, 4) == 0.5
    assert compute_pass_at_k(4, 4) == 1.0


def test_drop_solve_all() -> None:
    decision = decision_from_model_stats(
        {GROK: (4, 4), KIMI: (4, 4)},
    )
    assert decision.verdict == "drop"
    assert decision.rule == RULE_SOLVE_ALL
    assert decision.is_keep is False
    assert decision.frontier_pass_at_k == 1.0


def test_drop_solve_none() -> None:
    decision = decision_from_model_stats(
        {GROK: (0, 4), KIMI: (0, 4)},
    )
    assert decision.verdict == "drop"
    assert decision.rule == RULE_SOLVE_NONE
    assert decision.frontier_pass_at_k == 0.0


def test_drop_out_of_band_above_half() -> None:
    # 6/8 = 0.75 > 0.5
    decision = decision_from_model_stats(
        {GROK: (3, 4), KIMI: (3, 4)},
    )
    assert decision.verdict == "drop"
    assert decision.rule == RULE_OUT_OF_BAND
    assert decision.frontier_pass_at_k == 0.75


def test_keep_borderline_band_with_discrimination() -> None:
    # Aggregate 2/8 = 0.25; spread 0.5-0.0 = 0.5 → discrimination = 2.0
    decision = decision_from_model_stats(
        {GROK: (0, 4), KIMI: (2, 4)},
    )
    assert decision.verdict == "keep"
    assert decision.rule == RULE_KEEP
    assert 0.0 < decision.frontier_pass_at_k <= DEFAULT_BAND_HIGH
    assert decision.discrimination >= DEFAULT_DISCRIMINATION_FLOOR
    hardness = decision.to_panel_hardness()
    assert hardness.pass_at_k == decision.frontier_pass_at_k
    assert hardness.grok_4_5 == 0.0
    assert hardness.kimi_k2_6 == 0.5
    assert hardness.opus_4_8 is None
    assert hardness.discrimination is not None
    assert hardness.discrimination >= 1.0


def test_drop_low_discrimination_flat_partial() -> None:
    # Flat partial on both → discrimination 0
    decision = decision_from_model_stats(
        {GROK: (1, 4), KIMI: (1, 4)},
    )
    assert decision.verdict == "drop"
    assert decision.rule == RULE_LOW_DISCRIMINATION
    assert decision.frontier_pass_at_k == 0.25
    assert decision.discrimination < DEFAULT_DISCRIMINATION_FLOOR


def test_discrimination_just_meets_floor() -> None:
    disc = compute_discrimination({GROK: 0.0, KIMI: 0.25})
    assert disc == pytest.approx(1.0)
    # k=4 each: grok 0, kimi 1/4 → aggregate 1/8, disc = 1.0
    decision = decision_from_model_stats({GROK: (0, 4), KIMI: (1, 4)})
    assert decision.frontier_pass_at_k == pytest.approx(1 / 8)
    assert decision.discrimination == pytest.approx(1.0)
    assert decision.verdict == "keep"


def test_offline_panel_records_required_models_and_band_keep(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    result = offline_panel_from_matrix(
        task_id="cand-borderline-1",
        solve_matrix={
            GROK: [False, False, False, False],
            KIMI: [True, False, True, False],
        },
        ledger_path=path,
        cap_usd=Decimal("600"),
        pack_path=tmp_path / "packs" / "cand-borderline-1",
        pack_id="cand-borderline-1",
    )
    assert list(result.reserved_models) == [GROK, KIMI]
    assert [m.model for m in result.models] == [GROK, KIMI]
    assert result.models[0].pass_at_k == 0.0
    assert result.models[1].pass_at_k == 0.5
    assert result.decision.frontier_pass_at_k == pytest.approx(2 / 8)
    assert result.is_keep is True
    assert result.decision.rule == RULE_KEEP
    assert result.panel_complete is True
    assert result.budget_stop is False
    assert result.scaffold_meta.runtime == PANEL_SCAFFOLD_RUNTIME
    assert result.scaffold_meta.agent == PANEL_SCAFFOLD_AGENT
    assert result.scaffold_meta.name == PANEL_SCAFFOLD_NAME
    assert result.scaffold_meta.pack_id == "cand-borderline-1"

    # Ledger lines link stage/task for every call (2 models × 4 = 8)
    events = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    reserves = [e for e in events if e["event"] == "reserve"]
    settles = [e for e in events if e["event"] == "settle"]
    assert len(reserves) == 8
    assert len(settles) == 8
    for e in reserves + settles:
        assert e["task_id"] == "cand-borderline-1"
        assert e["stage"] == "hardness-panel"
        assert e["model"] in (GROK, KIMI)
        assert e["model"] != OPUS

    hardness = result.panel_hardness()
    assert hardness.pass_at_k == pytest.approx(2 / 8)
    assert hardness.grok_4_5 == 0.0
    assert hardness.kimi_k2_6 == 0.5

    payload = result.to_dict()
    assert payload["scaffold_meta"]["runtime"] == "pier"
    assert payload["scaffold_meta"]["agent"] == "mini-swe-agent"


def test_offline_panel_drops_solve_all(tmp_path: Path) -> None:
    result = offline_panel_from_matrix(
        task_id="cand-easy",
        solve_matrix={
            GROK: [True, True],
            KIMI: [True, True],
        },
        ledger_path=tmp_path / "l.jsonl",
    )
    assert result.is_keep is False
    assert result.decision.rule == RULE_SOLVE_ALL


def test_offline_panel_drops_solve_none(tmp_path: Path) -> None:
    result = offline_panel_from_matrix(
        task_id="cand-impossible",
        solve_matrix={
            GROK: [False, False],
            KIMI: [False, False],
        },
        ledger_path=tmp_path / "l.jsonl",
    )
    assert result.is_keep is False
    assert result.decision.rule == RULE_SOLVE_NONE


def test_run_panel_requires_both_pair_models(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "l.jsonl", cap_usd=600, worst_case_cost_usd=0.01)
    client = ScriptedChatClient(
        responses=[
            ChatResult(
                model=GROK,
                text="",
                usage=TokenUsage(),
                request_id="x",
                cost_usd=Decimal("0"),
                finish_reason="stop",
                raw_usage={},
            )
        ]
    )
    with pytest.raises(PanelRunnerError, match="required set"):
        run_panel(
            task_id="t",
            problem_statement="bug",
            ledger=ledger,
            client=client,
            models=[GROK, OPUS],  # missing kimi
            k=1,
            allow_missing_cost_as_zero=True,
        )


def test_run_panel_rejects_swapped_pair(tmp_path: Path) -> None:
    """Silent substitution / reorder of the exact two-model list fails closed."""
    ledger = BudgetLedger(tmp_path / "l.jsonl", cap_usd=600, worst_case_cost_usd=0.01)
    client = ScriptedChatClient(responses=[])
    with pytest.raises(PanelRunnerError, match="exact pair|required set"):
        run_panel(
            task_id="t",
            problem_statement="bug",
            ledger=ledger,
            client=client,
            models=[KIMI, GROK],
            k=1,
            allow_missing_cost_as_zero=True,
        )


def test_canary_affordable_respects_remaining(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "l.jsonl", cap_usd="2.00", worst_case_cost_usd="1.50")
    # Empty: need 1.50 * 2 models * k=1 = 3.00 > 2.00 → unaffordable
    assert canary_affordable(ledger, k=1, reserve_usd=Decimal("1.50")) is False
    assert canary_affordable(ledger, k=1, reserve_usd=Decimal("0.50")) is True
    # full panel (k=2) even stricter: 0.50 * 2 * 2 = 2.00 == remaining
    assert full_panel_affordable(ledger, k=2, reserve_usd=Decimal("0.50")) is True
    assert full_panel_affordable(ledger, k=2, reserve_usd=Decimal("0.51")) is False


def test_unknown_billing_cannot_keep(tmp_path: Path) -> None:
    """Missing provider cost fail-closes keeps even with a would-be keep matrix."""
    ledger = BudgetLedger(tmp_path / "l.jsonl", cap_usd=600, worst_case_cost_usd=1)

    # 2 models × k=2: Grok fails, Kimi solves first and misses cost on second.
    responses: list[ChatResult | Exception] = []
    for model, texts in (
        (GROK, ["g0", "g1"]),
        (KIMI, ["k0", "k1"]),
    ):
        for t in texts:
            cost = None if (model == KIMI and t == "k1") else Decimal("0.01")
            responses.append(
                ChatResult(
                    model=model,
                    text=t,
                    usage=TokenUsage(1, 1, 2),
                    request_id=f"{model}-{t}",
                    cost_usd=cost,
                    finish_reason="stop",
                    raw_usage={},
                )
            )
    client = ScriptedChatClient(responses=responses)

    def solver(model: str, messages: object, chat: ChatResult | None) -> bool:
        del messages
        return bool(chat and chat.text == "k0")

    result = run_panel(
        task_id="u-bill",
        problem_statement="issue",
        ledger=ledger,
        client=client,
        models=list(PAIR),
        k=2,
        soft_solver=solver,
        allow_missing_cost_as_zero=False,
        reserve_usd=Decimal("0.05"),
    )
    assert ledger.has_unknown_billing() is True
    assert result.is_keep is False


def test_stops_when_remaining_zero_mid_panel(tmp_path: Path) -> None:
    """VAL-RPANEL-002/004: hard stop when remaining cannot fund next reserve.

    Cap of $0.02 with reserve $0.01 → only 2 of 4 rollouts (2 models × k=2).
    No further calls after budget stop; panel incomplete; not a keep; no invented
    reward rows for skipped rollouts.
    """
    ledger = BudgetLedger(
        tmp_path / "budget_stop.jsonl",
        cap_usd=Decimal("0.02"),
        worst_case_cost_usd=Decimal("0.01"),
    )
    responses: list[ChatResult | Exception] = []
    for model in PAIR:
        for i in range(4):
            responses.append(
                ChatResult(
                    model=model,
                    text=f"{model}-{i}",
                    usage=TokenUsage(1, 1, 2),
                    request_id=f"{model}-{i}",
                    cost_usd=Decimal("0.01"),
                    finish_reason="stop",
                    raw_usage={},
                )
            )
    client = ScriptedChatClient(responses=responses)

    result = run_panel(
        task_id="budget-stop-mid",
        problem_statement="bug under budget pressure",
        ledger=ledger,
        client=client,
        models=list(PAIR),
        k=2,
        reserve_usd=Decimal("0.01"),
        soft_solver=lambda *_a, **_k: False,
        allow_missing_cost_as_zero=False,
        stop_on_budget=True,
        pack_path="/tmp/fake-pack/budget-stop-mid",
        pack_id="budget-stop-mid",
    )

    assert result.budget_stop is True
    assert result.panel_complete is False
    assert result.is_keep is False
    assert result.completed_rollouts == 2  # cap 0.02 / reserve 0.01
    assert result.planned_rollouts == 4
    assert result.stop_reason is not None
    assert "budget_stop" in result.stop_reason
    assert result.decision.rule == "budget-stop-incomplete"
    # No fabricated rollouts beyond completed set
    all_rollouts = [r for m in result.models for r in m.rollouts]
    assert len(all_rollouts) == 2
    assert all(not r.skipped_budget for r in all_rollouts)
    assert result.scaffold_meta.runtime == "pier"
    assert ledger.remaining_usd() == Decimal("0")
    assert ledger.summary().open_call_count == 0
    # Extra scripted responses must remain unused (no over-call)
    assert client._index == 2  # noqa: SLF001


def test_run_panel_until_budget_zero_multi_keeps(tmp_path: Path) -> None:
    """Full panel on keeps until remaining cannot fund the next full matrix.

    Cap $0.04, reserve $0.01, k=1, 2 models → full matrix costs $0.02 / keep.
    First keep completes; second completes; third skipped after remaining < need.
    """
    ledger = BudgetLedger(
        tmp_path / "multi.jsonl",
        cap_usd=Decimal("0.04"),
        worst_case_cost_usd=Decimal("0.01"),
    )

    # 2 keeps × 2 models × k=1 = 4 calls; then third skipped.
    responses: list[ChatResult | Exception] = []
    for keep_i in range(2):
        for model in PAIR:
            responses.append(
                ChatResult(
                    model=model,
                    text=f"keep{keep_i}-{model}",
                    usage=TokenUsage(1, 1, 2),
                    request_id=f"k{keep_i}-{model}",
                    cost_usd=Decimal("0.01"),
                    finish_reason="stop",
                    raw_usage={},
                )
            )
    # Extra would-be responses for keep3 — must never be consumed
    for model in PAIR:
        responses.append(
            ChatResult(
                model=model,
                text=f"FORBIDDEN-{model}",
                usage=TokenUsage(1, 1, 2),
                request_id=f"forbidden-{model}",
                cost_usd=Decimal("0.01"),
                finish_reason="stop",
                raw_usage={},
            )
        )

    client = ScriptedChatClient(responses=responses)
    keeps = [
        {
            "task_id": f"keep-{i}",
            "problem_statement": f"multi-file bug for keep {i}",
            "pack_path": f"/tmp/packs/keep-{i}",
            "pack_id": f"keep-{i}",
        }
        for i in range(3)
    ]

    batch = run_panel_until_budget_zero(
        keeps=keeps,
        ledger=ledger,
        client=client,
        models=list(PAIR),
        k=1,
        reserve_usd=Decimal("0.01"),
        soft_solver=lambda *_a, **_k: False,
        allow_missing_cost_as_zero=False,
    )

    assert batch.budget_stop is True
    payload = batch.to_dict()
    assert payload["invented_rewards"] is False
    assert set(batch.completed_keep_ids) == {"keep-0", "keep-1"}
    assert batch.skipped_keep_ids == ["keep-2"]
    assert batch.partial_keep_ids == []
    assert len(batch.keep_results) == 2
    for r in batch.keep_results:
        assert r.panel_complete is True
        assert list(r.reserved_models) == list(PAIR)
        assert r.scaffold_meta.agent == "mini-swe-agent"
        assert OPUS not in r.reserved_models
    # Remaining after 4 × $0.01 = $0
    assert ledger.remaining_usd() == Decimal("0")
    assert "budget_stop" in (batch.stop_reason or "")
    assert client._index < len(responses)  # noqa: SLF001
    leftover = responses[client._index :]  # noqa: SLF001
    assert any("FORBIDDEN" in str(getattr(x, "text", "")) for x in leftover)


def test_offline_borderline_matrix_includes_pair_only() -> None:
    matrix = offline_tworollout_borderline_matrix()
    assert set(matrix.keys()) == set(PAIR)
    assert OPUS not in matrix
    assert all(len(v) == 4 for v in matrix.values())


def test_pier_scaffold_dry_run_records_meta(tmp_path: Path) -> None:
    pack = tmp_path / "tasks" / "demo-pack"
    pack.mkdir(parents=True)
    (pack / "task.toml").write_text("[task]\nid='demo-pack'\n", encoding="utf-8")
    inv = dry_run_panel_scaffold(
        pack_path=pack,
        pack_id="demo-pack",
        model=GROK,
        jobs_dir=tmp_path / "jobs",
    )
    assert inv.ok is True
    assert inv.mode == "dry-run"
    assert inv.invented_reward is False
    assert inv.reward is None
    assert inv.scaffold_meta.runtime == "pier"
    assert inv.scaffold_meta.agent == "mini-swe-agent"
    assert inv.scaffold_meta.pack_id == "demo-pack"
    assert inv.spec.agent == "mini-swe-agent"
    cfg = inv.dry_run_config
    assert cfg["agent"]["name"] == "mini-swe-agent"
    assert cfg["metadata"]["scaffold"] == PANEL_SCAFFOLD_NAME

    spec = build_panel_job_spec(pack_path=pack, pack_id="demo-pack", model=KIMI)
    assert spec.model == KIMI
    assert "mini-swe" in spec.agent


def test_discover_real_pr_panel_keeps(tmp_path: Path) -> None:
    """Discover keeps only from real structured packs (no invented statements)."""
    tasks = tmp_path / "product" / "tasks"
    ok = tasks / "real-keep-a"
    ok.mkdir(parents=True)
    (ok / "instruction.md").write_text("Fix multi-file PR regression.\n", encoding="utf-8")
    (ok / "task.toml").write_text("[task]\nid='real-keep-a'\n", encoding="utf-8")
    empty = tasks / "no-instruction"
    empty.mkdir()
    (empty / "task.toml").write_text("[task]\nid='empty'\n", encoding="utf-8")

    found = discover_real_pr_panel_keeps([tmp_path / "product"])
    assert len(found) == 1
    assert found[0]["task_id"] == "real-keep-a"
    assert "Fix multi-file" in found[0]["problem_statement"]
    assert found[0]["source_track"] == "real_pr"


def test_extras_allowed_when_pair_present(tmp_path: Path) -> None:
    """Optional Opus is allowed only as an explicit *extra* with the pair present."""
    path = tmp_path / "ledger.jsonl"
    result = offline_panel_from_matrix(
        task_id="pair-plus-opus",
        solve_matrix={
            GROK: [False, False],
            KIMI: [True, False],
            OPUS: [False, False],
        },
        ledger_path=path,
    )
    assert GROK in result.reserved_models
    assert KIMI in result.reserved_models
    # Trio is voluntary; keep band still applies over presented models.
    assert result.panel_complete is True
