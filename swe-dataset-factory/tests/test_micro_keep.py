"""Offline unit tests for live micro-keep pipeline (VAL-CROSS-002 wiring).

No OpenRouter network. Uses FakeOracle + scripted panel soft solver that
achieves an in-band keep, then asserts export includes panel hardness +
labeled source_track. Also tests fail-closed export without panel.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_factory.accounting import BudgetLedger
from swe_factory.cli import app
from swe_factory.export.jsonl import read_tasks_jsonl, validate_export_fields
from swe_factory.export.workspace import ExportError, write_export_bundle
from swe_factory.openrouter import ChatResult, ScriptedChatClient, TokenUsage
from swe_factory.panel.runner import REQUIRED_PANEL_MODELS
from swe_factory.panel.score_solver import extract_unified_diff
from swe_factory.pipeline.micro_keep import (
    require_panel_hardness,
    run_micro_keep,
)
from swe_factory.producers.synth import produce_from_green_fixture
from swe_factory.schema import PanelHardness, SourceTrack
from swe_factory.sources.allowlist import get_seed
from swe_factory.sources.clone import is_immutable_sha

runner = CliRunner()
GROK = "x-ai/grok-4.5"
KIMI = "moonshotai/kimi-k2.6"
OPUS = "anthropic/claude-opus-4.8"
PAIR = (GROK, KIMI)
TRIAD = PAIR  # Real-PR wave pair; historical name kept for scripted clients


def test_boltons_allowlist_is_immutable_sha() -> None:
    seed = get_seed("python_boltons")
    assert is_immutable_sha(seed.base_commit)
    assert len(seed.base_commit) == 40


def test_extract_unified_diff_from_fence() -> None:
    text = (
        "Here is a fix:\n"
        "```diff\n"
        "diff --git a/demo_pkg/math_ops.py b/demo_pkg/math_ops.py\n"
        "--- a/demo_pkg/math_ops.py\n"
        "+++ b/demo_pkg/math_ops.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add(a: int, b: int) -> int:\n"
        "-    return a - b\n"
        "+    return a + b\n"
        "```\n"
        "Done."
    )
    patch = extract_unified_diff(text)
    assert patch.startswith("diff --git ")
    assert "return a + b" in patch


def test_require_panel_hardness_fail_closed() -> None:
    cand = produce_from_green_fixture(mutation_kind="multi_fault")
    with pytest.raises((ExportError, Exception)):
        require_panel_hardness(cand.task)
    with_panel = cand.task.model_copy(
        update={
            "panel": PanelHardness(
                grok_4_5=0.0,
                opus_4_8=0.5,
                pass_at_k=0.25,
                discrimination=2.0,
            )
        }
    )
    require_panel_hardness(with_panel)  # does not raise


def test_write_export_bundle_fail_closed_without_panel(tmp_path: Path) -> None:
    cand = produce_from_green_fixture(
        mutation_kind="multi_fault",
        work_root=tmp_path / "p",
    )
    with pytest.raises(ExportError):
        write_export_bundle(
            tasks=[cand.task],
            out_dir=tmp_path / "export_bad",
            broken_repos={cand.task.instance_id: cand.broken_workspace},
            require_panel=True,
        )


def test_offline_micro_keep_exports_certified_keep(tmp_path: Path) -> None:
    """sources→env→produce→oracle→panel keep→export with scripted band."""
    # k=2 each → scripted soft_solver: grok never solves, kimi solves 1/2
    # → aggregate 1/4=0.25, discrimination=2.0 → in-band keep.
    # Soft solver is index-driven (patch apply would need the same mutation gold).
    responses: list[ChatResult | Exception] = []
    for model in PAIR:
        for i in range(2):
            responses.append(
                ChatResult(
                    model=model,
                    text=f"PATCH {model} #{i}\n",
                    usage=TokenUsage(prompt_tokens=20, completion_tokens=40, total_tokens=60),
                    request_id=f"micro-{model}-{i}",
                    cost_usd=Decimal("0.001"),
                    finish_reason="stop",
                    raw_usage={},
                )
            )
    client = ScriptedChatClient(responses=responses)

    counters = {GROK: 0, KIMI: 0}

    def soft_solver(model: str, messages, chat) -> bool:  # noqa: ANN001
        del messages, chat
        i = counters.get(model, 0)
        counters[model] = i + 1
        # Kimi trial 0 only → 1/4 aggregate with discrimination
        return model == KIMI and i == 0

    ledger_path = tmp_path / "ledger.jsonl"
    ledger = BudgetLedger(
        ledger_path,
        cap_usd=Decimal("600"),
        worst_case_cost_usd=Decimal("0.10"),
        run_id="test-micro",
    )

    result = run_micro_keep(
        out_dir=tmp_path / "micro",
        seed_id="fixture_tiny_green",
        mutation="multi_fault",
        ledger=ledger,
        micro_cap_usd=Decimal("80"),
        panel_k=2,
        panel_reserve_usd=Decimal("0.10"),
        use_docker_oracle=False,
        use_docker_envbuild=False,
        soft_backend="local",
        soft_solver=soft_solver,
        live_panel=True,  # reserve/settle path with injected client
        client=client,
        require_immutable_sha=True,
    )

    assert result.ok is True
    assert result.is_keep is True
    assert result.escalated is False
    assert result.task is not None
    assert result.task.panel is not None
    assert result.task.panel.pass_at_k is not None
    assert 0.0 < result.task.panel.pass_at_k <= 0.5
    assert result.task.panel.discrimination is not None
    assert result.task.panel.discrimination >= 1.0
    track = result.task.source_track
    track_val = track.value if hasattr(track, "value") else str(track)
    assert track_val == SourceTrack.SYNTHETIC_GROUNDED.value
    assert is_immutable_sha(result.task.base_commit)

    assert result.export_dir is not None
    export_jsonl = Path(result.export_dir) / "tasks.jsonl"
    assert export_jsonl.is_file()
    records = read_tasks_jsonl(export_jsonl)
    assert len(records) == 1
    validate_export_fields(records[0], require_panel=True)
    assert records[0].panel is not None
    assert records[0].panel.pass_at_k == result.task.panel.pass_at_k

    # Stage order retained
    stage_names = [s.stage for s in result.stages]
    assert stage_names.index("sources") < stage_names.index("envbuild")
    assert stage_names.index("envbuild") < stage_names.index("produce")
    assert stage_names.index("produce") < stage_names.index("oracle")
    assert stage_names.index("oracle") < stage_names.index("panel")
    assert stage_names.index("panel") < stage_names.index("export")
    assert result.funnel["panel_keep"] == 1
    assert result.funnel["export_ok"] == 1
    assert result.spend_exact_usd >= 0

    # Agent workspace omits gold
    ws = Path(result.export_dir) / "tasks" / result.instance_id  # type: ignore[operator]
    assert (ws / "problem_statement.md").is_file()
    assert not (ws / "gold.patch").exists()
    meta = json.loads((ws / "task_meta.agent.json").read_text(encoding="utf-8"))
    assert "gold_patch" not in meta
    assert meta["source_track"] == "synthetic_grounded"


def test_offline_micro_keep_panel_drop_records_funnel(tmp_path: Path) -> None:
    # All responses non-diff → soft solver never solves → solve-none drop
    responses: list[ChatResult | Exception] = []
    for model in PAIR:
        for i in range(2):
            responses.append(
                ChatResult(
                    model=model,
                    text="no patch here",
                    usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                    request_id=f"drop-{model}-{i}",
                    cost_usd=Decimal("0"),
                    finish_reason="stop",
                    raw_usage={},
                )
            )
    client = ScriptedChatClient(responses=responses)
    ledger = BudgetLedger(
        tmp_path / "drop_ledger.jsonl",
        cap_usd=Decimal("600"),
        worst_case_cost_usd=Decimal("0.05"),
        run_id="test-drop",
    )
    result = run_micro_keep(
        out_dir=tmp_path / "micro_drop",
        seed_id="fixture_tiny_green",
        ledger=ledger,
        panel_k=2,
        panel_reserve_usd=Decimal("0.05"),
        use_docker_oracle=False,
        use_docker_envbuild=False,
        soft_backend="local",
        live_panel=True,
        client=client,
    )
    assert result.is_keep is False
    assert result.ok is True
    assert result.funnel["panel_drop"] == 1
    assert result.funnel["export_ok"] == 0
    assert "solve-none" in result.reason or "panel drop" in result.reason


def test_cli_micro_keep_offline_help_and_listed() -> None:
    help_r = runner.invoke(app, ["--help"])
    assert help_r.exit_code == 0
    assert "micro-keep" in help_r.stdout

    r = runner.invoke(app, ["micro-keep", "--help"])
    assert r.exit_code == 0
    assert "sources→env→produce→oracle→panel→export" in r.stdout or "micro" in r.stdout.lower()


def test_required_panel_models_constant() -> None:
    assert list(REQUIRED_PANEL_MODELS) == [GROK, KIMI]
    assert OPUS not in REQUIRED_PANEL_MODELS
