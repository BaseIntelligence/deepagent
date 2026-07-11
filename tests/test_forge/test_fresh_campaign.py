"""Offline tests for the fresh additional-budget campaign instrumentation."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import swe_forge.forge.fresh_campaign as campaign
from swe_forge.forge.fresh_campaign import (
    CAP_USD,
    DockerEvidenceCollector,
    DockerResource,
    DockerSnapshot,
    FreshCampaignConfig,
    FreshCampaignError,
    FreshCampaignLedger,
    FreshCandidateAuthority,
    StageEvidence,
    candidate_identity,
    run_fresh_campaign,
)
from swe_forge.forge.models import RepoSpec
from swe_forge.forge.pilot import CandidatePlan


def _plan(seed: int = 0) -> CandidatePlan:
    return CandidatePlan(
        repo=RepoSpec(
            repo_id="mahmoud/boltons",
            url="https://example.invalid/boltons.git",
            commit="a" * 40,
            commit_date="2024-01-01T00:00:00+00:00",
            language="python",
            license="MIT",
            instance_cap=5,
        ),
        generator="bug_combination",
        seed=seed,
        params={"faults": 2, "min_symbol_lines": 20, "prefer": "largest"},
    )


def test_fresh_ledger_has_independent_exact_cap_and_restart(tmp_path: Path) -> None:
    ledger = FreshCampaignLedger(
        tmp_path / "fresh.jsonl",
        run_id="fresh-run",
        worst_case_cost_usd="20",
        historical_ledger=tmp_path / "harvest_progress.json",
    )
    physical = ledger.reserve(
        candidate_identity="fresh-candidate",
        stage="fresh.stage-2",
        logical_call_id="call-0",
        model="offline/model",
    )
    assert ledger.total_active_reservations == Decimal("20")
    ledger.settle(
        physical,
        request_id="request-0",
        usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        cost="1.25",
        status="success",
        finish_reason="stop",
    )
    restarted = FreshCampaignLedger(
        tmp_path / "fresh.jsonl",
        run_id="fresh-run",
        worst_case_cost_usd="20",
        historical_ledger=tmp_path / "harvest_progress.json",
    )
    assert restarted.total_exact_cost == Decimal("1.25")
    assert restarted.total_active_reservations == Decimal("0")
    assert restarted.remaining_cap == CAP_USD - Decimal("1.25")
    assert restarted.events()[0]["candidate_identity"] == "fresh-candidate"


def test_fresh_ledger_refuses_reservation_over_cap_and_rejects_historical_reuse(
    tmp_path: Path,
) -> None:
    ledger = FreshCampaignLedger(
        tmp_path / "fresh.jsonl",
        run_id="fresh-run",
        worst_case_cost_usd="50",
        historical_ledger=tmp_path / "harvest_progress.json",
    )
    ledger.reserve(
        candidate_identity="first",
        stage="fresh.stage-1",
        logical_call_id="call-0",
        model="offline/model",
    )
    with pytest.raises(Exception, match="cap"):
        ledger.reserve(
            candidate_identity="second",
            stage="fresh.stage-1",
            logical_call_id="call-1",
            model="offline/model",
        )
    with pytest.raises(FreshCampaignError, match="must not reuse"):
        FreshCampaignLedger(
            tmp_path / "harvest_progress.json",
            run_id="fresh-run",
            worst_case_cost_usd="1",
            historical_ledger=tmp_path / "harvest_progress.json",
        )


def test_fresh_identity_authority_rejects_duplicate_and_historical_replay(
    tmp_path: Path,
) -> None:
    plan = _plan()
    identity = candidate_identity(plan)
    authority = FreshCandidateAuthority(
        tmp_path / "claims.jsonl", historical_identities=(identity,)
    )
    with pytest.raises(FreshCampaignError, match="historical"):
        authority.claim(plan, reason="should not replay")

    fresh = FreshCandidateAuthority(tmp_path / "fresh-claims.jsonl")
    fresh.claim(plan, reason="fresh unprocessed seed")
    with pytest.raises(FreshCampaignError, match="already attempted"):
        fresh.claim(plan, reason="duplicate")
    reloaded = FreshCandidateAuthority(tmp_path / "fresh-claims.jsonl")
    assert identity in reloaded.claimed_identities


def test_historical_dispositions_produce_replay_keys(tmp_path: Path) -> None:
    path = tmp_path / "dispositions.jsonl"
    path.write_text(
        json.dumps(
            {
                "repo_id": "mahmoud/boltons",
                "generator": "bug_combination",
                "seed": 0,
                "params": {"faults": 2, "min_symbol_lines": 20},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    historical = campaign.historical_candidate_identities(dispositions_path=path)
    authority = FreshCandidateAuthority(
        tmp_path / "claims.jsonl",
        historical_identities=(*historical, campaign.candidate_replay_key(_plan())),
    )
    with pytest.raises(FreshCampaignError, match="historical"):
        authority.claim(_plan(), reason="historical disposition")


def test_stage_evidence_is_ordered_and_durable(tmp_path: Path) -> None:
    path = tmp_path / "stage-evidence.log"
    evidence = StageEvidence(path)
    for stage in range(6):
        evidence.mark(stage)
    evidence.complete(0)
    assert path.read_text(encoding="utf-8").splitlines() == [
        *(f"Stage {stage}: started" for stage in range(6)),
        "exit_status: 0",
    ]
    with pytest.raises(FreshCampaignError, match="out of order"):
        evidence.mark(5)


def test_docker_snapshot_protects_resources_and_detects_teardown() -> None:
    protected = DockerResource("mission-test-pg", "id", "running", "start")
    before = DockerSnapshot(
        protected=(protected,),
        mission_owned=(
            DockerResource("swe-forge-fresh-run", "owned", "running", "start"),
        ),
        dangling_images=("old",),
    )
    after = DockerSnapshot(protected=(protected,), dangling_images=("old", "new"))
    comparison = DockerEvidenceCollector.compare(before, after)
    assert comparison["protected_equal"] is True
    assert comparison["mission_owned_teardown"] is True
    assert comparison["new_dangling_images"] == ["new"]
    with pytest.raises(FreshCampaignError, match="mission-owned"):
        DockerEvidenceCollector.compare(
            before,
            DockerSnapshot(
                protected=(protected,),
                mission_owned=(
                    DockerResource("swe-forge-fresh-new", "new", "running", "start"),
                ),
            ),
        )
    changed = DockerSnapshot(
        protected=(DockerResource("mission-test-pg", "different", "running", "start"),)
    )
    with pytest.raises(FreshCampaignError, match="protected"):
        DockerEvidenceCollector.compare(before, changed)


def test_fresh_campaign_stops_after_first_keep_and_marks_all_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = [_plan(0), _plan(1), _plan(2)]
    processed: list[int] = []

    async def process(plan: CandidatePlan, _workdir: Path) -> object:
        processed.append(plan.seed)
        return SimpleNamespace(
            failure_reason="",
            oracle_report=SimpleNamespace(details={}),
            calibration_report=SimpleNamespace(
                details={"band_filter": {"band_high": 0.5}},
                irt_discrimination=1.0,
            ),
        )

    class FakeProcessor:
        async def process(self, plan: CandidatePlan, workdir: Path) -> object:
            return await process(plan, workdir)

    request = SimpleNamespace(task_id="fresh-task-0")
    monkeypatch.setattr(campaign, "_keep_export_request", lambda _art: request)
    monkeypatch.setattr(
        campaign,
        "reconcile_recovery_reports",
        lambda *_args, **_kwargs: {
            "status": "reconciled",
            "exact_cost_usd": "0",
            "physical_calls": 0,
        },
    )
    published: list[object] = []

    def publish(value: object, *, result: object) -> str:
        published.append(value)
        return "generation-1"

    config = FreshCampaignConfig(
        plans=plans,
        out_dir=tmp_path / "output",
        ledger_path=tmp_path / "fresh.jsonl",
        authority_path=tmp_path / "claims.jsonl",
        evidence_path=tmp_path / "evidence.log",
        worst_case_cost_usd=Decimal("50"),
    )
    result = asyncio.run(
        run_fresh_campaign(
            config,
            processor=FakeProcessor(),  # type: ignore[arg-type]
            gold_prover=lambda _request: True,
            publisher=publish,  # type: ignore[arg-type]
        )
    )
    assert result.status == "kept"
    assert result.publication_generation == "generation-1"
    assert processed == [0]
    assert published == [request]
    assert result.stage_markers == tuple(
        f"Stage {stage}: started" for stage in range(6)
    )
    assert result.ledger["exact_cost_usd"] == "0"
    assert json.loads((tmp_path / "fresh.jsonl").read_text() or "[]") == []
    assert "terminal" in (tmp_path / "claims.jsonl").read_text()


def test_fresh_campaign_rejects_unfinished_claim_on_restart(tmp_path: Path) -> None:
    authority = FreshCandidateAuthority(tmp_path / "claims.jsonl")
    authority.claim(_plan(), reason="crash before processing")
    config = FreshCampaignConfig(
        plans=[_plan()],
        out_dir=tmp_path / "output",
        ledger_path=tmp_path / "fresh.jsonl",
        authority_path=tmp_path / "claims.jsonl",
        evidence_path=tmp_path / "evidence.log",
        historical_dispositions=None,
    )

    class NeverProcessor:
        async def process(self, _plan: CandidatePlan, _workdir: Path) -> object:
            raise AssertionError("unfinished claim must stop before processing")

    with pytest.raises(FreshCampaignError, match="unfinished"):
        asyncio.run(
            run_fresh_campaign(
                config,
                processor=NeverProcessor(),  # type: ignore[arg-type]
                gold_prover=lambda _request: True,
            )
        )


def test_fresh_campaign_rejects_changed_publication_predecessor(
    tmp_path: Path,
) -> None:
    config = FreshCampaignConfig(
        plans=[],
        out_dir=tmp_path / "output",
        ledger_path=tmp_path / "fresh.jsonl",
        authority_path=tmp_path / "claims.jsonl",
        evidence_path=tmp_path / "evidence.log",
        historical_dispositions=None,
    )
    publisher_calls: list[object] = []

    def publisher(request: object, *, result: object) -> str:
        publisher_calls.append(request)
        raise FreshCampaignError("publication predecessor changed")

    class NeverProcessor:
        async def process(self, _plan: CandidatePlan, _workdir: Path) -> object:
            raise AssertionError("candidate supply is empty")

    with pytest.raises(FreshCampaignError):
        asyncio.run(
            run_fresh_campaign(
                config,
                processor=NeverProcessor(),  # type: ignore[arg-type]
                gold_prover=lambda _request: True,
                publisher=publisher,  # type: ignore[arg-type]
            )
        )
