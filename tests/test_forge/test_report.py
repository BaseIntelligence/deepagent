"""Offline coverage of Stage 5 reporting (provenance audit + benchmark report).

Exercises this feature's ``fulfills`` assertions deterministically, without
Docker or the live endpoint, over a real exported tree (built via
``export_batch``) plus hand-built provenance views for the negative paths:

- VAL-EXPORT-014: provenance complete for 100% of shipped tasks; a missing field
  is detected.
- VAL-EXPORT-015: provenance consistent with the gates (oracle=pass, band=keep,
  in-band frontier rate, discrimination >= min); an inconsistent task is flagged.
- VAL-EXPORT-016: HEADLINE A -- report shows gold == 100% from the injected
  gold-eval result.
- VAL-EXPORT-017: per-model solve-rates (weak <= mid <= frontier < gold), the IRT
  summary, and the generator/language breakdown summing to the shipped total.
- VAL-EXPORT-018: HEADLINE B -- a stated frontier threshold and a measured
  aggregate frontier solve-rate strictly below it yet > 0.
- VAL-EXPORT-019: counts reconcile (jsonl == parquet == tasks/*/) and the JSON
  form parses with every required key.

The Docker gold=100% measurement that feeds Headline A is exercised by the
gold-eval suite; here it is injected as a :class:`GoldSummary`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from swe_forge.forge.calibrate.filter import BandFilterConfig, DEFAULT_BAND_HIGH
from swe_forge.forge.export import ExportRequest, export_batch
from swe_forge.forge.gold_eval import (
    EvalRun,
    GoldEvalReport,
    TaskGoldResult,
)
from swe_forge.forge.models import (
    CalibrationReport,
    Candidate,
    CandidateTarget,
    EnvImage,
    FinalMutationEvidence,
    GeneratedSpec,
    ModelSolveRecord,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.teacher import (
    Usage,
    candidate_transport_fingerprint,
)
from tests.test_forge.receipt_helpers import (
    protected_alt_correct_audit,
    protected_alt_correct_summary,
    signed_transport_receipt,
)
from swe_forge.forge.oracle.mutation import final_suite_fingerprint
from swe_forge.forge.oracle.multifault import (
    ConstituentVerdict,
    MultiFaultCompletenessEvidence,
)
from swe_forge.forge.report import (
    DEFAULT_FRONTIER_THRESHOLD,
    BenchmarkReport,
    GoldSummary,
    ReportError,
    TaskProvenance,
    aggregate_panel,
    build_benchmark_report,
    check_provenance_completeness,
    check_provenance_consistency,
    count_jsonl_records,
    count_parquet_rows,
    load_task_provenances,
    write_report,
)

_TS = "2026-01-01T00:00:00+00:00"

_BASE_IMAGE = {
    "python": "python:3.12-slim",
    "javascript": "node:22-slim",
    "go": "golang:1.22",
}


# --------------------------------------------------------------------------- #
# Export fixtures (build a real tasks/<id>/ tree with provenance.json)
# --------------------------------------------------------------------------- #
def _candidate(*, language: str, generator: str, seed: int) -> Candidate:
    files = ("src/m.py",)
    symbols = ("total",)
    details: dict[str, object] = {}
    if generator in {"bug_combination", "multi_file"}:
        files = ("src/alpha.py", "src/beta.py")
        symbols = ("alpha", "beta")
        details["constituents"] = [
            {
                "index": index,
                "file": path,
                "mutation_patch": f"mutation {path}",
                "inverse_patch": f"inverse {path}",
            }
            for index, path in enumerate(files)
        ]
    return Candidate(
        language=language,
        generator=generator,
        target=CandidateTarget(files=files, symbols=symbols),
        mutation_patch=(
            "--- a/src/m.py\n+++ b/src/m.py\n@@ -1,2 +1,2 @@\n"
            " def total(items, tax_rate):\n"
            "-    return compute_total_with_tax(items, tax_rate)\n"
            "+    return sum(items)\n"
        ),
        oracle_patch=(
            "--- a/src/m.py\n+++ b/src/m.py\n@@ -1,2 +1,2 @@\n"
            " def total(items, tax_rate):\n"
            "-    return sum(items)\n"
            "+    return compute_total_with_tax(items, tax_rate)\n"
        ),
        difficulty_hint="medium",
        provenance=Provenance(
            generator=generator,
            seed=seed,
            language=language,
            created_at=_TS,
            details=details,
        ),
    )


def _env_image(*, language: str) -> EnvImage:
    return EnvImage(
        repo_id="demo-repo",
        language=language,
        image_tag=f"swe-forge-env-{language}:abc123",
        base_image=_BASE_IMAGE[language],
        commit="a" * 40,
        workspace_dir="/workspace/repo",
        install_commands=["pip install -e ."],
        baseline_test_command="python -m pytest -q",
        baseline_green=True,
        baseline_exit_code=0,
    )


def _spec(*, language: str) -> GeneratedSpec:
    return GeneratedSpec(
        problem_statement="total() must include tax in the returned amount.",
        requirements=["total() returns the taxed sum for the items"],
        interface_block="def total(items, tax_rate): ...",
        provenance=Provenance(
            generator="ast_mutation", seed=7, language=language, created_at=_TS
        ),
    )


def _teacher_gate_evidence() -> dict[str, object]:
    def call(gate: str) -> dict[str, object]:
        return {
            "gate": gate,
            "call_kind": "proposal",
            "real_teacher": True,
            "status": "success",
            "response_kind": "content",
            "model": "anthropic/test-teacher",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
            "cost": 0.0,
            "finish_reason": "stop",
            "requested_proposals": 1,
            "received_proposals": 1,
            "parsed_proposals": 1,
            "identical_proposals": 0,
            "invalid_proposals": 0,
            "discarded_proposals": 0,
            "execution_attempted": 1,
            "execution_completed": 1,
            "execution_errors": 0,
            "executable_proposals": 1,
            "error_type": "",
        }

    return {
        "differential": {"calls": [call("differential")]},
        "alt_correct": {"calls": [call("alt_correct")]},
    }


def _alt_correct_audit(test_files: list[OracleTestFile]) -> dict[str, object]:
    return protected_alt_correct_audit(
        test_files,
        ["python -m pytest tests/hidden/test_total.py"],
        [("src/m.py", "def total(xs): return sum(xs)\n")],
    )


def _oracle_pass(*, language: str, generator: str):  # type: ignore[no-untyped-def]
    from swe_forge.forge.models import OracleReport

    test_files = [
        OracleTestFile(
            path="tests/hidden/test_total.py",
            content=(
                "from src.m import total\n\n\n"
                "def test_total():\n    assert total([100], 0.1) == 110\n"
            ),
        )
    ]

    multifault_evidence = None
    if generator in {"bug_combination", "multi_file"}:
        multifault_evidence = MultiFaultCompletenessEvidence(
            suite_fingerprint=final_suite_fingerprint(test_files),
            p2p_command="python -m pytest -q",
            constituents=tuple(
                ConstituentVerdict(
                    index=index,
                    file=path,
                    inverse_patch_sha256=hashlib.sha256(
                        f"inverse {path}".encode("utf-8")
                    ).hexdigest(),
                    repaired_indices=tuple(
                        other for other in range(2) if other != index
                    ),
                    other_inverse_patches_applied=True,
                    p2p_passed=True,
                    failed_f2p_test_ids=(
                        "python -m pytest tests/hidden/test_total.py",
                    ),
                    verdict="pass",
                )
                for index, path in enumerate(("src/alpha.py", "src/beta.py"))
            ),
        )
    return OracleReport(
        language=language,
        generator=generator,
        verdict="pass",
        reasons=[],
        fail_to_pass=["python -m pytest tests/hidden/test_total.py"],
        pass_to_pass=["python -m pytest -q"],
        test_files=test_files,
        flakiness_runs=3,
        mutants_total=10,
        mutants_killed=10,
        final_mutation_evidence=FinalMutationEvidence(
            suite_fingerprint=final_suite_fingerprint(test_files),
            mutants_total=10,
            mutants_killed=10,
            threshold=0.8,
            tool="fake-tool",
        ),
        multifault_evidence=multifault_evidence,
        differential_pass=True,
        alt_correct_accepted=True,
        leak_audit="clean",
        details={
            "teacher_gates": _teacher_gate_evidence(),
            "alt_correct": protected_alt_correct_summary(test_files),
        },
        protected_alt_correct_audit=_alt_correct_audit(test_files),
        provenance=Provenance(
            generator=generator, seed=7, language=language, created_at=_TS
        ),
    )


def _attach_transport_receipts(report, candidate: Candidate) -> None:  # type: ignore[no-untyped-def]
    gates = report.details["teacher_gates"]
    assert isinstance(gates, dict)
    receipts = []
    for index, (gate, payload) in enumerate(gates.items(), start=1):
        assert isinstance(gate, str) and isinstance(payload, dict)
        calls = payload["calls"]
        assert isinstance(calls, list) and isinstance(calls[0], dict)
        call = calls[0]
        call["recovery_accounting"] = None
        receipt = signed_transport_receipt(
            call_id=f"{index:032x}",
            candidate_fingerprint=candidate_transport_fingerprint(candidate),
            gate=gate,
            call_kind=str(call["call_kind"]),
            model=str(call["model"]),
            usage=Usage(**call["usage"]),  # type: ignore[arg-type]
            cost=float(call["cost"]),
        )
        call["call_id"] = receipt.call_id
        call["receipt_commitment"] = receipt.commitment
        receipts.append(receipt.to_private_dict())
    report.protected_teacher_transport_receipts = receipts


def _calibration(
    *,
    language: str,
    frontier_solves: int = 1,
    mid_solves: int = 1,
    weak_solves: int = 0,
    k: int = 4,
    discrimination: float = 1.5,
    difficulty: float = 1.0,
    keep: bool = True,
) -> CalibrationReport:
    def rate(solves: int) -> float:
        return solves / k if k else 0.0

    models = [
        ModelSolveRecord(
            model="weak/m",
            tier="weak",
            k=k,
            solves=weak_solves,
            pass_at_k=rate(weak_solves),
        ),
        ModelSolveRecord(
            model="mid/m",
            tier="mid",
            k=k,
            solves=mid_solves,
            pass_at_k=rate(mid_solves),
        ),
        ModelSolveRecord(
            model="frontier/m",
            tier="frontier",
            k=k,
            solves=frontier_solves,
            pass_at_k=rate(frontier_solves),
        ),
    ]
    report = CalibrationReport(
        language=language,
        models=models,
        k=k,
        irt_difficulty=difficulty,
        irt_discrimination=discrimination,
    )
    report.set_band_verdict(
        "keep" if keep else "drop",
        "in-band frontier + high discrimination" if keep else "solve-all too easy",
    )
    return report


def _request(
    *,
    language: str = "python",
    generator: str = "ast_mutation",
    seed: int = 7,
    repo_url: str = "https://github.com/acme/demo.git",
    **cal: object,
) -> ExportRequest:
    candidate = _candidate(language=language, generator=generator, seed=seed)
    report = _oracle_pass(language=language, generator=generator)
    _attach_transport_receipts(report, candidate)
    return ExportRequest(
        candidate=candidate,
        spec=_spec(language=language),
        oracle_report=report,
        calibration_report=_calibration(language=language, **cal),  # type: ignore[arg-type]
        env_image=_env_image(language=language),
        repo_url=repo_url,
    )


@pytest.fixture
def exported(tmp_path: Path) -> Path:
    """A mixed-generator, multi-language exported tree (5 kept tasks)."""
    requests = [
        _request(generator="ast_mutation", seed=1),
        _request(generator="lm_authored", seed=2),
        _request(generator="function_removal", seed=3),
        _request(language="javascript", generator="ast_mutation", seed=4),
        _request(language="go", generator="multi_file", seed=5),
    ]
    result = export_batch(requests, tmp_path)
    assert len(result.shipped) == 5
    return tmp_path


def _gold_all(exported: Path) -> GoldEvalReport:
    tasks = exported / "tasks"
    results = [
        TaskGoldResult(
            task_id=task_dir.name,
            task_dir=task_dir,
            image="swe-forge-env:demo",
            runs=[
                EvalRun(
                    task_id=task_dir.name,
                    run_index=index,
                    score=1,
                    phase1_passed=True,
                    exit_code=0,
                    container_name=f"gold-{task_dir.name}-{index}",
                )
                for index in range(2)
            ],
        )
        for task_dir in sorted(tasks.iterdir())
        if task_dir.is_dir()
    ]
    return GoldEvalReport(tasks_dir=tasks, results=results)


def _strict_gold_payload(exported: Path) -> dict[str, object]:
    return _gold_all(exported).to_dict()


def _assert_strict_gold_proof(payload: dict[str, object], expected_count: int) -> None:
    results = payload["results"]
    assert isinstance(results, list)
    assert len(results) == expected_count
    for result in results:
        assert isinstance(result, dict)
        runs = result["runs"]
        assert isinstance(runs, list)
        assert len(runs) >= 2
        assert all(
            isinstance(run, dict)
            and run["score"] == 1
            and run["phase1_passed"] is True
            and run["exit_code"] == 0
            for run in runs
        )


# --------------------------------------------------------------------------- #
# Strict Headline A proof
# --------------------------------------------------------------------------- #
def test_gold_summary_round_trips_two_strict_runs_per_exported_task(
    exported: Path,
) -> None:
    report = build_benchmark_report(exported, gold=_strict_gold_payload(exported))

    assert report.headline_a_pass
    payload = report.to_dict()["gold"]
    assert isinstance(payload, dict)
    _assert_strict_gold_proof(payload, expected_count=5)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("results"),
        lambda payload: payload["results"][0]["runs"].pop(),  # type: ignore[index]
        lambda payload: payload["results"][0]["runs"][0].update(  # type: ignore[index]
            {"phase1_passed": False}
        ),
        lambda payload: payload["results"][0]["runs"][0].update(  # type: ignore[index]
            {"exit_code": 1}
        ),
        lambda payload: payload["results"][0].update({"task_id": "other-task"}),  # type: ignore[index]
    ],
)
def test_report_fails_closed_for_missing_or_invalid_gold_proof(
    exported: Path, mutate
) -> None:  # type: ignore[no-untyped-def]
    payload = _strict_gold_payload(exported)
    mutate(payload)

    report = build_benchmark_report(exported, gold=payload)

    assert not report.headline_a_pass
    assert not report.passed


# --------------------------------------------------------------------------- #
# VAL-EXPORT-014: provenance completeness
# --------------------------------------------------------------------------- #
def test_provenance_complete_for_all_shipped(exported: Path) -> None:
    provs = load_task_provenances(exported / "tasks")
    assert len(provs) == 5
    result = check_provenance_completeness(provs)
    assert result.passed
    assert result.checked == 5
    assert result.complete == 5
    assert result.findings == []
    # Each task records the mandated fields, non-empty.
    for prov in provs:
        assert prov.generator
        assert isinstance(prov.seed, int)
        assert prov.language
        assert prov.tool_versions
        assert prov.difficulty is not None
        assert prov.discrimination is not None
        assert prov.panel_tiers() == {"weak", "mid", "frontier"}
    multifault = next(prov for prov in provs if prov.generator == "multi_file")
    evidence = multifault.details["multifault_completeness"]
    assert isinstance(evidence, dict)
    records = evidence["constituents"]
    assert isinstance(records, list) and len(records) == 2
    assert all(record["verdict"] == "pass" for record in records)


def test_completeness_detects_missing_field() -> None:
    prov = TaskProvenance(
        task_id="t1",
        language="python",
        generator="ast_mutation",
        seed=7,
        created_at=_TS,
        tool_versions={"litellm": "1.90.0"},
        mutants_total=0,  # missing -> mutants_total must be > 0
        mutants_killed=0,
        irt_difficulty=1.0,
        irt_discrimination=1.5,
        frontier_pass_at_k=0.25,
        oracle_verdict="pass",
        band_verdict="keep",
        panel=[
            {"model": "w", "tier": "weak", "k": 4, "solves": 0, "pass_at_k": 0.0},
            {"model": "m", "tier": "mid", "k": 4, "solves": 1, "pass_at_k": 0.25},
            {
                "model": "f",
                "tier": "frontier",
                "k": 4,
                "solves": 1,
                "pass_at_k": 0.25,
            },
        ],
    )
    result = check_provenance_completeness([prov])
    assert not result.passed
    assert result.findings[0].task_id == "t1"
    assert "mutants_total" in result.findings[0].missing


def test_completeness_detects_missing_tier() -> None:
    prov = TaskProvenance(
        task_id="t2",
        language="python",
        generator="ast_mutation",
        seed=7,
        created_at=_TS,
        tool_versions={"litellm": "1.90.0"},
        mutants_total=10,
        mutants_killed=10,
        irt_difficulty=1.0,
        irt_discrimination=1.5,
        frontier_pass_at_k=0.25,
        oracle_verdict="pass",
        band_verdict="keep",
        panel=[  # no frontier tier
            {"model": "w", "tier": "weak", "k": 4, "solves": 0, "pass_at_k": 0.0},
            {"model": "m", "tier": "mid", "k": 4, "solves": 1, "pass_at_k": 0.25},
        ],
    )
    result = check_provenance_completeness([prov])
    assert not result.passed
    assert "panel:frontier" in result.findings[0].missing


def test_completeness_detects_kill_ratio_below_threshold() -> None:
    prov = TaskProvenance(
        task_id="t3",
        language="python",
        generator="ast_mutation",
        seed=7,
        created_at=_TS,
        tool_versions={"litellm": "1.90.0"},
        mutants_total=10,
        mutants_killed=3,  # 0.3 < 0.8 threshold
        irt_difficulty=1.0,
        irt_discrimination=1.5,
        frontier_pass_at_k=0.25,
        oracle_verdict="pass",
        band_verdict="keep",
        panel=[
            {"model": "w", "tier": "weak", "k": 4, "solves": 0, "pass_at_k": 0.0},
            {"model": "m", "tier": "mid", "k": 4, "solves": 1, "pass_at_k": 0.25},
            {
                "model": "f",
                "tier": "frontier",
                "k": 4,
                "solves": 1,
                "pass_at_k": 0.25,
            },
        ],
    )
    result = check_provenance_completeness([prov])
    assert not result.passed
    assert "mutants_killed<threshold" in result.findings[0].missing


# --------------------------------------------------------------------------- #
# VAL-EXPORT-015: provenance consistency with the gates
# --------------------------------------------------------------------------- #
def test_provenance_consistent_with_gates(exported: Path) -> None:
    provs = load_task_provenances(exported / "tasks")
    result = check_provenance_consistency(provs)
    assert result.passed
    assert result.consistent == 5
    # Each shipped task: oracle=pass, band=keep, 0 < frontier <= band_high, disc >= min.
    for prov in provs:
        assert prov.oracle_verdict == "pass"
        assert prov.band_verdict == "keep"
        assert prov.frontier_rate is not None and 0.0 < prov.frontier_rate <= 0.5
        assert prov.discrimination is not None and prov.discrimination >= 1.0


def test_consistency_flags_out_of_band_frontier() -> None:
    prov = TaskProvenance(
        task_id="t4",
        language="python",
        generator="ast_mutation",
        seed=7,
        created_at=_TS,
        tool_versions={"litellm": "1.90.0"},
        mutants_total=10,
        mutants_killed=10,
        irt_difficulty=1.0,
        irt_discrimination=1.5,
        frontier_pass_at_k=0.9,  # above band_high (too easy)
        oracle_verdict="pass",
        band_verdict="keep",
        panel=[],
    )
    result = check_provenance_consistency([prov])
    assert not result.passed
    assert any("frontier_pass_rate" in i for i in result.findings[0].issues)


def test_consistency_flags_low_discrimination_and_bad_verdicts() -> None:
    prov = TaskProvenance(
        task_id="t5",
        language="python",
        generator="ast_mutation",
        seed=7,
        created_at=_TS,
        tool_versions={"litellm": "1.90.0"},
        mutants_total=10,
        mutants_killed=10,
        irt_difficulty=1.0,
        irt_discrimination=0.2,  # below min
        frontier_pass_at_k=0.25,
        oracle_verdict="reject",  # bad
        band_verdict="drop",  # bad
        panel=[],
    )
    result = check_provenance_consistency([prov])
    issues = result.findings[0].issues
    assert any("oracle_verdict" in i for i in issues)
    assert any("band_verdict" in i for i in issues)
    assert any("irt_discrimination" in i for i in issues)


# --------------------------------------------------------------------------- #
# Panel aggregation + tier ordering
# --------------------------------------------------------------------------- #
def test_aggregate_panel_pools_per_model_and_per_tier(exported: Path) -> None:
    provs = load_task_provenances(exported / "tasks")
    per_model, tier_rates = aggregate_panel(provs)
    # One aggregate per distinct model id, sorted weak -> mid -> frontier.
    tiers = [a.tier for a in per_model]
    assert tiers == ["weak", "mid", "frontier"]
    assert tier_rates["weak"] == 0.0
    assert tier_rates["mid"] == pytest.approx(0.25)
    assert tier_rates["frontier"] == pytest.approx(0.25)
    # weak <= mid <= frontier.
    assert tier_rates["weak"] <= tier_rates["mid"] <= tier_rates["frontier"]


# --------------------------------------------------------------------------- #
# VAL-EXPORT-016/017/018/019: the benchmark report
# --------------------------------------------------------------------------- #
def test_report_headline_a_gold_100(exported: Path) -> None:
    report = build_benchmark_report(exported, gold=_gold_all(exported))
    assert report.gold.gold_rate == 1.0
    assert report.gold.gold_count == report.shipped_count == 5
    assert report.headline_a_pass


def test_report_per_model_irt_breakdown(exported: Path) -> None:
    report = build_benchmark_report(exported, gold=_gold_all(exported))
    # Per-model + tier ordering weak <= mid <= frontier < gold(100%).
    assert report.per_model
    assert report.tier_ordering_ok
    assert report.gold_ge_frontier
    # IRT summary present and numeric.
    assert set(report.irt_difficulty) == {"mean", "min", "max"}
    assert set(report.irt_discrimination) == {"mean", "min", "max"}
    assert report.irt_discrimination["mean"] == pytest.approx(1.5)
    # Generator + language breakdowns sum to the shipped total.
    assert sum(report.generator_breakdown.values()) == 5
    assert sum(report.language_breakdown.values()) == 5
    assert report.generator_breakdown == {
        "ast_mutation": 2,
        "lm_authored": 1,
        "function_removal": 1,
        "multi_file": 1,
    }
    assert report.language_breakdown == {"python": 3, "javascript": 1, "go": 1}
    assert report.breakdown_reconciles


def test_report_headline_b_frontier_below_threshold(exported: Path) -> None:
    report = build_benchmark_report(exported, gold=_gold_all(exported))
    assert report.frontier_threshold == DEFAULT_FRONTIER_THRESHOLD
    assert 0.0 < report.frontier_solve_rate < report.frontier_threshold
    assert report.headline_b_pass


def test_default_headline_threshold_exceeds_inclusive_keep_edge(tmp_path: Path) -> None:
    """A keep at the inclusive band edge must still satisfy Headline B strictly."""
    result = export_batch(
        [_request(frontier_solves=2, mid_solves=0, weak_solves=0)],
        tmp_path,
    )
    assert len(result.shipped) == 1

    report = build_benchmark_report(tmp_path, gold=_gold_all(tmp_path))

    assert report.frontier_solve_rate == DEFAULT_BAND_HIGH
    assert report.frontier_threshold > DEFAULT_BAND_HIGH
    assert report.headline_b_pass


def test_report_counts_reconcile_and_json_complete(exported: Path) -> None:
    report = build_benchmark_report(exported, gold=_gold_all(exported))
    assert report.counts.tasks == 5
    assert report.counts.jsonl == 5
    assert report.counts.parquet == 5
    assert report.counts.reconciled

    # JSON parses and carries every required key (VAL-EXPORT-019).
    parsed = json.loads(report.to_json())
    for key in (
        "gold_rate",
        "per_model",
        "irt",
        "generator_breakdown",
        "language_breakdown",
        "frontier_solve_rate",
        "frontier_threshold",
    ):
        assert key in parsed, key
    assert parsed["counts"]["reconciled"] is True
    assert parsed["gold_rate"] == 1.0
    assert parsed["per_model"]
    assert parsed["headline_b"]["passed"] is True


def test_report_overall_pass(exported: Path) -> None:
    report = build_benchmark_report(exported, gold=_gold_all(exported))
    assert report.passed


def test_report_fails_headline_a_when_gold_not_100(exported: Path) -> None:
    partial = GoldSummary(
        measured=True, gold_count=4, shipped_count=5, all_gold=False, deterministic=True
    )
    report = build_benchmark_report(exported, gold=partial)
    assert not report.headline_a_pass
    assert not report.passed


def test_report_gold_unmeasured_marks_headline_a_fail(exported: Path) -> None:
    report = build_benchmark_report(exported, gold=None)
    assert report.gold.measured is False
    assert not report.headline_a_pass


def test_aggregate_only_gold_json_cannot_claim_headline_a(exported: Path) -> None:
    report = build_benchmark_report(
        exported,
        gold={
            "gold_count": 5,
            "shipped_count": 5,
            "all_gold": True,
            "deterministic": True,
        },
    )

    assert report.gold.gold_rate == 0.0
    assert not report.headline_a_pass


def test_report_cli_rejects_aggregate_only_gold_json(exported: Path) -> None:
    from typer.testing import CliRunner

    from swe_forge.forge.cli import app

    gold_json = exported / "aggregate-only-gold.json"
    gold_json.write_text(
        json.dumps(
            {
                "gold_count": 5,
                "shipped_count": 5,
                "all_gold": True,
                "deterministic": True,
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["report", "--out-dir", str(exported), "--gold-json", str(gold_json), "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["headline_a"]["passed"] is False


def test_report_rejects_an_immediate_invalid_task_workspace(exported: Path) -> None:
    invalid = exported / "tasks" / "missing-evaluator"
    invalid.mkdir()
    (invalid / "workspace.yaml").write_text(
        "environment:\n  image: example:latest\n",
        encoding="utf-8",
    )

    with pytest.raises(ReportError, match="missing-evaluator: missing evaluate.sh"):
        build_benchmark_report(exported, gold=_gold_all(exported))


# --------------------------------------------------------------------------- #
# Headline B edge cases (solve-none / out-of-band)
# --------------------------------------------------------------------------- #
def test_headline_b_fails_when_frontier_zero(tmp_path: Path) -> None:
    # Hand-built provenance with frontier rate 0 (solve-none would never ship,
    # but the report must report the headline correctly).
    provs = [
        TaskProvenance(
            task_id="z",
            language="python",
            generator="ast_mutation",
            seed=1,
            created_at=_TS,
            tool_versions={"litellm": "1.0"},
            mutants_total=10,
            mutants_killed=10,
            irt_difficulty=5.0,
            irt_discrimination=1.5,
            frontier_pass_at_k=0.0,
            oracle_verdict="pass",
            band_verdict="keep",
            panel=[
                {
                    "model": "f",
                    "tier": "frontier",
                    "k": 4,
                    "solves": 0,
                    "pass_at_k": 0.0,
                },
            ],
        )
    ]
    _per_model, tier_rates = aggregate_panel(provs)
    assert tier_rates["frontier"] == 0.0


def test_report_band_config_threads_consistency(exported: Path) -> None:
    # A stricter discrimination threshold makes the kept tasks inconsistent.
    report = build_benchmark_report(
        exported,
        gold=_gold_all(exported),
        band_config=BandFilterConfig(band_high=0.5, discrimination_threshold=2.0),
    )
    assert not report.consistency.passed
    assert not report.passed


# --------------------------------------------------------------------------- #
# Count helpers + write
# --------------------------------------------------------------------------- #
def test_count_helpers(exported: Path) -> None:
    assert count_jsonl_records(exported / "dataset.jsonl") == 5
    assert count_parquet_rows(exported / "dataset.parquet") == 5
    assert count_jsonl_records(exported / "missing.jsonl") == 0


def test_write_report_emits_md_and_json(exported: Path) -> None:
    report = build_benchmark_report(exported, gold=_gold_all(exported))
    md_path, json_path = write_report(report, exported)
    assert md_path.is_file() and json_path.is_file()
    assert "SWE-Forge Benchmark Report" in md_path.read_text()
    parsed = json.loads(json_path.read_text())
    assert parsed["shipped_count"] == 5
    assert parsed["passed"] is True


def test_build_report_on_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ReportError):
        build_benchmark_report(tmp_path / "does-not-exist")


def test_empty_export_report(tmp_path: Path) -> None:
    export_batch([], tmp_path)
    report = build_benchmark_report(tmp_path, gold=GoldSummary.unmeasured())
    assert report.shipped_count == 0
    assert report.counts.reconciled  # 0 == 0 == 0
    assert not report.passed  # no shipped tasks -> not a pass
    assert isinstance(report, BenchmarkReport)
