"""Unit tests for synthetic_grounded producer (VAL-PROD-002/003).

Offline fakes only (no Docker unless marker integration). Covers:
- multi-file multi-fault inverse gold
- function-removal multi-file floor
- source_track=synthetic_grounded labeling
- non-empty problem_statement
- stub + fake certified oracle path
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_factory.oracle import codes as C
from swe_factory.oracle.docker_run import FakeOracleRunner, OracleDockerError, ScriptedSuite
from swe_factory.oracle.gates import count_files_in_patch, run_certified_gates
from swe_factory.producers.synth import (
    MUTATION_FUNCTION_REMOVAL,
    MUTATION_MULTI_FAULT,
    SynthError,
    SynthProducer,
    build_problem_statement,
    mutate_python_function_removal,
    mutate_python_multi_fault,
    produce_from_green_fixture,
)
from swe_factory.schema import SourceTrack
from swe_factory.sources.allowlist import (
    ALLOWLIST,
    TINY_GREEN,
    allowlist_summary,
    local_offline_seeds,
)


def test_allowlist_prefers_modular_languages() -> None:
    langs = {s.language for s in ALLOWLIST}
    assert "python" in langs
    assert "javascript" in langs
    assert "go" in langs
    assert all(s.modular for s in ALLOWLIST)
    local = local_offline_seeds()
    assert any(s.seed_id == TINY_GREEN.seed_id for s in local)
    rows = allowlist_summary()
    assert any(r["local_available"] for r in rows)


def test_python_multi_fault_mutates_return() -> None:
    src = "def add(a: int, b: int) -> int:\n    return a + b\n"
    out, symbols = mutate_python_multi_fault(src)
    assert "add" in symbols
    assert out != src
    assert "return a - b" in out or "return a" in out


def test_python_function_removal_stubs_body() -> None:
    src = "def reverse_words(text: str) -> str:\n    return text[::-1]\n"
    out, symbols = mutate_python_function_removal(src)
    assert symbols == ["reverse_words"]
    assert "NotImplementedError" in out


def test_produce_multi_fault_labeled_multi_file(tmp_path: Path) -> None:
    producer = SynthProducer(work_root=tmp_path / "work")
    candidate = producer.produce(
        TINY_GREEN,
        mutation_kind=MUTATION_MULTI_FAULT,
        instance_suffix="unitmf",
        run_stub_oracle=True,
    )
    task = candidate.task
    assert task.source_track == SourceTrack.SYNTHETIC_GROUNDED
    assert candidate.source_track == "synthetic_grounded"
    assert task.problem_statement.strip()
    assert len(candidate.gold_files) >= 2
    assert len(count_files_in_patch(task.gold_patch)) >= 2
    assert candidate.inverse_meta["gold_is_inverse"] is True
    assert candidate.inverse_meta["derivation"] == "inverse_of_synthetic_mutation"
    assert candidate.provider_calls == 0
    assert candidate.gates is not None and candidate.gates.passed
    assert C.G4_MULTI_FILE_OK in (candidate.gates.reason_codes if candidate.gates else ())
    # Gold lives only under producer case dir, not as leaky agent answer keyed free-float
    assert candidate.broken_workspace.is_dir()
    math_txt = (candidate.broken_workspace / "demo_pkg" / "math_ops.py").read_text(encoding="utf-8")
    text_txt = (candidate.broken_workspace / "demo_pkg" / "text_ops.py").read_text(encoding="utf-8")
    # broken must differ from green for multi_fault
    green_math = (TINY_GREEN.resolve_local_path() or Path()) / "demo_pkg" / "math_ops.py"
    assert green_math.is_file()
    assert math_txt != green_math.read_text(encoding="utf-8") or text_txt != (
        green_math.parent / "text_ops.py"
    ).read_text(encoding="utf-8")


def test_produce_function_removal_multi_file(tmp_path: Path) -> None:
    candidate = SynthProducer(work_root=tmp_path / "work").produce(
        TINY_GREEN,
        mutation_kind=MUTATION_FUNCTION_REMOVAL,
        instance_suffix="unitfr",
        run_stub_oracle=True,
    )
    assert candidate.task.source_track == SourceTrack.SYNTHETIC_GROUNDED
    assert len(candidate.gold_files) >= 2
    assert "NotImplementedError" in (
        (candidate.broken_workspace / "demo_pkg" / "math_ops.py").read_text(encoding="utf-8")
        + (candidate.broken_workspace / "demo_pkg" / "text_ops.py").read_text(encoding="utf-8")
    )
    assert candidate.task.problem_statement
    assert "function" in candidate.task.problem_statement.lower()


def test_problem_statement_builder_nonempty() -> None:
    from swe_factory.producers.synth import MutationTarget

    prompt = build_problem_statement(
        mutation_kind=MUTATION_MULTI_FAULT,
        targets=(
            MutationTarget("a.py", MUTATION_MULTI_FAULT, "foo"),
            MutationTarget("b.py", MUTATION_MULTI_FAULT, "bar"),
        ),
        repo="owner/repo",
        language="python",
    )
    assert prompt.strip()
    assert "a.py" in prompt and "b.py" in prompt


def test_produce_from_green_fixture_helper(tmp_path: Path) -> None:
    candidate = produce_from_green_fixture(
        mutation_kind=MUTATION_MULTI_FAULT,
        work_root=tmp_path / "fx",
    )
    assert candidate.task.source_track == SourceTrack.SYNTHETIC_GROUNDED
    assert candidate.task.gate_proof is not None
    assert candidate.task.gate_proof.get("inverse_meta", {}).get("gold_is_inverse")


def test_produce_and_certify_with_fake_oracle(tmp_path: Path) -> None:
    runner = FakeOracleRunner(
        broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
        gold_runs=[
            ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
            ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
        ],
        null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
    )
    audit = tmp_path / "gate_audit.jsonl"
    candidate = SynthProducer(work_root=tmp_path / "work").produce_and_certify(
        TINY_GREEN,
        runner=runner,
        mutation_kind=MUTATION_MULTI_FAULT,
        instance_suffix="certfake",
        audit_out=audit,
    )
    assert candidate.gates is not None
    assert candidate.gates.passed is True
    assert C.ORACLE_PASS in candidate.gates.reason_codes
    assert C.G3_NULL_NOT_RESOLVE in candidate.gates.reason_codes
    assert candidate.task.source_track == SourceTrack.SYNTHETIC_GROUNDED
    row = json.loads(audit.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert row["disposition"] == "accept"
    assert row["source_track"] == "synthetic_grounded"


def test_g3_null_eval_error_distinct_from_resolves(tmp_path: Path) -> None:
    """Prefer G3_NULL_EVAL_ERROR when null evaluation raises (not G3_NULL_RESOLVES)."""

    class _FailNullRunner(FakeOracleRunner):
        def run_with_patch(  # type: ignore[override]
            self,
            *,
            workspace: Path,
            patch: str,
            fail_to_pass: object,
            pass_to_pass: object = (),
            phase: str = "gold",
        ):
            if not patch.strip():
                raise OracleDockerError("container vanished during null eval")
            return super().run_with_patch(
                workspace=workspace,
                patch=patch,
                fail_to_pass=fail_to_pass,  # type: ignore[arg-type]
                pass_to_pass=pass_to_pass,  # type: ignore[arg-type]
                phase=phase,
            )

    runner = _FailNullRunner(
        broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
        gold_runs=[
            ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
            ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
        ],
        null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
    )
    ws = tmp_path / "repo"
    ws.mkdir()
    gold = (
        "diff --git a/demo_pkg/math_ops.py b/demo_pkg/math_ops.py\n"
        "--- a/demo_pkg/math_ops.py\n"
        "+++ b/demo_pkg/math_ops.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def add(a: int, b: int) -> int:\n"
        "-    return a - b\n"
        "+    return a + b\n"
        "\n"
        "diff --git a/demo_pkg/text_ops.py b/demo_pkg/text_ops.py\n"
        "--- a/demo_pkg/text_ops.py\n"
        "+++ b/demo_pkg/text_ops.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def reverse_words(text: str) -> str:\n"
        "-    return text\n"
        '+    return " ".join(reversed(text.split()))\n'
    )
    result = run_certified_gates(
        gold_patch=gold,
        fail_to_pass=["pytest f2p"],
        pass_to_pass=["pytest p2p"],
        problem_statement="fix multi-file",
        image_digest="sha256:x",
        workspace=ws,
        runner=runner,
        check_leak=False,
    )
    assert result.passed is False
    assert C.G3_NULL_EVAL_ERROR in result.reason_codes
    assert C.G3_NULL_RESOLVES not in result.reason_codes
    assert result.details.get("g3_kind") == "eval_error"


def test_stub_reject_surfaces_synth_error(tmp_path: Path) -> None:
    """Empty problem_statement path is blocked before export."""
    producer = SynthProducer(work_root=tmp_path / "work")
    with pytest.raises(SynthError, match="problem_statement|stub oracle"):
        producer.produce(
            TINY_GREEN,
            mutation_kind=MUTATION_MULTI_FAULT,
            problem_statement="   ",
            run_stub_oracle=True,
        )


@pytest.mark.integration
def test_synth_docker_smoke_optional(tmp_path: Path) -> None:
    """Optional Docker smoke: synth candidate + certified oracle (no panel)."""
    docker = pytest.importorskip("subprocess")
    check = docker.run(["docker", "info"], capture_output=True, check=False)
    if check.returncode != 0:
        pytest.skip("docker daemon unavailable")

    from swe_factory.envbuild.builder import DockerCLI, remove_leftover_sdf_containers
    from swe_factory.oracle.docker_run import OracleDockerRunner

    runner = OracleDockerRunner(
        docker=DockerCLI(),
        base_image="python:3.12-slim",
        install_commands=["pip install -q pytest"],
        command_timeout=180.0,
    )
    try:
        candidate = SynthProducer(work_root=tmp_path / "work").produce_and_certify(
            TINY_GREEN,
            runner=runner,
            mutation_kind=MUTATION_MULTI_FAULT,
            instance_suffix="dockersmoke",
            dual_runs=2,
        )
    finally:
        remove_leftover_sdf_containers()

    assert candidate.gates is not None and candidate.gates.passed
    assert candidate.task.source_track == SourceTrack.SYNTHETIC_GROUNDED
    assert len(candidate.gold_files) >= 2
