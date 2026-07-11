"""Fail-closed, secret-safe teacher evidence for differential oracle gates."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace
import pytest

from swe_forge.forge.adapters import PythonAdapter
from swe_forge.forge.models import (
    Candidate,
    CandidateTarget,
    EnvImage,
    FinalMutationEvidence,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.oracle.alt_correct import (
    REASON_ALT_CORRECT_NO_EXECUTABLE,
    AltCorrectGenerationContext,
    AltImpl,
    AltImplFile,
    AltScore,
    assess_alt_correct,
    run_alt_correct_gate,
)
from swe_forge.forge.oracle.alt_correct_synth import TeacherAltCorrectGenerator
from swe_forge.forge.oracle.differential import (
    DifferentialSynthesisContext,
    REASON_DIFFERENTIAL_NO_EXECUTABLE,
    Variant,
    VariantFile,
    VariantGenerationContext,
    VariantScore,
    assess_differential,
    run_differential_gate,
)
from swe_forge.forge.oracle.differential_synth import (
    DifferentialKillSynthesizer,
    TeacherVariantGenerator,
)
from swe_forge.forge.oracle.mutation import final_suite_fingerprint
from swe_forge.forge.oracle.pipeline import (
    ExportRefusedError,
    ensure_oracle_exportable,
    verify_pass_consistency,
)
from swe_forge.forge.oracle.teacher_evidence import (
    aggregate_teacher_gate_usage,
    teacher_gate_evidence_issues,
    teacher_gate_failure_reason,
)
from swe_forge.forge.teacher import (
    LLMResult,
    TeacherClient,
    TransportReceipt,
    Usage,
    candidate_transport_fingerprint,
)
from tests.test_forge.receipt_helpers import (
    protected_alt_correct_audit,
    protected_alt_correct_summary,
    signed_transport_receipt,
)


class _DifferentialRunner:
    language = "python"
    discardable_tests = ()

    async def score_gold(self, _extra, *, exclude=()) -> VariantScore:  # type: ignore[no-untyped-def]
        assert not exclude
        return VariantScore(f2p_passed=True, p2p_passed=True)

    async def score_variant(self, _variant, _extra, *, exclude=()) -> VariantScore:  # type: ignore[no-untyped-def]
        assert not exclude
        return VariantScore(
            f2p_passed=False,
            p2p_passed=True,
            failing_test_ids=("hidden",),
        )

    async def read_sources(self) -> dict[str, str]:
        return {}


class _AltRunner:
    language = "python"

    async def score_gold(self, exclude=()) -> AltScore:  # type: ignore[no-untyped-def]
        assert not exclude
        return AltScore(f2p_passed=True, p2p_passed=True, public_suite_passed=True)

    async def score_alt(self, _alt, exclude=()) -> AltScore:  # type: ignore[no-untyped-def]
        assert not exclude
        return AltScore(f2p_passed=True, p2p_passed=True, public_suite_passed=True)

    async def read_sources(self) -> dict[str, str]:
        return {}


class _WrappedDifferentialRunner(_DifferentialRunner):
    """Docker-runner replacement that lets wrapper tests stay offline."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.score_calls = 0

    async def score_variant(self, *_args: object, **_kwargs: object) -> VariantScore:
        self.score_calls += 1
        return await super().score_variant(*_args, **_kwargs)

    async def read_sources(self) -> dict[str, str]:
        return {"src/m.py": "def f():\n    return 1\n"}


class _WrappedAltRunner(_AltRunner):
    """Docker-runner replacement that lets wrapper tests stay offline."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.score_calls = 0

    async def score_alt(self, *_args: object, **_kwargs: object) -> AltScore:
        self.score_calls += 1
        return await super().score_alt(*_args, **_kwargs)

    async def read_sources(self) -> dict[str, str]:
        return {"src/m.py": "def f():\n    return 1\n"}


class _Teacher:
    model = "anthropic/test-model"

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self._text = text
        self._error = error

    async def complete_text(self, *_args, **_kwargs) -> LLMResult:  # type: ignore[no-untyped-def]
        if self._error is not None:
            raise self._error
        return LLMResult(
            text=self._text,
            usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            cost=0.0125,
            finish_reason="stop",
        )


class _TeacherClientSubclass(TeacherClient):
    """A usable parser double that must not attest a real transport call."""

    def __init__(self, text: str) -> None:
        super().__init__(
            base_url="https://teacher.test",
            api_key="sk-subclass-test",
            model="anthropic/test-model",
        )
        self._text = text

    async def complete_text(self, *_args: object, **_kwargs: object) -> LLMResult:
        return LLMResult(
            text=self._text,
            usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            cost=0.0125,
            finish_reason="stop",
        )


def _concrete_teacher_client(
    *,
    base_url: str = "https://teacher.test",
    api_key: str = "sk-concrete-test",
) -> TeacherClient:
    return TeacherClient(
        base_url=base_url,
        api_key=api_key,
        model="anthropic/test-model",
    )


def _teacher_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=7,
            completion_tokens=3,
            total_tokens=10,
        ),
        _hidden_params={"response_cost": 0.0125},
    )


def _candidate() -> Candidate:
    return Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=("src/m.py",), symbols=("f",)),
        mutation_patch="--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1 @@\n-return 1\n+return 2\n",
        oracle_patch="--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1 @@\n-return 2\n+return 1\n",
        difficulty_hint="medium",
        provenance=Provenance(generator="ast_mutation", seed=1, language="python"),
    )


def _variant() -> Variant:
    return Variant(
        variant_id="wrong",
        files=(VariantFile(path="src/m.py", content="def f():\n    return 2\n"),),
    )


def _alt() -> AltImpl:
    return AltImpl(
        impl_id="correct",
        files=(AltImplFile(path="src/m.py", content="def f():\n    return 1 + 0\n"),),
    )


def _variant_context() -> VariantGenerationContext:
    return VariantGenerationContext(
        candidate=_candidate(),
        adapter=PythonAdapter(),
        gold_sources={"src/m.py": "def f():\n    return 1\n"},
        num_variants=2,
    )


def _alt_context() -> AltCorrectGenerationContext:
    return AltCorrectGenerationContext(
        candidate=_candidate(),
        adapter=PythonAdapter(),
        gold_sources={"src/m.py": "def f():\n    return 1\n"},
        num_alternatives=2,
    )


def _env_image() -> EnvImage:
    return EnvImage(
        repo_id="teacher-gate-test",
        language="python",
        image_tag="swe-forge-env-teacher-gate-test:abc123",
        base_image="python:3.12-slim",
        commit="0" * 40,
        workspace_dir="/workspace/repo",
        install_commands=["pip install -e ."],
        baseline_test_command="python -m pytest",
        original_public_test_command="python -m pytest",
        baseline_green=True,
        baseline_exit_code=0,
    )


def _prior_report() -> OracleReport:
    return OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="pass",
        fail_to_pass=["python -m pytest tests/hidden.py"],
        pass_to_pass=["python -m pytest"],
        test_files=[
            OracleTestFile(path="tests/hidden.py", content="def test_x(): pass\n")
        ],
        flakiness_runs=3,
        mutants_total=2,
        mutants_killed=2,
        provenance=Provenance(generator="ast_mutation", seed=1, language="python"),
    )


async def test_differential_zero_executable_variants_rejects() -> None:
    outcome = await assess_differential(_DifferentialRunner(), [])

    assert outcome.verdict == "reject"
    assert outcome.differential_pass is False
    assert outcome.reasons[0].startswith(REASON_DIFFERENTIAL_NO_EXECUTABLE)


async def test_alt_correct_zero_executable_alternatives_rejects() -> None:
    outcome = await assess_alt_correct(_AltRunner(), [])

    assert outcome.verdict == "reject"
    assert outcome.alt_correct_accepted is False
    assert outcome.reasons[0].startswith(REASON_ALT_CORRECT_NO_EXECUTABLE)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "empty"),
        ("I cannot provide code.", "unparseable"),
        ("```python\ndef f():\n    return 1\n```", "identical"),
        ("```python\nreturn 2\n```", "invalid"),
    ],
)
async def test_teacher_variant_records_distinct_empty_parse_and_identical_evidence(
    text: str, expected: str
) -> None:
    generator = TeacherVariantGenerator(client=_Teacher(text))  # type: ignore[arg-type]

    assert await generator(_variant_context()) == []
    evidence = generator.last_call
    assert evidence is not None
    assert evidence.response_kind == expected
    assert evidence.status == "success"
    assert evidence.model == "anthropic/test-model"
    assert evidence.usage.total_tokens == 10
    assert evidence.cost == 0.0125
    assert evidence.finish_reason == "stop"
    assert expected in teacher_gate_failure_reason("differential", [evidence])


async def test_teacher_variant_error_is_secret_free_and_attributable() -> None:
    generator = TeacherVariantGenerator(
        client=_Teacher(error=RuntimeError("credential sk-secret-value rejected"))  # type: ignore[arg-type]
    )

    assert await generator(_variant_context()) == []
    evidence = generator.last_call
    assert evidence is not None
    assert evidence.status == "error"
    assert evidence.error_type == "RuntimeError"
    assert "teacher_call_failed" in teacher_gate_failure_reason(
        "differential", [evidence]
    )
    assert "sk-secret-value" not in json.dumps(evidence.to_dict())


async def test_teacher_alt_records_invalid_and_successful_proposal_evidence() -> None:
    generator = TeacherAltCorrectGenerator(
        client=_Teacher(
            "```\n\n```\n```python\ndef f():\n    value = 1\n    return value\n```\n"
        )  # type: ignore[arg-type]
    )

    alternatives = await generator(_alt_context())
    assert len(alternatives) == 1
    evidence = generator.last_call
    assert evidence is not None
    assert evidence.invalid_proposals == 1
    assert evidence.parsed_proposals == 1


@pytest.mark.parametrize(
    ("generator_cls", "context", "text"),
    [
        (
            TeacherVariantGenerator,
            _variant_context,
            "```python\ndef f():\n    return 2\n```",
        ),
        (
            TeacherAltCorrectGenerator,
            _alt_context,
            "```python\ndef f():\n    return 1 + 0\n```",
        ),
    ],
)
async def test_injected_teacher_doubles_remain_usable_but_non_authoritative(
    generator_cls: type[TeacherVariantGenerator] | type[TeacherAltCorrectGenerator],
    context: object,
    text: str,
) -> None:
    generator = generator_cls(client=_Teacher(text))  # type: ignore[arg-type, call-arg]

    assert await generator(context())  # type: ignore[misc]
    evidence = generator.last_call
    assert evidence is not None
    assert evidence.real_teacher is False
    assert evidence.usage.total_tokens == 10
    assert evidence.cost == 0.0125


@pytest.mark.parametrize(
    ("generator_cls", "context", "text"),
    [
        (
            TeacherVariantGenerator,
            _variant_context,
            "```python\ndef f():\n    return 2\n```",
        ),
        (
            TeacherAltCorrectGenerator,
            _alt_context,
            "```python\ndef f():\n    return 1 + 0\n```",
        ),
    ],
)
async def test_teacher_client_subclasses_are_non_authoritative(
    generator_cls: type[TeacherVariantGenerator] | type[TeacherAltCorrectGenerator],
    context: object,
    text: str,
) -> None:
    generator = generator_cls(  # type: ignore[arg-type, call-arg]
        client=_TeacherClientSubclass(text)
    )

    assert await generator(context())  # type: ignore[misc]
    evidence = generator.last_call
    assert evidence is not None
    assert evidence.real_teacher is False


async def test_injected_differential_strengthener_is_non_authoritative() -> None:
    synthesizer = DifferentialKillSynthesizer(
        client=_Teacher(
            "```python\ndef test_f():\n    from m import f\n    assert f() == 1\n```"
        )  # type: ignore[arg-type]
    )

    proposals = await synthesizer(
        DifferentialSynthesisContext(
            candidate=_candidate(),
            adapter=PythonAdapter(),
            gold_sources={"src/m.py": "def f():\n    return 1\n"},
            survivors=(_variant(),),
            round_index=1,
        )
    )

    assert proposals
    assert synthesizer.last_call is not None
    assert synthesizer.last_call.real_teacher is False


@pytest.mark.parametrize(
    ("generator_cls", "context", "text", "expected_kind"),
    [
        (
            TeacherVariantGenerator,
            _variant_context,
            (
                "```python\ndef f():\n    return 2\n```\n"
                "```python\ndef f():\n    return 2\n```\n"
                "```python\ndef f():\n    return 3\n```\n"
            ),
            "differential",
        ),
        (
            TeacherAltCorrectGenerator,
            _alt_context,
            (
                "```python\ndef f():\n    return 1 + 0\n```\n"
                "```python\ndef f():\n    return 1 + 0\n```\n"
                "```python\ndef f():\n    return 0 + 1\n```\n"
            ),
            "alt_correct",
        ),
    ],
)
async def test_teacher_generators_account_for_duplicate_and_overflow_proposals(
    generator_cls: type[TeacherVariantGenerator] | type[TeacherAltCorrectGenerator],
    context: object,
    text: str,
    expected_kind: str,
) -> None:
    generator = generator_cls(client=_Teacher(text))  # type: ignore[arg-type, call-arg]
    proposals = await generator(
        replace(context(), num_variants=1)
        if expected_kind == "differential"
        else replace(context(), num_alternatives=1)
    )  # type: ignore[misc]

    assert len(proposals) == 1
    evidence = generator.last_call
    assert evidence is not None
    assert evidence.received_proposals == 3
    assert evidence.invalid_proposals == 0
    assert evidence.parsed_proposals == 3
    assert evidence.identical_proposals == 0
    assert evidence.discarded_proposals == 2
    assert evidence.executable_proposals == 1
    assert evidence.received_proposals == (
        evidence.invalid_proposals + evidence.parsed_proposals
    )
    assert evidence.parsed_proposals == (
        evidence.identical_proposals
        + evidence.discarded_proposals
        + evidence.executable_proposals
    )


@pytest.mark.parametrize(
    ("generator_cls", "context", "expected_gate"),
    [
        (TeacherVariantGenerator, _variant_context, "differential"),
        (TeacherAltCorrectGenerator, _alt_context, "alt_correct"),
    ],
)
async def test_teacher_generators_record_truncated_valid_proposals_as_discarded(
    generator_cls: type[TeacherVariantGenerator] | type[TeacherAltCorrectGenerator],
    context: object,
    expected_gate: str,
) -> None:
    generator = generator_cls(  # type: ignore[arg-type, call-arg]
        client=_Teacher("```python\ndef f():\n    return 1 + 0\n")
    )

    assert await generator(context()) == []
    evidence = generator.last_call
    assert evidence is not None
    assert evidence.received_proposals == 1
    assert evidence.invalid_proposals == 0
    assert evidence.parsed_proposals == 1
    assert evidence.identical_proposals == 0
    assert evidence.discarded_proposals == 1
    assert evidence.executable_proposals == 0
    assert teacher_gate_failure_reason(expected_gate, [evidence]) == (
        f"{expected_gate}_teacher_discarded_proposals"
    )


@pytest.mark.parametrize(
    ("generator_cls", "context"),
    [
        (TeacherVariantGenerator, _variant_context),
        (TeacherAltCorrectGenerator, _alt_context),
    ],
)
async def test_teacher_generators_reconcile_source_unavailable_without_inventing_invalid_proposals(
    generator_cls: type[TeacherVariantGenerator] | type[TeacherAltCorrectGenerator],
    context: object,
) -> None:
    generator = generator_cls(  # type: ignore[arg-type, call-arg]
        client=_Teacher("```python\ndef f():\n    return 2\n```")
    )

    assert await generator(replace(context(), gold_sources={})) == []  # type: ignore[misc]
    evidence = generator.last_call
    assert evidence is not None
    assert evidence.real_teacher is False
    assert evidence.received_proposals == 0
    assert evidence.invalid_proposals == 0
    assert evidence.parsed_proposals == 0


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("received_proposals", 2, "received"),
        ("parsed_proposals", 2, "parsed"),
        ("execution_completed", 0, "execution"),
        ("execution_errors", 1, "execution"),
    ],
)
def test_teacher_gate_evidence_requires_reconciled_proposal_and_execution_counts(
    field: str, value: object, expected: str
) -> None:
    evidence = _positive_gate_evidence("differential")
    evidence["calls"][0][field] = value  # type: ignore[index]
    issues = teacher_gate_evidence_issues(
        {"teacher_gates": {"differential": evidence}},
        gates=("differential",),
    )

    assert len(issues) == 1
    assert expected in issues[0]


async def test_standalone_differential_rejects_passing_injected_proposals_without_teacher_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import swe_forge.forge.oracle.differential as differential_module

    class _InjectedGenerator:
        async def __call__(self, _ctx: VariantGenerationContext) -> list[Variant]:
            return [_variant()]

    monkeypatch.setattr(
        differential_module, "DockerDifferentialRunner", _WrappedDifferentialRunner
    )
    report = await run_differential_gate(
        _candidate(),
        _env_image(),
        _prior_report(),
        variant_generator=_InjectedGenerator(),
        adapter=PythonAdapter(),
    )

    assert report.verdict == "reject"
    assert report.differential_pass is False
    assert report.reasons == ["differential_no_real_teacher_proposal"]


async def test_standalone_alt_correct_rejects_passing_injected_proposals_without_teacher_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import swe_forge.forge.oracle.alt_correct as alt_correct_module

    class _InjectedGenerator:
        async def __call__(self, _ctx: AltCorrectGenerationContext) -> list[AltImpl]:
            return [_alt()]

    monkeypatch.setattr(alt_correct_module, "DockerAltCorrectRunner", _WrappedAltRunner)
    report = await run_alt_correct_gate(
        _candidate(),
        _env_image(),
        _prior_report(),
        alt_generator=_InjectedGenerator(),
        adapter=PythonAdapter(),
    )

    assert report.verdict == "reject"
    assert report.alt_correct_accepted is False
    assert report.reasons == ["alt_correct_no_real_teacher_proposal"]


@pytest.mark.parametrize(
    ("generator_cls", "runner_cls", "gate"),
    [
        (TeacherVariantGenerator, "differential", "differential"),
        (TeacherAltCorrectGenerator, "alt_correct", "alt_correct"),
    ],
)
async def test_standalone_gates_accept_positive_real_teacher_evidence(
    monkeypatch: pytest.MonkeyPatch,
    generator_cls: type[TeacherVariantGenerator] | type[TeacherAltCorrectGenerator],
    runner_cls: str,
    gate: str,
) -> None:
    text = (
        "```python\ndef f():\n    return 2\n```"
        if runner_cls == "differential"
        else "```python\ndef f():\n    return 1 + 0\n```"
    )
    client = _concrete_teacher_client()
    client._authority_test_responses = [  # noqa: SLF001 - child-only test mock
        {
            "text": text,
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            "cost": 0.0125,
            "request_id": "gate-positive",
        }
    ]
    if runner_cls == "differential":
        import swe_forge.forge.oracle.differential as differential_module

        monkeypatch.setattr(
            differential_module,
            "DockerDifferentialRunner",
            _WrappedDifferentialRunner,
        )
        result = await run_differential_gate(
            _candidate(),
            _env_image(),
            _prior_report(),
            variant_generator=generator_cls(  # type: ignore[arg-type, call-arg]
                client=client
            ),
            adapter=PythonAdapter(),
        )
    else:
        import swe_forge.forge.oracle.alt_correct as alt_correct_module

        monkeypatch.setattr(
            alt_correct_module,
            "DockerAltCorrectRunner",
            _WrappedAltRunner,
        )
        result = await run_alt_correct_gate(
            _candidate(),
            _env_image(),
            _prior_report(),
            alt_generator=generator_cls(  # type: ignore[arg-type, call-arg]
                client=client
            ),
            adapter=PythonAdapter(),
        )

    assert result.is_pass
    calls = result.details["teacher_gates"][gate]["calls"]  # type: ignore[index]
    assert calls[0]["parsed_proposals"] == 1  # type: ignore[index]
    assert calls[0]["executable_proposals"] == 1  # type: ignore[index]
    assert calls[0]["execution_completed"] == 1  # type: ignore[index]


@pytest.mark.parametrize(
    ("generator_cls", "runner_cls", "gate", "client_kind"),
    [
        (TeacherVariantGenerator, "differential", "differential", "fake"),
        (TeacherAltCorrectGenerator, "alt_correct", "alt_correct", "fake"),
        (TeacherVariantGenerator, "differential", "differential", "subclass"),
        (TeacherAltCorrectGenerator, "alt_correct", "alt_correct", "subclass"),
    ],
)
async def test_standalone_gates_reject_nonconcrete_teacher_calls(
    monkeypatch: pytest.MonkeyPatch,
    generator_cls: type[TeacherVariantGenerator] | type[TeacherAltCorrectGenerator],
    runner_cls: str,
    gate: str,
    client_kind: str,
) -> None:
    text = (
        "```python\ndef f():\n    return 2\n```"
        if runner_cls == "differential"
        else "```python\ndef f():\n    return 1 + 0\n```"
    )
    client = _Teacher(text) if client_kind == "fake" else _TeacherClientSubclass(text)
    if runner_cls == "differential":
        import swe_forge.forge.oracle.differential as differential_module

        monkeypatch.setattr(
            differential_module, "DockerDifferentialRunner", _WrappedDifferentialRunner
        )
        result = await run_differential_gate(
            _candidate(),
            _env_image(),
            _prior_report(),
            variant_generator=generator_cls(  # type: ignore[arg-type, call-arg]
                client=client  # type: ignore[arg-type]
            ),
            adapter=PythonAdapter(),
        )
    else:
        import swe_forge.forge.oracle.alt_correct as alt_correct_module

        monkeypatch.setattr(
            alt_correct_module, "DockerAltCorrectRunner", _WrappedAltRunner
        )
        result = await run_alt_correct_gate(
            _candidate(),
            _env_image(),
            _prior_report(),
            alt_generator=generator_cls(  # type: ignore[arg-type, call-arg]
                client=client  # type: ignore[arg-type]
            ),
            adapter=PythonAdapter(),
        )

    assert result.verdict == "reject"
    assert result.reasons == [
        f"{gate}_teacher_evidence_invalid: "
        f"{gate}: no real-teacher proposal call was recorded"
    ]


@pytest.mark.parametrize(
    ("generator_cls", "context", "gate"),
    [
        (TeacherVariantGenerator, _variant_context, "differential"),
        (TeacherAltCorrectGenerator, _alt_context, "alt_correct"),
    ],
)
async def test_concrete_teacher_errors_are_attributable_and_secret_free(
    generator_cls: type[TeacherVariantGenerator] | type[TeacherAltCorrectGenerator],
    context: object,
    gate: str,
) -> None:
    generator = generator_cls(  # type: ignore[arg-type, call-arg]
        client=_concrete_teacher_client(api_key="sk-never-serialize", base_url="")
    )

    assert await generator(context()) == []  # type: ignore[misc]
    evidence = generator.last_call
    assert evidence is not None
    assert evidence.real_teacher is True
    assert evidence.status == "error"
    assert evidence.error_type == "MissingCredentialsError"
    assert teacher_gate_failure_reason(gate, [evidence]) == (
        f"{gate}_teacher_call_failed:MissingCredentialsError"
    )
    assert "sk-never-serialize" not in json.dumps(evidence.to_dict())


def _pass_report(*, teacher_gates: object) -> OracleReport:
    tests = [OracleTestFile(path="tests/hidden.py", content="def test_x(): pass\n")]
    return OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="pass",
        fail_to_pass=["python -m pytest tests/hidden.py"],
        pass_to_pass=["python -m pytest"],
        test_files=tests,
        flakiness_runs=3,
        mutants_total=2,
        mutants_killed=2,
        final_mutation_evidence=FinalMutationEvidence(
            suite_fingerprint=final_suite_fingerprint(tests),
            mutants_total=2,
            mutants_killed=2,
            threshold=0.8,
            tool="fake",
        ),
        differential_pass=True,
        alt_correct_accepted=True,
        leak_audit="clean",
        details={
            "teacher_gates": teacher_gates,
            "alt_correct": protected_alt_correct_summary(tests),
        },
        protected_alt_correct_audit=protected_alt_correct_audit(
            tests,
            ["python -m pytest tests/hidden.py"],
            [("src/m.py", "def f(): return 1\n")],
        ),
    )


def _positive_gate_evidence(gate: str) -> dict[str, object]:
    return {
        "calls": [
            {
                "gate": gate,
                "call_kind": "proposal",
                "real_teacher": True,
                "status": "success",
                "response_kind": "content",
                "model": "anthropic/test-model",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "cost": 0.01,
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
        ]
    }


def test_oracle_consistency_requires_positive_teacher_execution_evidence() -> None:
    missing = _pass_report(teacher_gates={})
    assert any(
        "differential: teacher" in problem
        for problem in teacher_gate_evidence_issues(missing.details)
    )
    with pytest.raises(ExportRefusedError):
        ensure_oracle_exportable(missing, candidate=_candidate(), calibration_kept=True)

    valid = _pass_report(
        teacher_gates={
            "differential": _positive_gate_evidence("differential"),
            "alt_correct": _positive_gate_evidence("alt_correct"),
        }
    )
    assert verify_pass_consistency(valid) == []
    ensure_oracle_exportable(valid, calibration_kept=True)
    usage, cost = aggregate_teacher_gate_usage(valid.details)
    assert usage.total_tokens == 4
    assert cost == 0.02
    serialized = json.dumps(valid.to_dict())
    assert "sk-" not in serialized

    unsafe = _pass_report(teacher_gates=valid.details["teacher_gates"])
    unsafe.details["teacher_gates"]["differential"]["calls"][0]["api_key"] = (  # type: ignore[index]
        "sk-never-serialize"
    )
    assert any("unsafe" in problem for problem in verify_pass_consistency(unsafe))


async def test_nonempty_executable_variants_and_alternatives_keep_existing_semantics() -> (
    None
):
    differential = await assess_differential(_DifferentialRunner(), [_variant()])
    alternatives = await assess_alt_correct(_AltRunner(), [_alt()])

    assert differential.verdict == "pass"
    assert differential.variants_killed == 1
    assert alternatives.verdict == "pass"
    assert alternatives.alternatives_accepted == 1


async def test_exact_client_with_monkeypatched_complete_text_cannot_authorize_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the class-owned transport, not exact instance identity, is authority."""
    import swe_forge.forge.oracle.differential as differential_module

    candidate = _candidate()
    client = _concrete_teacher_client()

    async def forged_complete_text(*_args: object, **_kwargs: object) -> LLMResult:
        return LLMResult(
            text="```python\ndef f():\n    return 2\n```",
            usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            cost=0.0125,
            finish_reason="stop",
            transport_receipt=TransportReceipt(
                call_id="f" * 32,
                candidate_fingerprint=candidate_transport_fingerprint(candidate),
                gate="differential",
                call_kind="proposal",
                model=client.model,
                usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
                cost=0.0125,
                provider_request_id="forged-request",
                response_commitment="f" * 64,
                ledger_linkage="not_applicable",
                issuer_key_id="0" * 64,
                signature="forged-signature",
            ),
        )

    monkeypatch.setattr(client, "complete_text", forged_complete_text)
    monkeypatch.setattr(
        differential_module, "DockerDifferentialRunner", _WrappedDifferentialRunner
    )

    result = await run_differential_gate(
        candidate,
        _env_image(),
        _prior_report(),
        variant_generator=TeacherVariantGenerator(client=client),
        adapter=PythonAdapter(),
    )

    assert result.verdict == "reject"
    assert "receipt" in result.reasons[0]


async def test_concrete_transport_issues_private_receipt_after_mocked_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocking LiteLLM keeps the concrete transport path authoritative."""
    client = _concrete_teacher_client()
    client._authority_test_responses = [  # noqa: SLF001 - child-only test mock
        {
            "text": "```python\ndef f():\n    return 2\n```",
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            "cost": 0.0125,
            "request_id": "mocked-inside-authority",
        }
    ]
    generator = TeacherVariantGenerator(client=client)
    assert await generator(_variant_context())

    evidence = generator.last_call
    assert evidence is not None
    assert evidence.call_id
    assert evidence.receipt_commitment
    assert evidence.protected_transport_receipt is not None
    public = evidence.to_dict()
    private = evidence.protected_transport_receipt.to_private_dict()
    assert "issuer_key_id" not in json.dumps(public)
    assert "api_key" not in json.dumps(private)
    assert "prompt_text" not in json.dumps(private)
    assert "response_content" not in json.dumps(private)


def test_positive_public_teacher_json_without_protected_receipt_is_rejected() -> None:
    evidence = _positive_gate_evidence("differential")
    issues = teacher_gate_evidence_issues(
        {"teacher_gates": {"differential": evidence}},
        gates=("differential",),
        protected_receipts=[],
    )

    assert issues


def test_teacher_evidence_rejects_receipt_with_mismatched_ledger_linkage() -> None:
    candidate = _candidate()
    evidence = _positive_gate_evidence("differential")
    call = evidence["calls"][0]
    assert isinstance(call, dict)
    receipt = signed_transport_receipt(
        call_id="ledger-binding",
        candidate_fingerprint=candidate_transport_fingerprint(candidate),
        gate="differential",
        call_kind="proposal",
        model="anthropic/test-model",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        cost=0.01,
    )
    call.update(
        {
            "recovery_accounting": {
                "logical_call_id": "recovery-logical-id",
                "physical_calls": [],
            },
            "call_id": receipt.call_id,
            "receipt_commitment": receipt.commitment,
        }
    )

    issues = teacher_gate_evidence_issues(
        {"teacher_gates": {"differential": evidence}},
        gates=("differential",),
        candidate=candidate,
        protected_receipts=[receipt.to_private_dict()],
    )

    assert any("ledger linkage mismatches" in issue for issue in issues)
