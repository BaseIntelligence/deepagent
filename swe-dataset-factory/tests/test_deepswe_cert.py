"""DeepSWE Docker oracle cert API (VAL-ORCD-001..007) — offline unit tests.

Integration Docker dual-run remains in test_harbor_oracle_integration.py;
these tests pin cert-path refuse-fake, isolation, audit, flake, pier hooks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.harbor.deepswe_cert import (
    DeepSWECertError,
    FakeBackendRejected,
    append_cert_audit,
    build_pier_ready_hooks,
    certify_deepswe_pack,
    collect_cert_audit_fields,
    evaluate_flake_gate,
    is_real_base_sha,
    is_real_repository_url,
    read_pack_meta,
    refuse_fake_backend,
    scan_pack_agent_isolation,
    write_oracle_evidence,
)
from swe_factory.harbor.export_pack import export_harbor_pack
from swe_factory.harbor.harbor_oracle import FakeHarborVerifier, VerifierRunResult
from swe_factory.harbor.schema import (
    MODEL_PATCH_ARTIFACT,
    HarborMetadata,
    HarborPackSpec,
    HarborTaskIdentity,
    HarborTaskToml,
    HarborVerifier,
    TestsConfig,
)
from swe_factory.oracle import codes as C
from swe_factory.producers.harbor_labeling import SuiteOutcome

runner = CliRunner()

# Real-looking but offline metadata for audit gates (not claimed live remote).
_REAL_URL = "https://github.com/pallets/click.git"
_REAL_SHA = "a1b2c3d4e5f6789012345678901234567890abcd"


@dataclass
class ScriptedDockerVerifier:
    """Docker-named injectable backend for offline cert unit tests."""

    solution_reward: int | float = 1
    null_reward: int | float = 0
    cleaned: bool = False
    call_log: list[str] = field(default_factory=list)

    def run_solution(self, pack_dir: Path) -> VerifierRunResult:
        del pack_dir
        self.call_log.append("solution")
        reward = self.solution_reward
        return VerifierRunResult(
            phase="solution",
            reward=reward,
            reward_json={
                "reward": reward,
                "f2p_total": 2,
                "f2p_passed": 2 if reward == 1 else 0,
                "p2p_total": 1,
                "p2p_passed": 1 if reward == 1 else 0,
            },
            logs="scripted docker solution",
            ok=True,
        )

    def run_null(self, pack_dir: Path) -> VerifierRunResult:
        del pack_dir
        self.call_log.append("null")
        reward = self.null_reward
        return VerifierRunResult(
            phase="null",
            reward=reward,
            reward_json={
                "reward": reward,
                "f2p_total": 2,
                "f2p_passed": 0,
                "p2p_total": 1,
                "p2p_passed": 1 if reward == 0 else 0,
            },
            logs="scripted docker null",
            ok=True,
        )

    def cleanup(self) -> None:
        self.cleaned = True


def _spec(
    *,
    task_id: str = "deepswe-cert-demo",
    repository_url: str = _REAL_URL,
    base_commit: str = _REAL_SHA,
    f2p: list[str] | None = None,
) -> HarborPackSpec:
    return HarborPackSpec.model_validate(
        {
            "task_id": task_id,
            "instruction_md": "Fix multi-file bug in real public repo.\n",
            "task_toml": HarborTaskToml(
                schema_version="1.1",
                artifacts=[MODEL_PATCH_ARTIFACT],
                task=HarborTaskIdentity(name=f"swe-factory/{task_id}"),
                metadata=HarborMetadata(
                    language="python",
                    repository_url=repository_url,
                    base_commit_hash=base_commit,
                    task_id=task_id,
                    source_track="real_pr",
                    license="BSD-3-Clause",
                ),
                verifier=HarborVerifier(environment_mode="separate", timeout_sec=120.0),
            ),
            "tests_config": TestsConfig(
                base_commit=base_commit,
                f2p_node_ids=f2p or ["tests.test_math.test_add"],
                p2p_node_ids=["tests.test_ok.test_always_ok"],
            ),
            "solution_patch": (
                "diff --git a/src/a.py b/src/a.py\n"
                "--- a/src/a.py\n"
                "+++ b/src/a.py\n"
                "@@ -1 +1 @@\n"
                "-broken\n"
                "+fixed\n"
                "diff --git a/src/b.py b/src/b.py\n"
                "--- a/src/b.py\n"
                "+++ b/src/b.py\n"
                "@@ -1 +1 @@\n"
                "-x\n"
                "+y\n"
            ),
            "test_patch": (
                "diff --git a/tests/new.py b/tests/new.py\n"
                "--- /dev/null\n"
                "+++ b/tests/new.py\n"
                "@@ -0,0 +1,2 @@\n"
                "+def test_add():\n"
                "+    assert True\n"
            ),
            "environment_dockerfile": "FROM python:3.12-slim\nWORKDIR /app\n",
            "tests_dockerfile": "FROM deepswe-agent:local\nCOPY test.sh /tests/test.sh\n",
        }
    )


def _export_pack(tmp_path: Path, **kwargs: Any) -> Path:
    pack = export_harbor_pack(
        _spec(**kwargs), dest=tmp_path / "tasks" / kwargs.get("task_id", "deepswe-cert-demo")
    )
    return pack.pack_dir


# ---------------------------------------------------------------------------
# VAL-ORCD-004 — refuse fake backend on cert path
# ---------------------------------------------------------------------------


def test_refuse_fake_backend_string() -> None:
    with pytest.raises(FakeBackendRejected, match="refuses fake"):
        refuse_fake_backend("fake", certified=True)


def test_refuse_fake_verifier_instance() -> None:
    with pytest.raises(FakeBackendRejected):
        refuse_fake_backend(FakeHarborVerifier(), certified=True)


def test_refuse_fake_on_deepswe_dest_even_without_certified_flag() -> None:
    with pytest.raises(FakeBackendRejected):
        refuse_fake_backend("fake", certified=False, dest="datasets/deepswe_v1/tasks/x")


def test_fake_still_allowed_for_non_cert_fixture_path() -> None:
    # fixture demos under harbor_fixture remain OK without certified flag
    refuse_fake_backend("fake", certified=False, dest="datasets/harbor_fixture")


def test_certify_rejects_fake_backend(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    with pytest.raises(FakeBackendRejected, match="refuses fake|backend=docker"):
        certify_deepswe_pack(pack, backend="fake")


def test_certify_rejects_fake_verifier_object(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    with pytest.raises(FakeBackendRejected):
        certify_deepswe_pack(pack, backend=FakeHarborVerifier())


# ---------------------------------------------------------------------------
# VAL-ORCD-001 / 002 — sol=1 null=0 via docker-named backend
# ---------------------------------------------------------------------------


def test_certify_sol1_null0_with_scripted_docker(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    backend = ScriptedDockerVerifier(solution_reward=1, null_reward=0)
    result = certify_deepswe_pack(
        pack,
        backend=backend,
        run_pier_hooks=False,
    )
    assert result.solution_reward == 1
    assert result.null_reward == 0
    assert result.backend == "docker"
    assert result.certified is True, result.reasons
    assert result.disposition == "accept"
    assert backend.cleaned is True
    assert "solution" in backend.call_log and "null" in backend.call_log


def test_certify_rejects_when_null_resolves(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = certify_deepswe_pack(
        pack,
        backend=ScriptedDockerVerifier(solution_reward=1, null_reward=1),
        run_pier_hooks=False,
    )
    assert result.certified is False
    assert result.null_reward == 1
    assert C.G3_NULL_RESOLVES in result.reason_codes


def test_certify_rejects_when_solution_fails(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = certify_deepswe_pack(
        pack,
        backend=ScriptedDockerVerifier(solution_reward=0, null_reward=0),
        run_pier_hooks=False,
    )
    assert result.certified is False
    assert result.solution_reward == 0
    assert any("solution reward" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# VAL-ORCD-003 — isolation scan fails on leaks
# ---------------------------------------------------------------------------


def test_isolation_clean_on_export_pack(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    evidence = scan_pack_agent_isolation(pack)
    assert evidence.clean is True
    assert evidence.hits == ()
    assert evidence.to_dict()["isolation"] == "clean"


def test_isolation_fails_when_test_patch_leaks_into_env(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    # Plant held-out leak into agent environment context
    leak = pack / "environment" / "test.patch"
    leak.write_text("held-out leak\n", encoding="utf-8")
    evidence = scan_pack_agent_isolation(pack)
    assert evidence.clean is False
    assert any("test.patch" in h for h in evidence.hits)


def test_isolation_fails_when_solution_dir_in_env(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    sol = pack / "environment" / "solution"
    sol.mkdir()
    (sol / "solution.patch").write_text("gold\n", encoding="utf-8")
    evidence = scan_pack_agent_isolation(pack)
    assert evidence.clean is False
    assert any("solution" in h for h in evidence.hits)


def test_certify_rejects_isolation_leak(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    (pack / "environment" / "gold.patch").write_text("secret gold\n", encoding="utf-8")
    result = certify_deepswe_pack(
        pack,
        backend=ScriptedDockerVerifier(),
        run_pier_hooks=False,
    )
    assert result.certified is False
    assert result.isolation.clean is False
    assert C.G5_LEAK in result.reason_codes


# ---------------------------------------------------------------------------
# VAL-ORCD-006 — audit fields sol/null/isolation + real_url/real_sha
# ---------------------------------------------------------------------------


def test_pack_meta_real_url_and_sha(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    meta = read_pack_meta(pack)
    assert meta.real_url_ok is True
    assert meta.real_sha_ok is True
    assert meta.repository_url == _REAL_URL
    assert meta.base_commit_hash == _REAL_SHA


def test_pack_meta_rejects_file_url_and_short_sha(tmp_path: Path) -> None:
    pack = _export_pack(
        tmp_path,
        repository_url="file:///tmp/motor.git",
        base_commit="abc123",
        task_id="bad-meta",
    )
    # tests_config base must match; rewrite task.toml fields after export for this test
    # _spec with short sha may fail validation on HarborPackSpec for base_commit
    # so plant meta by rewriting task.toml after export of a valid pack.
    pack2 = _export_pack(tmp_path / "ok")
    text = (pack2 / "task.toml").read_text(encoding="utf-8")
    text = text.replace(_REAL_URL, "file:///tmp/motor.git")
    text = text.replace(_REAL_SHA, "abc123")
    (pack2 / "task.toml").write_text(text, encoding="utf-8")
    meta = read_pack_meta(pack2)
    assert meta.real_url_ok is False
    assert meta.real_sha_ok is False
    del pack  # unused pack built for comparison post-validation path


def test_is_real_helpers() -> None:
    assert is_real_repository_url("https://github.com/foo/bar.git") is True
    assert is_real_repository_url("file://x") is False
    assert is_real_repository_url("https://example.com/x") is False
    assert is_real_base_sha(_REAL_SHA) is True
    assert is_real_base_sha("deadbeef") is False


def test_cert_audit_complete_on_accept(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    evidence_path = tmp_path / "oracle_evidence.json"
    audit_path = tmp_path / "gate_audit.jsonl"
    result = certify_deepswe_pack(
        pack,
        backend=ScriptedDockerVerifier(),
        evidence_out=evidence_path,
        audit_out=audit_path,
        run_pier_hooks=False,
    )
    assert result.certified is True
    audit = collect_cert_audit_fields(result)
    assert audit["complete"] is True
    assert audit["blocks_ship"] is False
    assert audit["sol_ok"] is True
    assert audit["null_ok"] is True
    assert audit["isolation_ok"] is True

    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert data["solution_reward"] == 1
    assert data["null_reward"] == 0
    assert data["isolation_status"] == "clean"
    assert data["repository_url"] == _REAL_URL
    assert data["base_commit_hash"] == _REAL_SHA
    assert data["backend"] == "docker"

    rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["sol"] == 1
    assert rows[0]["null"] == 0
    assert rows[0]["isolation"] == "clean"
    assert rows[0]["repository_url"] == _REAL_URL


def test_write_oracle_evidence_helper(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = certify_deepswe_pack(pack, backend=ScriptedDockerVerifier(), run_pier_hooks=False)
    path = write_oracle_evidence(tmp_path / "ev.json", result)
    assert path.is_file()
    append_cert_audit(tmp_path / "a.jsonl", result)


def test_certify_rejects_when_repo_meta_not_real(tmp_path: Path) -> None:
    pack = _export_pack(
        tmp_path, repository_url="https://example.com/fake.git", task_id="fake-repo"
    )
    result = certify_deepswe_pack(
        pack,
        backend=ScriptedDockerVerifier(),
        run_pier_hooks=False,
    )
    assert result.certified is False
    assert result.pack_meta.real_url_ok is False


# ---------------------------------------------------------------------------
# VAL-ORCD-007 — flake dual disagreement rejects
# ---------------------------------------------------------------------------


def test_evaluate_flake_gate_detects_disagreement() -> None:
    a = SuiteOutcome.from_summary(
        language="python",
        passed=["t1"],
        failed=["t2"],
    )
    b = SuiteOutcome.from_summary(
        language="python",
        passed=["t1", "t2"],  # different signature
        failed=[],
    )
    flake = evaluate_flake_gate(gold_runs=[a, b])
    assert flake.is_flake is True
    assert C.G2_FLAKE in flake.reason_codes or "G2_FLAKE" in flake.reason_codes
    assert C.FLAKE_REJECT in flake.reason_codes or "FLAKE_REJECT" in flake.reason_codes


def test_certify_rejects_flake_and_never_certifies(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path, task_id="flake-task")
    a = SuiteOutcome.from_summary(language="python", passed=["n1"], failed=["n2"])
    b = SuiteOutcome.from_summary(language="python", passed=["n1", "n2"], failed=[])
    result = certify_deepswe_pack(
        pack,
        backend=ScriptedDockerVerifier(),
        gold_runs=[a, b],
        run_pier_hooks=False,
    )
    assert result.certified is False
    assert result.flake.is_flake is True
    assert result.disposition == "reject"
    assert any(
        c in result.reason_codes for c in (C.G2_FLAKE, C.FLAKE_REJECT, "G2_FLAKE", "FLAKE_REJECT")
    )


def test_forced_flake_rejects(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = certify_deepswe_pack(
        pack,
        backend=ScriptedDockerVerifier(),
        forced_flake=True,
        run_pier_hooks=False,
    )
    assert result.certified is False
    assert result.flake.is_flake is True


# ---------------------------------------------------------------------------
# VAL-ORCD-005 — pier-ready pack hooks
# ---------------------------------------------------------------------------


def test_pier_ready_hooks_structural(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    hooks = build_pier_ready_hooks(pack, run_load_smoke=False)
    assert hooks.required_relpaths_ok is True
    assert hooks.missing_relpaths == ()
    assert hooks.pier_job_prefix.startswith("/tmp/harbor-deepswe-jobs")
    assert Path(hooks.pack_dir).is_dir()


def test_certify_includes_pier_hooks(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = certify_deepswe_pack(
        pack,
        backend=ScriptedDockerVerifier(),
        run_pier_hooks=True,
    )
    assert result.pier_ready.required_relpaths_ok is True
    assert "pier_ready" in result.to_dict()
    assert result.pier_ready.pier_job_prefix.startswith("/tmp/harbor-deepswe-jobs")


# ---------------------------------------------------------------------------
# CLI — harbor-oracle --certified refuses fake; deepswe-oracle wrapper
# ---------------------------------------------------------------------------


def test_cli_harbor_oracle_certified_refuses_fake(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = runner.invoke(
        app,
        [
            "harbor-oracle",
            "--pack-dir",
            str(pack),
            "--backend",
            "fake",
            "--certified",
            "--json",
        ],
    )
    assert result.exit_code != 0, result.output
    combined = (result.output or "") + (result.stderr or "")
    assert re.search(r"fake|docker|certified|deepswe", combined, re.I)


def test_cli_deepswe_oracle_refuses_fake(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = runner.invoke(
        app,
        [
            "deepswe-oracle",
            "--pack-dir",
            str(pack),
            "--backend",
            "fake",
            "--json",
        ],
    )
    assert result.exit_code != 0, result.output
    combined = (result.output or "") + (result.stderr or "")
    assert re.search(r"fake|docker", combined, re.I)


def test_cli_harbor_oracle_fake_without_certified_still_ok(tmp_path: Path) -> None:
    """Historical fixture path: fake backend without --certified remains valid."""
    out = tmp_path / "harbor_fixture"
    result = runner.invoke(
        app,
        [
            "harbor-oracle",
            "--from-fixture",
            "--backend",
            "fake",
            "--out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["passed"] is True
    assert payload["mode"] == "fake"


def test_cli_help_lists_deepswe_oracle() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "deepswe-oracle" in result.output or "deepswe" in result.output.lower()


def test_cli_harbor_oracle_help_mentions_certified() -> None:
    result = runner.invoke(app, ["harbor-oracle", "--help"])
    assert result.exit_code == 0
    assert "certified" in result.output


def test_missing_pack_raises() -> None:
    with pytest.raises(DeepSWECertError, match="not found"):
        certify_deepswe_pack(
            "/tmp/does-not-exist-deepswe-pack-xyz", backend=ScriptedDockerVerifier()
        )
