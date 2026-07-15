"""Real-PR Docker oracle cert API (VAL-RORC-001..004) — offline unit tests.

Pins product cert path:
- sol=1 / null=0 on docker-named backend (001/002)
- sol/null evidence files written
- isolation fail-closed on leaks (003)
- refuse fake / oracle_mode=fake (004)
- source_track=real_pr required; hybrid refused
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
from swe_factory.harbor.export_pack import export_harbor_pack
from swe_factory.harbor.harbor_oracle import FakeHarborVerifier, VerifierRunResult
from swe_factory.harbor.real_oracle_cert import (
    REAL_PR_SOURCE_TRACK,
    RORC_HYBRID_REFUSED,
    RORC_ISOLATION_LEAK,
    RORC_NULL_0,
    RORC_PASS,
    RORC_SOL_1,
    RealOracleCertError,
    RealPrFakeOracleRejected,
    assert_real_pr_source_track,
    certify_real_pr_pack,
    collect_rorc_audit_fields,
    isolation_blocks_real_pr_promote,
    refuse_fake_oracle_mode_real_pr,
    write_sol_null_evidence_files,
)
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

runner = CliRunner()

_REAL_URL = "https://github.com/pallets/click.git"
_REAL_SHA = "a1b2c3d4e5f6789012345678901234567890abcd"


@dataclass
class ScriptedDockerVerifier:
    """Docker-named injectable backend for offline Real-PR cert unit tests."""

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
            logs="scripted docker solution (real_pr)",
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
            logs="scripted docker null (real_pr)",
            ok=True,
        )

    def cleanup(self) -> None:
        self.cleaned = True


def _spec(
    *,
    task_id: str = "realpr-orc-demo",
    repository_url: str = _REAL_URL,
    base_commit: str = _REAL_SHA,
    source_track: str = "real_pr",
    f2p: list[str] | None = None,
) -> HarborPackSpec:
    # Real-PR agent Dockerfile: clone@SHA, no motor COPY
    env_df = (
        "FROM python:3.12-slim\n"
        "WORKDIR /app\n"
        f"ARG BASE_SHA={base_commit}\n"
        f"ARG REPO_URL={repository_url}\n"
        'RUN git clone --filter=blob:none "$REPO_URL" /app '
        '&& cd /app && git checkout --force "$BASE_SHA"\n'
        'LABEL swe_factory.source_track="real_pr"\n'
        f'LABEL swe_factory.base_commit="{base_commit}"\n'
        "ENV HARBOR_ALLOW_INTERNET=false\n"
    )
    return HarborPackSpec.model_validate(
        {
            "task_id": task_id,
            "instruction_md": "Restore multi-file behavior in real upstream tree.\n",
            "task_toml": HarborTaskToml(
                schema_version="1.1",
                artifacts=[MODEL_PATCH_ARTIFACT],
                task=HarborTaskIdentity(name=f"swe-factory/{task_id}"),
                metadata=HarborMetadata(
                    language="python",
                    repository_url=repository_url,
                    base_commit_hash=base_commit,
                    task_id=task_id,
                    source_track=source_track,
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
            "environment_dockerfile": env_df,
            "tests_dockerfile": "FROM deepswe-agent:local\nCOPY test.sh /tests/test.sh\n",
        }
    )


def _export_pack(tmp_path: Path, **kwargs: Any) -> Path:
    pack = export_harbor_pack(
        _spec(**kwargs),
        dest=tmp_path / "tasks" / kwargs.get("task_id", "realpr-orc-demo"),
    )
    return pack.pack_dir


# ---------------------------------------------------------------------------
# VAL-RORC-004 — refuse fake on Real-PR product cert
# ---------------------------------------------------------------------------


def test_refuse_fake_oracle_mode_real_pr_string() -> None:
    with pytest.raises(RealPrFakeOracleRejected):
        refuse_fake_oracle_mode_real_pr("fake", certified=True)


def test_refuse_fake_oracle_mode_keyword() -> None:
    with pytest.raises(RealPrFakeOracleRejected, match="oracle_mode=fake|refuses"):
        refuse_fake_oracle_mode_real_pr(
            "docker", certified=True, oracle_mode="fake", dest="datasets/deepswe_v1"
        )


def test_refuse_fake_verifier_instance() -> None:
    with pytest.raises((RealPrFakeOracleRejected, Exception)):
        refuse_fake_oracle_mode_real_pr(FakeHarborVerifier(), certified=True)


def test_certify_real_pr_rejects_fake_backend(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    with pytest.raises(RealPrFakeOracleRejected):
        certify_real_pr_pack(pack, backend="fake")


def test_certify_real_pr_rejects_oracle_mode_fake(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    with pytest.raises(RealPrFakeOracleRejected):
        certify_real_pr_pack(pack, backend="docker", oracle_mode="fake")


# ---------------------------------------------------------------------------
# VAL-RORC-001 / 002 — sol=1 null=0 + evidence files
# ---------------------------------------------------------------------------


def test_val_rorc_001_002_sol1_null0_and_evidence(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path, task_id="rorc-solnull")
    backend = ScriptedDockerVerifier(solution_reward=1, null_reward=0)
    evidence_dir = tmp_path / "evidence"
    result = certify_real_pr_pack(
        pack,
        backend=backend,
        evidence_dir=evidence_dir,
        run_pier_hooks=False,
    )
    assert result.solution_reward == 1
    assert result.null_reward == 0
    assert result.backend == "docker"
    assert result.oracle_mode == "docker"
    assert result.certified is True, result.reasons
    assert result.source_track == REAL_PR_SOURCE_TRACK
    assert RORC_SOL_1 in result.reason_codes
    assert RORC_NULL_0 in result.reason_codes
    assert RORC_PASS in result.reason_codes
    assert "solution" in backend.call_log and "null" in backend.call_log
    assert backend.cleaned is True

    assert result.evidence_files is not None
    sol_path = Path(result.evidence_files.sol_path)
    null_path = Path(result.evidence_files.null_path)
    assert sol_path.is_file()
    assert null_path.is_file()
    sol_data = json.loads(sol_path.read_text(encoding="utf-8"))
    null_data = json.loads(null_path.read_text(encoding="utf-8"))
    assert sol_data["reward"] == 1
    assert sol_data["backend"] == "docker"
    assert sol_data["source_track"] == "real_pr"
    assert null_data["reward"] == 0
    assert null_data["phase"] == "null"

    combined = Path(result.evidence_files.combined_path or "")
    assert combined.is_file()
    combo = json.loads(combined.read_text(encoding="utf-8"))
    assert combo["sol_ok"] is True
    assert combo["null_ok"] is True
    assert combo["pair_ok"] is True


def test_certify_rejects_when_null_resolves(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path, task_id="rorc-null-resolves")
    result = certify_real_pr_pack(
        pack,
        backend=ScriptedDockerVerifier(solution_reward=1, null_reward=1),
        run_pier_hooks=False,
        evidence_dir=tmp_path / "ev",
    )
    assert result.certified is False
    assert result.null_reward == 1
    assert C.G3_NULL_RESOLVES in result.reason_codes
    assert result.evidence_files is not None
    null_data = json.loads(Path(result.evidence_files.null_path).read_text(encoding="utf-8"))
    assert null_data["reward"] == 1


def test_certify_rejects_when_solution_fails(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path, task_id="rorc-sol-fail")
    result = certify_real_pr_pack(
        pack,
        backend=ScriptedDockerVerifier(solution_reward=0, null_reward=0),
        run_pier_hooks=False,
    )
    assert result.certified is False
    assert result.solution_reward == 0


# ---------------------------------------------------------------------------
# VAL-RORC-003 — isolation fail-closed
# ---------------------------------------------------------------------------


def test_isolation_clean_on_real_pr_pack(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    evidence = isolation_blocks_real_pr_promote(pack)
    assert evidence.clean is True
    assert evidence.hits == ()


def test_isolation_fails_on_held_out_test_patch_leak(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path, task_id="rorc-leak-test")
    (pack / "environment" / "test.patch").write_text("held-out leak\n", encoding="utf-8")
    evidence = isolation_blocks_real_pr_promote(pack)
    assert evidence.clean is False
    assert any("test.patch" in h for h in evidence.hits)


def test_isolation_fails_on_solution_dir_leak(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path, task_id="rorc-leak-sol")
    sol = pack / "environment" / "solution"
    sol.mkdir()
    (sol / "solution.patch").write_text("gold\n", encoding="utf-8")
    evidence = isolation_blocks_real_pr_promote(pack)
    assert evidence.clean is False


def test_certify_rejects_isolation_leak_blocks_promote(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path, task_id="rorc-isol-block")
    (pack / "environment" / "gold.patch").write_text("secret gold\n", encoding="utf-8")
    result = certify_real_pr_pack(
        pack,
        backend=ScriptedDockerVerifier(),
        run_pier_hooks=False,
        evidence_dir=tmp_path / "ev",
    )
    assert result.certified is False
    assert result.isolation.clean is False
    assert result.disposition == "reject"
    assert RORC_ISOLATION_LEAK in result.reason_codes or C.G5_LEAK in result.reason_codes
    # Evidence still written for sol/null observability even on isolation refuse
    assert result.evidence_files is not None
    combo = json.loads(Path(result.evidence_files.combined_path or "").read_text(encoding="utf-8"))
    assert combo["isolation"] == "leak"


# ---------------------------------------------------------------------------
# Real_pr track required / hybrid refused
# ---------------------------------------------------------------------------


def test_assert_real_pr_source_track_happy() -> None:
    assert assert_real_pr_source_track("real_pr") == REAL_PR_SOURCE_TRACK


def test_assert_real_pr_source_track_refuses_hybrid() -> None:
    with pytest.raises(RealOracleCertError, match="hybrid|real_pr"):
        assert_real_pr_source_track("hybrid_curated", task_id="x")


def test_certify_refuses_hybrid_source_track(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path, task_id="rorc-hybrid", source_track="hybrid_curated")
    # Harbor schema may coerce; rewrite task.toml if needed
    text = (pack / "task.toml").read_text(encoding="utf-8")
    if "hybrid_curated" not in text:
        text = text.replace('source_track = "real_pr"', 'source_track = "hybrid_curated"')
        (pack / "task.toml").write_text(text, encoding="utf-8")
    result = certify_real_pr_pack(
        pack,
        backend=ScriptedDockerVerifier(),
        run_pier_hooks=False,
    )
    assert result.certified is False
    assert RORC_HYBRID_REFUSED in result.reason_codes or any(
        "hybrid" in r.lower() or "source_track" in r.lower() for r in result.reasons
    )


def test_certify_refuses_missing_source_track(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path, task_id="rorc-no-track")
    text = (pack / "task.toml").read_text(encoding="utf-8")
    text = re.sub(r'source_track\s*=\s*"[^"]*"\n?', "", text)
    (pack / "task.toml").write_text(text, encoding="utf-8")
    result = certify_real_pr_pack(
        pack,
        backend=ScriptedDockerVerifier(),
        run_pier_hooks=False,
    )
    assert result.certified is False


# ---------------------------------------------------------------------------
# Audit completeness + helpers
# ---------------------------------------------------------------------------


def test_rorc_audit_complete_on_accept(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path, task_id="rorc-audit")
    result = certify_real_pr_pack(
        pack,
        backend=ScriptedDockerVerifier(),
        evidence_dir=tmp_path / "ev",
        audit_out=tmp_path / "gate_audit.jsonl",
        run_pier_hooks=False,
    )
    assert result.certified is True
    audit = collect_rorc_audit_fields(result)
    assert audit["complete"] is True
    assert audit["blocks_promote"] is False
    assert audit["sol_ok"] is True
    assert audit["null_ok"] is True
    assert audit["isolation_ok"] is True
    assert audit["track_ok"] is True
    rows = [
        json.loads(line)
        for line in (tmp_path / "gate_audit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["sol"] == 1
    assert rows[0]["null"] == 0
    assert rows[0]["source_track"] == "real_pr"
    assert rows[0]["oracle_mode"] == "docker"


def test_write_sol_null_evidence_files_helper(tmp_path: Path) -> None:
    files = write_sol_null_evidence_files(
        tmp_path / "ev",
        task_id="x",
        solution_reward=1,
        null_reward=0,
        repository_url=_REAL_URL,
        base_commit_hash=_REAL_SHA,
    )
    assert Path(files.sol_path).is_file()
    assert Path(files.null_path).is_file()
    assert Path(files.combined_path or "").is_file()


# ---------------------------------------------------------------------------
# CLI — deepswe-oracle --real-pr refuses fake; accepts docker scripted path
# ---------------------------------------------------------------------------


def test_cli_deepswe_oracle_real_pr_refuses_fake(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = runner.invoke(
        app,
        [
            "deepswe-oracle",
            "--pack-dir",
            str(pack),
            "--backend",
            "fake",
            "--real-pr",
            "--json",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0, result.output
    combined = (result.output or "") + (result.stderr or "")
    assert re.search(r"fake|docker|real.?pr|oracle_mode", combined, re.I)


def test_cli_deepswe_oracle_refuses_oracle_mode_fake(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = runner.invoke(
        app,
        [
            "deepswe-oracle",
            "--pack-dir",
            str(pack),
            "--backend",
            "docker",
            "--oracle-mode",
            "fake",
            "--real-pr",
            "--json",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0, result.output
    combined = (result.output or "") + (result.stderr or "")
    assert re.search(r"fake", combined, re.I)


def test_cli_help_lists_real_pr_flag() -> None:
    result = runner.invoke(app, ["deepswe-oracle", "--help"])
    assert result.exit_code == 0
    assert "real-pr" in result.output or "real_pr" in result.output.lower()
    assert "fake" in result.output.lower() or "refuse" in result.output.lower()


def test_missing_pack_raises() -> None:
    with pytest.raises((RealOracleCertError, Exception), match="not found"):
        certify_real_pr_pack(
            "/tmp/does-not-exist-realpr-pack-xyz",
            backend=ScriptedDockerVerifier(),
        )
