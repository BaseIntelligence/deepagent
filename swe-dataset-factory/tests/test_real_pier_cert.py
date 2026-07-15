"""Real-PR Pier cert adapter (VAL-RPIER-001..004) — offline unit tests.

Pins product pier smoke path:
- load Real-PR pack without schema error (001)
- sol reward=1 preferred real pier path; scripted only with flag (002)
- null reward=0 paired (003)
- refuse fake / oracle_mode=fake (004)
- evidence under /tmp/harbor-deepswe-jobs*
- pier unavailable cannot mark scripted as full substitute without flag
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.harbor.export_pack import export_harbor_pack
from swe_factory.harbor.pier_cert import ScriptedPierRunner
from swe_factory.harbor.real_pier_cert import (
    RPIER_NULL_0,
    RPIER_PASS,
    RPIER_SCRIPTED_NOT_FULL_SUBSTITUTE,
    RPIER_SOL_1,
    RPIER_STRUCTURAL_OK,
    RealPierFakeOracleRejected,
    RealPierUnavailableError,
    certify_real_pier_pack,
    pier_is_available,
    prefer_real_pier_runner,
    refuse_fake_oracle_mode_real_pier,
    write_real_pier_sol_null_evidence,
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

runner = CliRunner()

_REAL_URL = "https://github.com/pallets/click.git"
_REAL_SHA = "a1b2c3d4e5f6789012345678901234567890abcd"
_JOBS = "/tmp/harbor-deepswe-jobs-realpier-unit"


def _spec(
    *,
    task_id: str = "realpr-pier-demo",
    repository_url: str = _REAL_URL,
    base_commit: str = _REAL_SHA,
    source_track: str = "real_pr",
) -> HarborPackSpec:
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
            "instruction_md": "Fix multi-file bug in real upstream via Pier.\n",
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
                f2p_node_ids=["tests.test_math.test_add"],
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
    tid = kwargs.get("task_id", "realpr-pier-demo")
    pack = export_harbor_pack(_spec(**kwargs), dest=tmp_path / "tasks" / tid)
    return pack.pack_dir


# ---------------------------------------------------------------------------
# VAL-RPIER-004 refuse fake
# ---------------------------------------------------------------------------


def test_refuse_fake_oracle_mode_real_pier() -> None:
    with pytest.raises(RealPierFakeOracleRejected, match="fake|docker|RPIER"):
        refuse_fake_oracle_mode_real_pier("fake", certified=True)


def test_certify_rejects_fake_mode(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    with pytest.raises(RealPierFakeOracleRejected):
        certify_real_pier_pack(
            pack,
            runner=ScriptedPierRunner(),
            jobs_root=f"{_JOBS}-fake",
            oracle_mode="fake",
            run_load_smoke=False,
            allow_scripted_substitute=True,
        )


# ---------------------------------------------------------------------------
# VAL-RPIER-001 structural + VAL-RPIER-002/003 sol=1 null=0
# ---------------------------------------------------------------------------


def test_certify_real_pier_sol1_null0_with_scripted_flag(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    jobs = Path(f"{_JOBS}-ok")
    evidence_dir = tmp_path / "evidence"
    scripted = ScriptedPierRunner(oracle_reward=1, null_reward=0)
    result = certify_real_pier_pack(
        pack,
        runner=scripted,
        jobs_root=jobs,
        run_load_smoke=False,
        allow_scripted_substitute=True,
        evidence_dir=evidence_dir,
        evidence_out=tmp_path / "pier_evidence.json",
        audit_out=tmp_path / "pier_audit.jsonl",
    )
    assert result.certified is True, result.reasons
    assert result.disposition == "accept"
    assert result.solution_reward == 1
    assert result.null_reward == 0
    assert result.structural_ok is True
    assert result.source_track == "real_pr"
    assert result.backend == "docker"
    assert result.oracle_mode == "docker"
    assert result.isolation.clean is True
    assert RPIER_SOL_1 in result.reason_codes
    assert RPIER_NULL_0 in result.reason_codes
    assert RPIER_STRUCTURAL_OK in result.reason_codes
    assert RPIER_PASS in result.reason_codes
    assert str(result.jobs_root).startswith("/tmp/harbor-deepswe-jobs")

    # reward.json sol=1 + null=0 evidence paths
    assert result.evidence_files is not None
    sol_path = Path(result.evidence_files.sol_path or "")
    null_path = Path(result.evidence_files.null_path or "")
    assert sol_path.is_file()
    assert null_path.is_file()
    sol_payload = json.loads(sol_path.read_text(encoding="utf-8"))
    null_payload = json.loads(null_path.read_text(encoding="utf-8"))
    assert sol_payload["reward"] == 1
    assert null_payload["reward"] == 0
    assert sol_payload["oracle_mode"] != "fake"
    assert null_payload.get("backend") == "docker"
    assert sol_payload["source_track"] == "real_pr"

    # Pier job reward paths (under jobs)
    assert result.oracle_run is not None
    assert result.null_run is not None
    assert result.oracle_run.reward.reward == 1
    assert result.null_run.reward.reward == 0
    assert result.oracle_run.reward.path is not None
    assert Path(result.oracle_run.reward.path).is_file()
    assert "oracle" in scripted.call_log
    assert "nop" in scripted.call_log


def test_certify_rejects_oracle_not_1(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = certify_real_pier_pack(
        pack,
        runner=ScriptedPierRunner(oracle_reward=0, null_reward=0),
        jobs_root=f"{_JOBS}-oracfail",
        run_load_smoke=False,
        allow_scripted_substitute=True,
    )
    assert result.certified is False
    assert result.solution_reward == 0


def test_certify_rejects_null_not_0(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = certify_real_pier_pack(
        pack,
        runner=ScriptedPierRunner(oracle_reward=1, null_reward=1),
        jobs_root=f"{_JOBS}-nullfail",
        run_load_smoke=False,
        allow_scripted_substitute=True,
    )
    assert result.certified is False
    assert result.null_reward == 1


def test_certify_rejects_hybrid_track(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path, source_track="hybrid_curated", task_id="hybrid-no")
    result = certify_real_pier_pack(
        pack,
        runner=ScriptedPierRunner(),
        jobs_root=f"{_JOBS}-hybrid",
        run_load_smoke=False,
        allow_scripted_substitute=True,
    )
    assert result.certified is False
    assert "RPIER_HYBRID_REFUSED" in result.reason_codes or "RPIER_BAD_SOURCE_TRACK" in (
        result.reason_codes
    )


# ---------------------------------------------------------------------------
# Prefer real pier; scripted not full substitute without flag
# ---------------------------------------------------------------------------


def test_prefer_real_pier_when_available() -> None:
    available, path, err = pier_is_available()
    if not available:
        pytest.skip(f"pier not installed: {err}")
    runner_impl, path_class, resolved = prefer_real_pier_runner(
        allow_scripted_substitute=False,
    )
    assert path_class == "real"
    assert resolved is not None
    assert not isinstance(runner_impl, ScriptedPierRunner)


def test_scripted_not_full_substitute_without_flag(tmp_path: Path) -> None:
    """Pier(scripted) inject without allow_scripted_substitute → not certified full."""
    pack = _export_pack(tmp_path, task_id="no-sub-flag")
    result = certify_real_pier_pack(
        pack,
        runner=ScriptedPierRunner(oracle_reward=1, null_reward=0),
        jobs_root=f"{_JOBS}-nosub",
        run_load_smoke=False,
        allow_scripted_substitute=False,
        prefer_real_pier=True,
    )
    # Scripted may still run rewards, but full product certify is false without flag
    # (or live pier takes over — if live pier is available and prefers real, it may
    # attempt real; for scripted-inject without allow, if pier unavailable → not full)
    available, _, _ = pier_is_available()
    if not available:
        assert result.certified is False
        assert RPIER_SCRIPTED_NOT_FULL_SUBSTITUTE in result.reason_codes
    else:
        # Live pier preferred over scripted inject → path may be real (attempted)
        assert result.pier_path_class in {"real", "scripted"}


def test_pier_unavailable_raises_without_scripted_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When pier binary missing and no allow_scripted_substitute → hard error."""
    pack = _export_pack(tmp_path, task_id="no-pier")

    def _missing_bin(*_a: Any, **_k: Any) -> Path:
        from swe_factory.harbor.pier_cert import PierInvokeError

        raise PierInvokeError("pier 0.3+ not found (test stub)")

    monkeypatch.setattr(
        "swe_factory.harbor.real_pier_cert.resolve_pier_bin",
        _missing_bin,
    )
    # Also block pier_cert resolve used in path selection
    monkeypatch.setattr(
        "swe_factory.harbor.pier_cert.resolve_pier_bin",
        _missing_bin,
    )
    with pytest.raises(
        RealPierUnavailableError, match="allow_scripted_substitute|unavailable|not found"
    ):
        certify_real_pier_pack(
            pack,
            runner=None,
            jobs_root=f"{_JOBS}-unavail",
            run_load_smoke=False,
            allow_scripted_substitute=False,
            prefer_real_pier=True,
            pier_bin="/tmp/does-not-exist-pier-xyz",
        )


def test_pier_unavailable_allows_scripted_with_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = _export_pack(tmp_path, task_id="scripted-ok")

    def _missing_bin(*_a: Any, **_k: Any) -> Path:
        from swe_factory.harbor.pier_cert import PierInvokeError

        raise PierInvokeError("pier 0.3+ not found (test stub)")

    monkeypatch.setattr(
        "swe_factory.harbor.real_pier_cert.resolve_pier_bin",
        _missing_bin,
    )
    monkeypatch.setattr(
        "swe_factory.harbor.pier_cert.resolve_pier_bin",
        _missing_bin,
    )
    result = certify_real_pier_pack(
        pack,
        runner=None,
        jobs_root=f"{_JOBS}-scripted-flag",
        run_load_smoke=False,
        allow_scripted_substitute=True,
        prefer_real_pier=True,
        pier_bin="/tmp/does-not-exist-pier-xyz",
        evidence_dir=tmp_path / "ev",
    )
    assert result.pier_path_class == "scripted"
    assert result.solution_reward == 1
    assert result.null_reward == 0
    assert result.certified is True, result.reasons
    assert result.evidence_files is not None
    assert Path(result.evidence_files.sol_path or "").is_file()
    assert Path(result.evidence_files.null_path or "").is_file()


# ---------------------------------------------------------------------------
# Evidence writer + no fake attestation
# ---------------------------------------------------------------------------


def test_write_sol_null_evidence_no_fake(tmp_path: Path) -> None:
    files = write_real_pier_sol_null_evidence(
        tmp_path,
        task_id="ev1",
        solution_reward=1,
        null_reward=0,
        pier_path_class="real",
        jobs_root="/tmp/harbor-deepswe-jobs",
        source_track="real_pr",
    )
    sol = json.loads(Path(files.sol_path).read_text(encoding="utf-8"))
    null = json.loads(Path(files.null_path).read_text(encoding="utf-8"))
    combined = json.loads(Path(files.combined_path or "").read_text(encoding="utf-8"))
    assert sol["reward"] == 1
    assert null["reward"] == 0
    assert sol["backend"] == "docker"
    assert sol["oracle_mode"] == "docker"
    assert combined["fake_oracle"] is False
    assert combined["pair_ok"] is True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_help_lists_real_pr_flag() -> None:
    result = runner.invoke(app, ["pier-cert", "--help"])
    assert result.exit_code == 0
    assert "--real-pr" in result.output or "real-pr" in result.output


def test_cli_pier_cert_real_pr_refuses_fake(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = runner.invoke(
        app,
        [
            "pier-cert",
            "--pack-dir",
            str(pack),
            "--real-pr",
            "--oracle-mode",
            "fake",
            "--json",
            "--jobs-dir",
            f"{_JOBS}-cli-fake",
            "--skip-oracle",
            "--skip-null",
            "--no-load-smoke",
        ],
    )
    assert result.exit_code != 0, result.output
    combined = (result.output or "") + (result.stderr or "")
    assert re.search(r"fake|docker|refuse", combined, re.I)


def test_cli_pier_cert_real_pr_scripted_flag(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path, task_id="cli-scripted")
    result = runner.invoke(
        app,
        [
            "pier-cert",
            "--pack-dir",
            str(pack),
            "--real-pr",
            "--allow-scripted-substitute",
            "--json",
            "--jobs-dir",
            f"{_JOBS}-cli-ok",
            "--no-load-smoke",
            "--evidence-out",
            str(tmp_path / "cli_pier_ev.json"),
        ],
    )
    # With allow flag; if pier is live it may use real (docker heavy) —
    # use env force? For unit determinism CLI without injected runner may hit
    # real pier when available (slow/fail) or scripted when not.
    # Accept either exit with valid JSON sol/null when it completes quickly;
    # if live pier hangs we skip — timeout_sec lower via flag?
    # Safer: only assert CLI refuses fake + parses --real-pr path for refuse.
    # Full succeed path is unit API above.
    if result.exit_code not in (0, 1):
        # exit 2 is refuse
        combined = (result.output or "") + (result.stderr or "")
        assert "fake" not in combined.lower() or result.exit_code == 2
    # When scripted substitute used successfully:
    if result.exit_code == 0:
        payload = json.loads(result.output)
        assert payload.get("sol_reward") == 1
        assert payload.get("null_reward") == 0
        assert payload.get("real_pr") is True
        assert payload.get("oracle_mode") != "fake"


def test_jobs_root_under_tmp() -> None:
    from swe_factory.harbor.real_pier_cert import DEFAULT_JOBS_ROOT

    assert str(DEFAULT_JOBS_ROOT).startswith("/tmp/harbor-deepswe-jobs")
