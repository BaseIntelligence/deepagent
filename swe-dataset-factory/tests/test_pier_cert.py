"""Pier cert adapter (VAL-PIER-001..005) — offline unit tests + evidence parsers.

Live pier/docker dual-runs remain optional / integration; this suite pins:
- pack structural load hooks
- oracle reward parseable == 1
- null/nop reward == 0
- fake oracle_mode refused
- agent isolation re-check
- jobs under /tmp/harbor-deepswe-jobs*
- clear error surfaces for missing rewards / bad jobs roots
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.harbor.deepswe_cert import FakeBackendRejected
from swe_factory.harbor.export_pack import export_harbor_pack
from swe_factory.harbor.pier_cert import (
    DEFAULT_JOBS_ROOT,
    PierCertError,
    ScriptedPierRunner,
    append_pier_audit,
    certify_pier_pack,
    ensure_jobs_root,
    find_latest_trial_dir,
    find_reward_jsons,
    parse_pier_job_result,
    parse_reward_json,
    refuse_fake_oracle_mode,
    resolve_pier_bin,
    structural_load_pack,
    write_pier_evidence,
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


def _spec(
    *,
    task_id: str = "deepswe-pier-demo",
    repository_url: str = _REAL_URL,
    base_commit: str = _REAL_SHA,
) -> HarborPackSpec:
    return HarborPackSpec.model_validate(
        {
            "task_id": task_id,
            "instruction_md": "Fix multi-file bug via Pier-loadable pack.\n",
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
            "environment_dockerfile": "FROM python:3.12-slim\nWORKDIR /app\n",
            "tests_dockerfile": "FROM deepswe-agent:local\nCOPY test.sh /tests/test.sh\n",
        }
    )


def _export_pack(tmp_path: Path, **kwargs: Any) -> Path:
    pack = export_harbor_pack(
        _spec(**kwargs), dest=tmp_path / "tasks" / kwargs.get("task_id", "deepswe-pier-demo")
    )
    return pack.pack_dir


def _seed_reward_job(
    jobs_root: Path,
    *,
    agent: str,
    reward: int | float,
    job_name: str = "seeded",
) -> Path:
    """Plant a pier-like job tree with verifier/reward.json for parser tests."""
    trial = jobs_root / job_name / f"trial-{agent}"
    verifier = trial / "verifier"
    verifier.mkdir(parents=True, exist_ok=True)
    payload = {
        "reward": reward,
        "f2p_total": 4,
        "f2p_passed": 4 if reward == 1 else 0,
        "p2p_total": 3,
        "p2p_passed": 3,
        "f2p": 1.0 if reward == 1 else 0.0,
        "p2p": 1.0,
        "partial": float(reward),
    }
    (verifier / "reward.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (trial / "result.json").write_text(
        json.dumps(
            {
                "verifier_result": {"rewards": dict(payload)},
                "agent_info": {"name": agent},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return trial


# ---------------------------------------------------------------------------
# Evidence parsers
# ---------------------------------------------------------------------------


def test_parse_reward_json_object(tmp_path: Path) -> None:
    path = tmp_path / "reward.json"
    path.write_text(json.dumps({"reward": 1, "f2p_total": 2}) + "\n", encoding="utf-8")
    evidence = parse_reward_json(path, agent="oracle")
    assert evidence.parse_ok is True
    assert evidence.reward == 1
    assert evidence.agent == "oracle"
    assert evidence.path == str(path)


def test_parse_reward_json_null_zero(tmp_path: Path) -> None:
    path = tmp_path / "reward.json"
    path.write_text(json.dumps({"reward": 0.0}) + "\n", encoding="utf-8")
    evidence = parse_reward_json(path, agent="nop")
    assert evidence.parse_ok is True
    assert evidence.reward == 0.0


def test_parse_reward_json_bare_number(tmp_path: Path) -> None:
    path = tmp_path / "reward.json"
    path.write_text("1\n", encoding="utf-8")
    evidence = parse_reward_json(path)
    assert evidence.parse_ok is True
    assert evidence.reward == 1


def test_parse_reward_missing_surfaces_error(tmp_path: Path) -> None:
    evidence = parse_reward_json(tmp_path / "nope.json", agent="oracle")
    assert evidence.parse_ok is False
    assert evidence.reward is None
    assert any("missing" in e for e in evidence.errors)


def test_parse_reward_empty_surfaces_error(tmp_path: Path) -> None:
    path = tmp_path / "reward.json"
    path.write_text("", encoding="utf-8")
    evidence = parse_reward_json(path)
    assert evidence.parse_ok is False
    assert any("empty" in e for e in evidence.errors)


def test_find_reward_jsons_prefers_verifier(tmp_path: Path) -> None:
    trial = _seed_reward_job(tmp_path, agent="oracle", reward=1)
    found = find_reward_jsons(tmp_path)
    assert found
    assert found[0].parent.name == "verifier"
    assert find_latest_trial_dir(tmp_path) == trial or find_latest_trial_dir(tmp_path) is not None


def test_parse_pier_job_result_from_seeded_tree(tmp_path: Path) -> None:
    _seed_reward_job(tmp_path, agent="oracle", reward=1, job_name="oracle-job")
    evidence = parse_pier_job_result(tmp_path, agent="oracle")
    assert evidence.parse_ok is True
    assert evidence.reward == 1


def test_parse_pier_job_result_from_trial_result_json_only(tmp_path: Path) -> None:
    trial = tmp_path / "ts" / "trial-x"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": 0}}}) + "\n",
        encoding="utf-8",
    )
    evidence = parse_pier_job_result(tmp_path, agent="nop")
    assert evidence.parse_ok is True
    assert evidence.reward == 0


def test_parse_pier_job_result_from_job_stats(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "stats": {
                    "evals": {
                        "oracle__adhoc": {
                            "metrics": [{"reward": 1.0, "f2p": 1.0}],
                        }
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = parse_pier_job_result(tmp_path, agent="oracle")
    assert evidence.parse_ok is True
    assert evidence.reward == 1.0


# ---------------------------------------------------------------------------
# Jobs root + pier binary resolve
# ---------------------------------------------------------------------------


def test_ensure_jobs_root_default() -> None:
    root = ensure_jobs_root()
    assert str(root).startswith("/tmp/harbor-deepswe-jobs")
    assert root.is_dir()


def test_ensure_jobs_root_suffix_allowed(tmp_path: Path) -> None:
    # Use real /tmp path with allowed prefix
    target = Path("/tmp/harbor-deepswe-jobs-test-pier")
    root = ensure_jobs_root(target)
    assert root.is_dir()
    assert "harbor-deepswe-jobs" in str(root)


def test_ensure_jobs_root_rejects_unsafe(tmp_path: Path) -> None:
    with pytest.raises(PierCertError, match="harbor-deepswe-jobs"):
        ensure_jobs_root(tmp_path / "not-allowed")


def test_resolve_pier_bin_missing_raises() -> None:
    with pytest.raises(PierCertError, match="not found|not executable"):
        resolve_pier_bin("/tmp/does-not-exist-pier-bin-xyz")


# ---------------------------------------------------------------------------
# VAL-PIER-004 refuse fake
# ---------------------------------------------------------------------------


def test_refuse_fake_oracle_mode() -> None:
    with pytest.raises(FakeBackendRejected, match="fake|docker|PIER"):
        refuse_fake_oracle_mode("fake", certified=True)


def test_refuse_fake_on_deepswe_path() -> None:
    with pytest.raises(FakeBackendRejected):
        refuse_fake_oracle_mode("fake", certified=False, pack_or_dest="datasets/deepswe_v1/x")


def test_certify_rejects_fake_mode(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    with pytest.raises(FakeBackendRejected):
        certify_pier_pack(
            pack,
            runner=ScriptedPierRunner(),
            jobs_root="/tmp/harbor-deepswe-jobs-unit",
            oracle_mode="fake",
            run_load_smoke=False,
        )


# ---------------------------------------------------------------------------
# VAL-PIER-001 structural load
# ---------------------------------------------------------------------------


def test_structural_load_pack_tree(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    hooks = structural_load_pack(pack, run_load_smoke=False)
    assert hooks.required_relpaths_ok is True
    assert hooks.missing_relpaths == ()
    assert hooks.pier_job_prefix.startswith("/tmp/harbor-deepswe-jobs")


# ---------------------------------------------------------------------------
# VAL-PIER-002 / 003 scripted oracle=1 null=0
# ---------------------------------------------------------------------------


def test_certify_pier_oracle1_null0_scripted(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    jobs = Path("/tmp/harbor-deepswe-jobs-unit-ok")
    runner_impl = ScriptedPierRunner(oracle_reward=1, null_reward=0)
    evidence_path = tmp_path / "pier_evidence.json"
    audit_path = tmp_path / "pier_audit.jsonl"
    result = certify_pier_pack(
        pack,
        runner=runner_impl,
        jobs_root=jobs,
        oracle_mode="docker",
        run_load_smoke=False,
        evidence_out=evidence_path,
        audit_out=audit_path,
    )
    assert result.certified is True, result.reasons
    assert result.disposition == "accept"
    assert result.oracle_run is not None
    assert result.null_run is not None
    assert result.oracle_run.reward.reward == 1
    assert result.null_run.reward.reward == 0
    assert result.oracle_run.reward.parse_ok is True
    assert result.null_run.reward.parse_ok is True
    assert result.isolation.clean is True
    assert result.backend == "docker"
    assert result.oracle_mode == "docker"
    assert "PIER_ORACLE_REWARD_1" in result.reason_codes
    assert "PIER_NULL_REWARD_0" in result.reason_codes
    assert str(result.jobs_root).startswith("/tmp/harbor-deepswe-jobs")
    assert "oracle" in runner_impl.call_log
    assert "nop" in runner_impl.call_log

    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert data["sol_reward"] == 1
    assert data["null_reward"] == 0
    assert data["isolation_status"] == "clean"
    assert data["backend"] == "docker"

    rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["sol"] == 1
    assert rows[0]["null"] == 0


def test_certify_rejects_when_oracle_not_1(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = certify_pier_pack(
        pack,
        runner=ScriptedPierRunner(oracle_reward=0, null_reward=0),
        jobs_root="/tmp/harbor-deepswe-jobs-unit-oracfail",
        run_load_smoke=False,
    )
    assert result.certified is False
    assert result.oracle_run is not None
    assert result.oracle_run.reward.reward == 0
    assert "PIER_ORACLE_REWARD_NOT_1" in result.reason_codes


def test_certify_rejects_when_null_not_0(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = certify_pier_pack(
        pack,
        runner=ScriptedPierRunner(oracle_reward=1, null_reward=1),
        jobs_root="/tmp/harbor-deepswe-jobs-unit-nullfail",
        run_load_smoke=False,
    )
    assert result.certified is False
    assert result.null_run is not None
    assert result.null_run.reward.reward == 1
    assert "PIER_NULL_REWARD_NOT_0" in result.reason_codes


def test_certify_surfaces_unparseable_oracle_reward(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = certify_pier_pack(
        pack,
        runner=ScriptedPierRunner(force_missing_reward=True),
        jobs_root="/tmp/harbor-deepswe-jobs-unit-missing",
        run_load_smoke=False,
    )
    assert result.certified is False
    assert "PIER_ORACLE_REWARD_UNPARSEABLE" in result.reason_codes or any(
        "not parseable" in r for r in result.reasons
    )
    assert any(result.reasons)  # errors surface clearly


# ---------------------------------------------------------------------------
# VAL-PIER-005 isolation re-check
# ---------------------------------------------------------------------------


def test_certify_rejects_isolation_leak(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    (pack / "environment" / "solution.patch").write_text("leak\n", encoding="utf-8")
    result = certify_pier_pack(
        pack,
        runner=ScriptedPierRunner(),
        jobs_root="/tmp/harbor-deepswe-jobs-unit-isol",
        run_load_smoke=False,
    )
    assert result.certified is False
    assert result.isolation.clean is False
    assert C.G5_LEAK in result.reason_codes


def test_certify_rejects_bad_meta(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path, repository_url="https://example.com/fake.git", task_id="badmeta")
    result = certify_pier_pack(
        pack,
        runner=ScriptedPierRunner(),
        jobs_root="/tmp/harbor-deepswe-jobs-unit-meta",
        run_load_smoke=False,
    )
    assert result.certified is False
    assert result.pack_meta.real_url_ok is False


# ---------------------------------------------------------------------------
# Helpers / write evidence
# ---------------------------------------------------------------------------


def test_write_and_append_helpers(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = certify_pier_pack(
        pack,
        runner=ScriptedPierRunner(),
        jobs_root="/tmp/harbor-deepswe-jobs-unit-helpers",
        run_load_smoke=False,
    )
    path = write_pier_evidence(tmp_path / "ev.json", result)
    assert path.is_file()
    append_pier_audit(tmp_path / "a.jsonl", result)


def test_missing_pack_raises() -> None:
    with pytest.raises(PierCertError, match="not found"):
        certify_pier_pack(
            "/tmp/does-not-exist-pier-pack-xyz",
            runner=ScriptedPierRunner(),
            jobs_root="/tmp/harbor-deepswe-jobs-unit-missing-pack",
            run_load_smoke=False,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_help_lists_pier_cert() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "pier-cert" in result.output


def test_cli_pier_cert_refuses_fake(tmp_path: Path) -> None:
    pack = _export_pack(tmp_path)
    result = runner.invoke(
        app,
        [
            "pier-cert",
            "--pack-dir",
            str(pack),
            "--oracle-mode",
            "fake",
            "--json",
            "--jobs-dir",
            "/tmp/harbor-deepswe-jobs-cli-fake",
            "--skip-oracle",
            "--skip-null",
            "--no-load-smoke",
        ],
    )
    assert result.exit_code != 0, result.output
    combined = (result.output or "") + (result.stderr or "")
    assert re.search(r"fake|docker|refuse", combined, re.I)


def test_cli_pier_cert_requires_pack_dir() -> None:
    result = runner.invoke(app, ["pier-cert", "--json"])
    assert result.exit_code != 0
    combined = (result.output or "") + (result.stderr or "")
    assert "pack-dir" in combined.lower() or "pack" in combined.lower()


def test_default_jobs_root_constant() -> None:
    assert str(DEFAULT_JOBS_ROOT) == "/tmp/harbor-deepswe-jobs"
