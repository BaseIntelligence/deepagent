"""Contract tests for the one-shot final alternate recovery controller."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest

import swe_forge.forge.alternate_recovery as alternate_recovery
from swe_forge.forge.alternate_recovery import (
    ALTERNATE_RECOVERY_TASK_ID,
    ApprovedInputManifest,
    ApprovedTree,
    UPDATE_WRAPPER_F2P_COMMAND,
    UPDATE_WRAPPER_F2P_CONTENT,
    UPDATE_WRAPPER_F2P_PATH,
    AlternateRecoveryError,
    RecoveryCertification,
    verify_original_budget,
)
from swe_forge.forge.models import OracleTestFile


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tree_digest(entries: dict[str, bytes]) -> str:
    return _sha256(
        "".join(
            f"{path}\0{_sha256(content)}\n" for path, content in sorted(entries.items())
        ).encode("utf-8")
    )


def _write_recovery_fixture(root: Path) -> tuple[Path, Path, ApprovedInputManifest]:
    workspace = root / "retained"
    workspace.mkdir(parents=True)
    root_files = {
        "workspace.yaml": b"task_id: fixed-task\n",
        "patch.diff": b"gold\n",
        "deletion_patch.diff": b"mutation\n",
        "provenance.json": b'{"details": {}}\n',
    }
    for name, content in root_files.items():
        (workspace / name).write_bytes(content)
    hidden_tests = {"test_hidden.py": b"def test_hidden():\n    assert True\n"}
    repo_files = {
        "package.py": b"VALUE = 1\n",
        "nested/data.txt": b"retained\n",
    }
    for tree, files in (
        (workspace / "tests", hidden_tests),
        (workspace / "repo", repo_files),
    ):
        for relative, content in files.items():
            destination = tree / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    budget = root / "budget.json"
    budget.write_bytes(b'{"budget_usd":1400,"spend_usd":0,"reserved_usd":0}\n')
    manifest = ApprovedInputManifest(
        manifest_id="test-approved-manifest",
        task_id="fixed-task",
        workspace_relative="retained",
        budget_relative="budget.json",
        budget_sha256=_sha256(budget.read_bytes()),
        workspace_files={
            name: _sha256(content) for name, content in root_files.items()
        },
        hidden_tests=ApprovedTree(
            path="tests",
            file_count=len(hidden_tests),
            tree_sha256=_tree_digest(hidden_tests),
        ),
        repository=ApprovedTree(
            path="repo",
            file_count=len(repo_files),
            tree_sha256=_tree_digest(repo_files),
        ),
    )
    return workspace, budget, manifest


def test_approved_input_manifest_rehydrates_only_a_verified_private_snapshot(
    tmp_path: Path,
) -> None:
    workspace, budget, manifest = _write_recovery_fixture(tmp_path)

    verified = alternate_recovery._verify_approved_recovery_inputs(
        workspace,
        budget,
        repository_root=tmp_path,
        manifest=manifest,
    )

    assert verified.manifest_id == "test-approved-manifest"
    assert verified.manifest_digest == manifest.digest
    assert verified.workspace_digests == {
        "workspace.yaml": _sha256(b"task_id: fixed-task\n"),
        "patch.diff": _sha256(b"gold\n"),
        "deletion_patch.diff": _sha256(b"mutation\n"),
        "provenance.json": _sha256(b'{"details": {}}\n'),
        "tests": _tree_digest(
            {"test_hidden.py": b"def test_hidden():\n    assert True\n"}
        ),
        "repo": _tree_digest(
            {"package.py": b"VALUE = 1\n", "nested/data.txt": b"retained\n"}
        ),
        "budget": _sha256(b'{"budget_usd":1400,"spend_usd":0,"reserved_usd":0}\n'),
    }
    assert verified.snapshot_root != workspace
    assert (verified.snapshot_root / "repo" / "package.py").read_text() == "VALUE = 1\n"

    (workspace / "repo" / "package.py").write_text("VALUE = 999\n", encoding="utf-8")
    assert (verified.snapshot_root / "repo" / "package.py").read_text() == "VALUE = 1\n"
    evidence = verified.audit_evidence()
    assert evidence["manifest_id"] == manifest.manifest_id
    assert evidence["manifest_digest"] == manifest.digest
    assert str(workspace) not in json.dumps(evidence)
    assert "VALUE = 1" not in json.dumps(evidence)
    verified.cleanup()


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda workspace, _budget: (workspace / "patch.diff").write_text(
                "changed\n", encoding="utf-8"
            ),
            id="gold-patch-changed",
        ),
        pytest.param(
            lambda workspace, _budget: (workspace / "deletion_patch.diff").unlink(),
            id="mutation-patch-missing",
        ),
        pytest.param(
            lambda workspace, _budget: (workspace / "provenance.json").write_text(
                "{}", encoding="utf-8"
            ),
            id="provenance-changed",
        ),
        pytest.param(
            lambda workspace, _budget: (workspace / "tests" / "extra.py").write_text(
                "x = 1\n", encoding="utf-8"
            ),
            id="hidden-test-extra",
        ),
        pytest.param(
            lambda workspace, _budget: (
                workspace / "repo" / "nested" / "data.txt"
            ).write_text("changed\n", encoding="utf-8"),
            id="repository-changed",
        ),
        pytest.param(
            lambda _workspace, budget: budget.write_text(
                '{"budget_usd":1400,"spend_usd":1,"reserved_usd":0}\n',
                encoding="utf-8",
            ),
            id="budget-changed",
        ),
    ],
)
def test_approved_input_manifest_rejects_tampering_before_snapshot(
    tmp_path: Path,
    mutation: Callable[[Path, Path], None],
) -> None:
    workspace, budget, manifest = _write_recovery_fixture(tmp_path)
    mutation(workspace, budget)

    with pytest.raises(AlternateRecoveryError, match="approved manifest"):
        alternate_recovery._verify_approved_recovery_inputs(
            workspace,
            budget,
            repository_root=tmp_path,
            manifest=manifest,
        )


def test_approved_input_manifest_rejects_symlinks_aliases_and_traversal(
    tmp_path: Path,
) -> None:
    workspace, budget, manifest = _write_recovery_fixture(tmp_path)
    external = tmp_path / "external.patch"
    external.write_text("gold\n", encoding="utf-8")
    (workspace / "patch.diff").unlink()
    (workspace / "patch.diff").symlink_to(external)

    with pytest.raises(AlternateRecoveryError, match="symlink"):
        alternate_recovery._verify_approved_recovery_inputs(
            workspace,
            budget,
            repository_root=tmp_path,
            manifest=manifest,
        )

    workspace, budget, manifest = _write_recovery_fixture(tmp_path / "second")
    with pytest.raises(AlternateRecoveryError, match="path alias"):
        alternate_recovery._verify_approved_recovery_inputs(
            workspace.parent / "retained" / ".." / "retained",
            budget,
            repository_root=workspace.parent,
            manifest=manifest,
        )


def test_approved_input_manifest_detects_a_target_swap_between_verification_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, budget, manifest = _write_recovery_fixture(tmp_path)
    original = alternate_recovery._read_approved_manifest_material
    calls = 0

    def read_then_swap(
        source: Path,
        progress: Path,
        repository_root: Path,
        approved_manifest: ApprovedInputManifest,
    ) -> tuple[dict[str, bytes], bytes]:
        nonlocal calls
        result = original(source, progress, repository_root, approved_manifest)
        calls += 1
        if calls == 1:
            (workspace / "repo" / "package.py").write_text(
                "VALUE = 2\n", encoding="utf-8"
            )
        return result

    monkeypatch.setattr(
        alternate_recovery, "_read_approved_manifest_material", read_then_swap
    )

    with pytest.raises(AlternateRecoveryError, match="approved manifest"):
        alternate_recovery._verify_approved_recovery_inputs(
            workspace,
            budget,
            repository_root=tmp_path,
            manifest=manifest,
        )


def test_approved_input_manifest_rejects_duplicate_casefolded_tree_paths(
    tmp_path: Path,
) -> None:
    workspace, budget, manifest = _write_recovery_fixture(tmp_path)
    (workspace / "repo" / "PACKAGE.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(AlternateRecoveryError, match="path alias"):
        alternate_recovery._verify_approved_recovery_inputs(
            workspace,
            budget,
            repository_root=tmp_path,
            manifest=manifest,
        )


def test_manifest_rejection_has_no_recovery_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, budget, manifest = _write_recovery_fixture(tmp_path)
    (workspace / "patch.diff").write_text("changed\n", encoding="utf-8")
    output = tmp_path / "output"
    work = tmp_path / "work"
    monkeypatch.setattr(alternate_recovery, "APPROVED_INPUT_MANIFEST", manifest)

    with pytest.raises(AlternateRecoveryError, match="approved manifest"):
        asyncio.run(
            alternate_recovery.run_final_alternate_recovery(
                out_dir=output,
                source_workspace=workspace,
                budget_progress=budget,
                work_root=work,
                repository_root=tmp_path,
            )
        )

    assert not output.exists()
    assert not work.exists()


def test_update_wrapper_hidden_test_is_upstream_grounded_and_named() -> None:
    """The recovered suite exercises the public wraps behavior, not implementation."""
    assert UPDATE_WRAPPER_F2P_PATH == "test_update_wrapper_wraps_basic.py"
    assert UPDATE_WRAPPER_F2P_COMMAND == (
        "python -m pytest "
        "test_update_wrapper_wraps_basic.py::"
        "test_wraps_basic_regular_function_preserves_metadata_and_wrapped"
    )
    assert "from boltons.funcutils import wraps" in UPDATE_WRAPPER_F2P_CONTENT
    assert "@wraps(source)" in UPDATE_WRAPPER_F2P_CONTENT
    assert "wrapped(3)" in UPDATE_WRAPPER_F2P_CONTENT
    assert "wrapped.__name__ == source.__name__" in UPDATE_WRAPPER_F2P_CONTENT
    assert "wrapped.__doc__ == source.__doc__" in UPDATE_WRAPPER_F2P_CONTENT
    assert "wrapped.__wrapped__ is source" in UPDATE_WRAPPER_F2P_CONTENT


def test_update_wrapper_hidden_test_is_added_without_changing_existing_suite() -> None:
    existing = [
        OracleTestFile(
            path="test_complement_bug.py",
            content="def test_complement():\n    assert True\n",
            origin="provided",
        )
    ]

    recovered = RecoveryCertification.freeze_suite(existing)

    assert [test.path for test in recovered] == [
        "test_complement_bug.py",
        UPDATE_WRAPPER_F2P_PATH,
    ]
    assert recovered[0] is existing[0]
    assert recovered[1].content == UPDATE_WRAPPER_F2P_CONTENT
    assert recovered[1].origin == "provided"


def test_update_wrapper_hidden_test_refuses_conflicting_path() -> None:
    conflicting = [
        OracleTestFile(
            path=UPDATE_WRAPPER_F2P_PATH,
            content="def test_different():\n    assert True\n",
            origin="provided",
        )
    ]

    with pytest.raises(AlternateRecoveryError, match="different content"):
        RecoveryCertification.freeze_suite(conflicting)


def test_original_budget_verification_is_exact_and_caps_incremental_attempt(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "harvest_progress.json"
    progress.write_text(
        json.dumps(
            {
                "budget_usd": 1400.0,
                "spend_usd": 1301.8979,
                "reserved_usd": 0.0,
                "status": "budget_exhausted",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    verified = verify_original_budget(progress)

    assert verified.original_budget_usd == "1400.0"
    assert verified.spent_usd == "1301.8979"
    assert verified.remaining_usd == "98.1021"
    assert verified.incremental_cap_usd == "25"


@pytest.mark.parametrize(
    "payload",
    [
        {"budget_usd": 1400.0, "spend_usd": 1400.01, "reserved_usd": 0.0},
        {"budget_usd": 1300.0, "spend_usd": 1.0, "reserved_usd": 0.0},
        {"budget_usd": 1400.0, "spend_usd": 1.0, "reserved_usd": -0.01},
        {"budget_usd": 1400.0, "spend_usd": "not-money", "reserved_usd": 0.0},
    ],
)
def test_original_budget_verification_rejects_non_authoritative_progress(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    progress = tmp_path / "harvest_progress.json"
    progress.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(AlternateRecoveryError):
        verify_original_budget(progress)


def test_certification_payload_never_marks_stale_task_as_passed() -> None:
    pending = RecoveryCertification.pending(
        run_id="alternate-run",
        previous_generation_id="stale-generation",
        task_id=ALTERNATE_RECOVERY_TASK_ID,
    ).to_dict()
    tombstone = RecoveryCertification.tombstone(
        run_id="alternate-run",
        reason="oracle rejected",
    ).to_dict()

    assert pending["state"] == "pending"
    assert pending["passed"] is False
    assert pending["task_ids"] == [ALTERNATE_RECOVERY_TASK_ID]
    assert tombstone["state"] == "tombstone"
    assert tombstone["passed"] is False
    assert tombstone["task_ids"] == []
