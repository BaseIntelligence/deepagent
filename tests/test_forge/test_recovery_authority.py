"""Durability contracts for the machine-global alternate-recovery authority."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from swe_forge.forge.recovery_authority import (
    RecoveryAttemptAuthority,
    RecoveryAuthorityError,
    default_authority_root,
)


_IDENTITY = "mahmoud-boltons__bug_combination__7bb4e61cc98c"


def test_default_authority_root_is_host_global() -> None:
    """The default cannot vary with HOME or another invoking user."""
    assert default_authority_root() == Path("/var/lib/swe_forge/recovery-authority")


def test_new_nested_root_fsyncs_created_directories_bottom_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh authority roots durably record every created ancestor before use."""
    first_existing_parent = tmp_path / "machine-state"
    first_existing_parent.mkdir()
    root = first_existing_parent / "nested" / "authority" / "records"
    fsync_paths: list[Path] = []
    original_fsync = os.fsync

    def tracked_fsync(descriptor: int) -> None:
        fsync_paths.append(Path(os.readlink(f"/proc/self/fd/{descriptor}")))
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", tracked_fsync)

    RecoveryAttemptAuthority(root, _IDENTITY)

    assert fsync_paths == [
        first_existing_parent,
        root,
        root.parent,
        root.parent.parent,
        first_existing_parent,
        first_existing_parent,
    ]


def test_parent_fsync_failure_aborts_before_claim_or_lock_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An undurable root never becomes an authority that can write a claim."""
    first_existing_parent = tmp_path / "machine-state"
    first_existing_parent.mkdir()
    root = first_existing_parent / "nested" / "authority"
    observed_paths: list[Path] = []
    parent_fsyncs = 0
    original_fsync = os.fsync

    def fail_first_parent_fsync(descriptor: int) -> None:
        nonlocal parent_fsyncs
        path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        observed_paths.append(path)
        if path == first_existing_parent:
            parent_fsyncs += 1
            if parent_fsyncs == 2:
                raise OSError("injected parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_parent_fsync)

    with pytest.raises(OSError, match="injected parent fsync failure"):
        RecoveryAttemptAuthority(root, _IDENTITY)

    assert observed_paths == [
        first_existing_parent,
        root,
        root.parent,
        first_existing_parent,
    ]
    assert not (root / f"{_IDENTITY}.json").exists()
    assert not (root / f"{_IDENTITY}.lock").exists()
    with pytest.raises(
        RecoveryAuthorityError, match="incomplete durable initialization"
    ):
        RecoveryAttemptAuthority(root, _IDENTITY)


def test_authority_root_rejects_symlinked_ancestor_without_escape(
    tmp_path: Path,
) -> None:
    """Creation cannot be redirected through an intermediate symlink."""
    containing = tmp_path / "containing"
    containing.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = containing / "alias"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RecoveryAuthorityError, match="contains a symlink"):
        RecoveryAttemptAuthority(alias / "nested" / "authority", _IDENTITY)

    assert not (outside / "nested").exists()


def test_authority_root_rejects_final_symlink_without_escape(tmp_path: Path) -> None:
    """A symlinked final authority root cannot host authority files."""
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "authority"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RecoveryAuthorityError, match="contains a symlink"):
        RecoveryAttemptAuthority(alias, _IDENTITY)

    assert not (outside / f"{_IDENTITY}.json").exists()
    assert not (outside / f"{_IDENTITY}.lock").exists()


def test_authority_root_swap_to_symlink_aborts_claim_without_escape(
    tmp_path: Path,
) -> None:
    """Every authority operation re-pins the root before writing state."""
    root = tmp_path / "authority"
    authority = RecoveryAttemptAuthority(root, _IDENTITY)
    outside = tmp_path / "outside"
    outside.mkdir()
    root.rename(tmp_path / "relocated-authority")
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RecoveryAuthorityError, match="contains a symlink"):
        authority.claim(
            run_id="swapped-root",
            expected_current_generation_id="",
            ledger_path=tmp_path / "work" / "recovery-ledger.jsonl",
        )

    assert not (outside / f"{_IDENTITY}.json").exists()
    assert not (outside / f"{_IDENTITY}.lock").exists()


def test_authority_root_rejects_lexical_alias_before_creation(tmp_path: Path) -> None:
    """A dot-dot root spelling cannot create a second path to authority state."""
    alias = tmp_path / "containing" / ".." / "authority"

    with pytest.raises(RecoveryAuthorityError, match="path alias"):
        RecoveryAttemptAuthority(alias, _IDENTITY)

    assert not (tmp_path / "authority").exists()


def test_existing_root_is_reused_without_creation_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing roots retain the normal lock and claim behavior."""
    root = tmp_path / "machine-state"
    root.mkdir()
    before = root.stat()
    fsync_calls: list[int] = []
    original_fsync = os.fsync

    def tracked_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", tracked_fsync)
    authority = RecoveryAttemptAuthority(root, _IDENTITY)

    assert root.stat().st_ino == before.st_ino
    assert fsync_calls == []

    authority.claim(
        run_id="existing-root",
        expected_current_generation_id="",
        ledger_path=tmp_path / "work" / "recovery-ledger.jsonl",
    )
    with pytest.raises(RecoveryAuthorityError, match="already consumed"):
        RecoveryAttemptAuthority(root, _IDENTITY).claim(
            run_id="existing-root-replay",
            expected_current_generation_id="",
            ledger_path=tmp_path / "replay" / "recovery-ledger.jsonl",
        )


def test_claim_is_exclusive_durable_and_survives_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persisted claim consumes the identity even when its process crashes."""
    fsync_calls: list[int] = []
    original_fsync = os.fsync

    def tracked_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", tracked_fsync)
    authority = RecoveryAttemptAuthority(tmp_path / "machine-state", _IDENTITY)

    claimed = authority.claim(
        run_id="alternate-first",
        expected_current_generation_id="previous-generation",
        ledger_path=tmp_path / "first-work" / "recovery-ledger.jsonl",
    )

    assert claimed.run_id == "alternate-first"
    assert claimed.state == "claimed"
    assert claimed.expected_current_generation_id == "previous-generation"
    assert fsync_calls
    persisted = json.loads(authority.path.read_text(encoding="utf-8"))
    assert persisted["state"] == "claimed"
    assert persisted["run_id"] == "alternate-first"

    restarted = RecoveryAttemptAuthority(tmp_path / "machine-state", _IDENTITY)
    with pytest.raises(RecoveryAuthorityError, match="already consumed"):
        restarted.claim(
            run_id="alternate-second",
            expected_current_generation_id="other-generation",
            ledger_path=tmp_path / "second-work" / "recovery-ledger.jsonl",
        )


def test_terminal_tombstone_migration_is_consumed_and_reconciled(
    tmp_path: Path,
) -> None:
    """The legacy terminal tombstone becomes a durable consumed authority."""
    authority = RecoveryAttemptAuthority(tmp_path / "machine-state", _IDENTITY)

    migrated = authority.migrate_terminal_tombstone(
        run_id="alternate-aa6ced470de24a51bff3adeabce207a4",
        ledger_run_id="alternate-aa6ced470de24a51bff3adeabce207a4",
        certification_run_id="alternate-aa6ced470de24a51bff3adeabce207a4",
        selected_generation_id="eb215983bd91465cbc07dce80cf5e015",
        ledger_path=tmp_path / "legacy" / "recovery-ledger.jsonl",
    )

    assert migrated.state == "consumed"
    assert migrated.terminal_state == "tombstone"
    assert migrated.ledger_run_id == migrated.run_id
    assert migrated.certification_run_id == migrated.run_id
    assert migrated.selected_generation_id == "eb215983bd91465cbc07dce80cf5e015"

    with pytest.raises(RecoveryAuthorityError, match="already consumed"):
        authority.claim(
            run_id="alternate-replay",
            expected_current_generation_id="",
            ledger_path=tmp_path / "replay" / "recovery-ledger.jsonl",
        )


def test_claimed_authority_reconciles_to_a_terminal_generation(
    tmp_path: Path,
) -> None:
    """A restart can finish reconciliation but must never create another claim."""
    authority = RecoveryAttemptAuthority(tmp_path / "machine-state", _IDENTITY)
    claim = authority.claim(
        run_id="alternate-crashed",
        expected_current_generation_id="prior",
        ledger_path=tmp_path / "work" / "recovery-ledger.jsonl",
    )

    consumed = authority.consume(
        claim,
        terminal_state="tombstone",
        certification_run_id=claim.run_id,
        ledger_run_id=claim.run_id,
        selected_generation_id="terminal-generation",
    )

    assert consumed.state == "consumed"
    assert consumed.expected_current_generation_id == "prior"
    assert consumed.selected_generation_id == "terminal-generation"
    with pytest.raises(RecoveryAuthorityError, match="already consumed"):
        authority.claim(
            run_id="alternate-retry",
            expected_current_generation_id="terminal-generation",
            ledger_path=tmp_path / "retry" / "recovery-ledger.jsonl",
        )
