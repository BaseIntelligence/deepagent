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
