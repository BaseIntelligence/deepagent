"""Forge test fixtures for isolated machine-global authority state."""

from __future__ import annotations

from pathlib import Path

import pytest

from swe_forge.forge import receipt_authority


@pytest.fixture
def isolated_teacher_receipt_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Keep unit-test issuer keys out of the real host-global authority root."""
    root = tmp_path / "machine-state" / "teacher-receipts"
    monkeypatch.setattr(
        receipt_authority,
        "default_authority_root",
        lambda: root,
    )
    return root
