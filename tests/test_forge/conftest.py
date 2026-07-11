"""Forge test fixtures for isolated non-production authority state."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from swe_forge.forge import receipt_authority
from swe_forge.forge.oracle import alt_correct_synth, differential_synth
from swe_forge.forge.oracle import teacher_evidence
from swe_forge.forge.teacher import verify_test_transport_receipt
from tests.test_forge import receipt_helpers


@pytest.fixture
def isolated_teacher_receipt_authority(tmp_path: Path) -> Path:
    """Pin every test to a fresh non-production authority public root."""
    root = tmp_path / "forge-teacher-receipts"
    receipt_authority.initialize_test_authority_root(root)
    previous_root = receipt_authority.default_authority_root
    previous_environment = os.environ.get("SWE_FORGE_TEST_RECEIPT_AUTHORITY_ROOT")
    receipt_authority.default_authority_root = lambda: root
    os.environ["SWE_FORGE_TEST_RECEIPT_AUTHORITY_ROOT"] = str(root)
    try:
        yield root
    finally:
        receipt_authority.default_authority_root = previous_root
        if previous_environment is None:
            os.environ.pop("SWE_FORGE_TEST_RECEIPT_AUTHORITY_ROOT", None)
        else:
            os.environ["SWE_FORGE_TEST_RECEIPT_AUTHORITY_ROOT"] = previous_environment


@pytest.fixture(autouse=True)
def isolated_test_receipt_fixture_authority(
    isolated_teacher_receipt_authority: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixture receipt helpers transport through one child, never a parent signer."""
    # Fixture receipts exercise the explicit test verifier only. Production
    # verification remains hard-pinned and is never redirected by test roots.
    monkeypatch.setattr(
        teacher_evidence,
        "verify_transport_receipt",
        lambda receipt: verify_test_transport_receipt(
            receipt, root=isolated_teacher_receipt_authority
        ),
    )
    monkeypatch.setattr(
        differential_synth,
        "is_authoritative_transport_receipt",
        lambda receipt: verify_test_transport_receipt(
            receipt, root=isolated_teacher_receipt_authority
        ),
    )
    monkeypatch.setattr(
        alt_correct_synth,
        "is_authoritative_transport_receipt",
        lambda receipt: verify_test_transport_receipt(
            receipt, root=isolated_teacher_receipt_authority
        ),
    )
    receipt_helpers.configure_test_authority()
    yield
    receipt_helpers.close_test_authority()
