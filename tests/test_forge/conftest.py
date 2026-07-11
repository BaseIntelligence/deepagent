"""Forge test fixtures for isolated non-production authority state."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from swe_forge.forge import receipt_authority
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
    isolated_teacher_receipt_authority: Path,
) -> None:
    """Fixture receipt helpers transport through one child, never a parent signer."""
    receipt_helpers.configure_test_authority()
    yield
    receipt_helpers.close_test_authority()
