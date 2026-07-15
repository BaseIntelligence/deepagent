"""Unit tests for DeepAgent license allow/deny gate (VAL-MINE-003)."""

from __future__ import annotations

import pytest

from swe_factory.sources.license_gate import (
    LicenseGateError,
    assert_permissive_license,
    classify_license,
    is_copyleft,
    is_permissive,
)


@pytest.mark.parametrize(
    "label",
    [
        "MIT",
        "Apache-2.0",
        "Apache 2.0",
        "BSD-3-Clause",
        "BSD-2-Clause",
        "ISC",
        "Unlicense",
        "MPL-2.0",
        "0BSD",
        "BSL-1.0",
    ],
)
def test_permissive_licenses_allowed(label: str) -> None:
    decision = classify_license(label)
    assert decision.permitted is True
    assert is_permissive(label) is True
    assert is_copyleft(label) is False
    ok = assert_permissive_license(label, repo="owner/demo")
    assert ok.permitted is True


@pytest.mark.parametrize(
    "label",
    [
        "GPL-3.0",
        "GPL-2.0",
        "GPLv3",
        "AGPL-3.0",
        "LGPL-2.1",
        "copyleft",
        "SSPL-1.0",
        "GNU Affero General Public License v3",
        "MIT OR GPL-3.0",  # dual with GPL option still fails closed
    ],
)
def test_copyleft_licenses_rejected(label: str) -> None:
    decision = classify_license(label)
    assert decision.permitted is False
    assert decision.reason_code == "license_copyleft_rejected"
    assert is_copyleft(label) is True
    with pytest.raises(LicenseGateError, match="copyleft|license") as excinfo:
        assert_permissive_license(label, repo="owner/gpl-repo")
    assert excinfo.value.reason_code == "license_copyleft_rejected"


def test_missing_license_fail_closed() -> None:
    decision = classify_license("")
    assert decision.permitted is False
    assert decision.reason_code == "license_missing_rejected"
    with pytest.raises(LicenseGateError, match="missing"):
        assert_permissive_license(None)


def test_unknown_license_fail_closed() -> None:
    decision = classify_license("Proprietary-Internal-Only")
    assert decision.permitted is False
    assert decision.reason_code == "license_unknown_rejected"
    with pytest.raises(LicenseGateError):
        assert_permissive_license("Proprietary-Internal-Only")
