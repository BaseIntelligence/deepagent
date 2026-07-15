"""License allow/deny gate for DeepSWE real-repo mining.

Fail-closed on copyleft (GPL family, AGPL, LGPL-copyleft-style, SSPL, etc.).
Permissive licenses (MIT, Apache-2.0, BSD, ISC, MPL-2.0, Unlicense, …) may
proceed. Used by discover + sanitize before any DeepSWE candidate is kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# SPDX-like identifiers and common free-text labels (case-insensitive).
_PERMISSIVE: Final[frozenset[str]] = frozenset(
    {
        "mit",
        "apache",
        "apache-2.0",
        "apache 2.0",
        "apache2",
        "apache-2",
        "bsd",
        "bsd-2-clause",
        "bsd-3-clause",
        "bsd-2",
        "bsd-3",
        "isc",
        "0bsd",
        "unlicense",
        "cc0",
        "cc0-1.0",
        "mpl-2.0",
        "mpl2",
        "mozilla public license 2.0",
        "zlib",
        "boost",
        "bsl-1.0",
        "python-2.0",
        "psf",
        "psfl",
        "wtfpl",
        "blueoak-1.0.0",
        "mit-0",
        "artistic-2.0",
    }
)

# Fatal copyleft / non-certifiable for our ship surface.
_COPYLEFT: Final[frozenset[str]] = frozenset(
    {
        "gpl",
        "gpl-2.0",
        "gpl-3.0",
        "gplv2",
        "gplv3",
        "gpl-2",
        "gpl-3",
        "agpl",
        "agpl-3.0",
        "agplv3",
        "lgpl",
        "lgpl-2.1",
        "lgpl-3.0",
        "lgplv2",
        "lgplv3",
        "copyleft",
        "sspl",
        "sspl-1.0",
        "osl-3.0",
        "epl-1.0",
        "epl-2.0",
        "cddl",
        "cddl-1.0",
        "cddl-1.1",
        "sleepycat",
        "affero",
    }
)

_REASON_COPYLEFT = "license_copyleft_rejected"
_REASON_UNKNOWN = "license_unknown_rejected"
_REASON_MISSING = "license_missing_rejected"
_REASON_OK = "license_permissive_ok"

# Patterns that imply GPL-family even when embedded in free text.
_COPYLEFT_RE = re.compile(
    r"\b(?:"
    r"a?gpl(?:v?\d+(?:\.\d+)?)?"
    r"|lgpl(?:v?\d+(?:\.\d+)?)?"
    r"|affero"
    r"|sspl"
    r"|copyleft"
    r")\b",
    re.IGNORECASE,
)

_PERMISSIVE_RE = re.compile(
    r"\b(?:"
    r"mit(?:-0)?"
    r"|apache(?:[-\s]?2(?:\.0)?)?"
    r"|bsd(?:-[23](?:-clause)?)?"
    r"|isc"
    r"|unlicense"
    r"|cc0(?:-1\.0)?"
    r"|mpl-?2(?:\.0)?"
    r"|zlib"
    r"|bsl-?1\.0"
    r"|python-2\.0"
    r"|psf(?:l)?"
    r")\b",
    re.IGNORECASE,
)


class LicenseGateError(RuntimeError):
    """Raised when a repository license is not certifiable for DeepSWE."""

    def __init__(self, message: str, *, reason_code: str, license_raw: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.license_raw = license_raw


@dataclass(frozen=True, slots=True)
class LicenseDecision:
    """Result of checking a license label for mining eligibility."""

    permitted: bool
    normalized: str
    reason_code: str
    license_raw: str

    @property
    def is_copyleft(self) -> bool:
        return self.reason_code == _REASON_COPYLEFT


def normalize_license(value: str | None) -> str:
    """Collapse free-text / SPDX license strings for comparison."""
    if value is None:
        return ""
    cleaned = value.strip().lower()
    cleaned = cleaned.replace("_", "-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Drop trailing words like "license" or "licence" for matching.
    cleaned = re.sub(r"\blicen[cs]e\b", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    return cleaned


def classify_license(value: str | None) -> LicenseDecision:
    """Classify a license string as permissive, copyleft, unknown, or missing.

    Copyleft corruptions (``MIT OR GPL-3.0`` dual with GPL option, pure GPL
    labels, AGPL suites) are rejected. Unknown / empty are fail-closed for
    certified DeepSWE keeps.
    """
    raw = (value or "").strip()
    if not raw:
        return LicenseDecision(
            permitted=False,
            normalized="",
            reason_code=_REASON_MISSING,
            license_raw=raw,
        )

    norm = normalize_license(raw)
    # Exact known buckets first.
    if norm in _COPYLEFT or any(norm == c or norm.startswith(c + "-") for c in _COPYLEFT):
        return LicenseDecision(
            permitted=False,
            normalized=norm,
            reason_code=_REASON_COPYLEFT,
            license_raw=raw,
        )
    if norm in _PERMISSIVE:
        return LicenseDecision(
            permitted=True,
            normalized=norm,
            reason_code=_REASON_OK,
            license_raw=raw,
        )

    # Free-text heuristics: any copyleft token is fatal even alongside MIT.
    if _COPYLEFT_RE.search(raw) or _COPYLEFT_RE.search(norm):
        return LicenseDecision(
            permitted=False,
            normalized=norm,
            reason_code=_REASON_COPYLEFT,
            license_raw=raw,
        )

    if _PERMISSIVE_RE.search(raw) or _PERMISSIVE_RE.search(norm):
        return LicenseDecision(
            permitted=True,
            normalized=norm,
            reason_code=_REASON_OK,
            license_raw=raw,
        )

    return LicenseDecision(
        permitted=False,
        normalized=norm,
        reason_code=_REASON_UNKNOWN,
        license_raw=raw,
    )


def assert_permissive_license(value: str | None, *, repo: str | None = None) -> LicenseDecision:
    """Return a permitting decision or raise :class:`LicenseGateError`.

    Called by discover/sanitize before candidates are exported toward
    ``datasets/deepswe_v1``.
    """
    decision = classify_license(value)
    if decision.permitted:
        return decision
    where = f" for {repo}" if repo else ""
    if decision.reason_code == _REASON_COPYLEFT:
        raise LicenseGateError(
            f"copyleft license rejected{where}: {decision.license_raw!r} "
            f"(reason={decision.reason_code})",
            reason_code=decision.reason_code,
            license_raw=decision.license_raw,
        )
    if decision.reason_code == _REASON_MISSING:
        raise LicenseGateError(
            f"license missing{where}; refuse DeepSWE candidate (reason={decision.reason_code})",
            reason_code=decision.reason_code,
            license_raw=decision.license_raw,
        )
    raise LicenseGateError(
        f"unknown/non-permissive license rejected{where}: {decision.license_raw!r} "
        f"(reason={decision.reason_code})",
        reason_code=decision.reason_code,
        license_raw=decision.license_raw,
    )


def is_copyleft(value: str | None) -> bool:
    """True when the license is classified as copyleft."""
    return classify_license(value).is_copyleft


def is_permissive(value: str | None) -> bool:
    """True when the license is classified as allowed permissive."""
    return classify_license(value).permitted


__all__ = [
    "LicenseDecision",
    "LicenseGateError",
    "assert_permissive_license",
    "classify_license",
    "is_copyleft",
    "is_permissive",
    "normalize_license",
]
