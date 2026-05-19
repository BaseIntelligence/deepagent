"""Models for synthetic SWE-Forge tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PythonFeatureDeletion:
    """Patch pair produced by deleting a Python feature body."""

    source_file: Path
    symbol: str
    deletion_patch: str
    oracle_patch: str
    original_source: str
    mutated_source: str


@dataclass(frozen=True)
class LeakAuditResult:
    """Static audit result for synthetic task leakage."""

    passed: bool
    risk_score: float
    findings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SanitizerResult:
    """Result of removing potentially leaky build/cache artifacts."""

    removed_paths: list[Path] = field(default_factory=list)
    skipped_paths: list[Path] = field(default_factory=list)
