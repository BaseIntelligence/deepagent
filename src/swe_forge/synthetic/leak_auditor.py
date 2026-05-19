"""Static leak checks for synthetic feature-deletion tasks."""

from __future__ import annotations

import re
from pathlib import Path

from swe_forge.synthetic.models import LeakAuditResult
from swe_forge.synthetic.sanitizer import is_leaky_artifact


def _added_lines(patch: str) -> list[str]:
    lines: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++") or not line.startswith("+"):
            continue
        content = line[1:].strip()
        if len(content) >= 24:
            lines.append(content)
    return lines


def audit_patch_leaks(
    repo_root: Path | str,
    *,
    oracle_patch: str,
    forbidden_filenames: list[str] | None = None,
) -> LeakAuditResult:
    """Detect obvious answer leaks in a prepared repository tree."""
    root = Path(repo_root).resolve()
    findings: list[str] = []
    forbidden_filenames = forbidden_filenames or [
        "removed.patch",
        "oracle.patch",
        "solution.patch",
    ]

    for path in root.rglob("*"):
        if path.name in forbidden_filenames:
            findings.append(f"Forbidden artifact present: {path.relative_to(root)}")
        if is_leaky_artifact(path):
            findings.append(
                f"Leaky build/cache artifact present: {path.relative_to(root)}"
            )

    snippets = _added_lines(oracle_patch)[:50]
    if snippets:
        searchable_files = [
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.name not in {"patch.diff", "deletion_patch.diff"}
            and p.stat().st_size <= 1_000_000
            and p.suffix not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip"}
        ]
        for path in searchable_files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for snippet in snippets:
                normalized = re.sub(r"\s+", " ", snippet)
                if normalized and normalized in re.sub(r"\s+", " ", text):
                    findings.append(
                        f"Oracle snippet appears in {path.relative_to(root)}"
                    )
                    break

    risk = min(1.0, len(findings) / 10)
    return LeakAuditResult(passed=not findings, risk_score=risk, findings=findings)
