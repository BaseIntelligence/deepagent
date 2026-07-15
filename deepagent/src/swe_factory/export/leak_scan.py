"""Leak scanner over exported agent trees and dataset outputs.

VAL-EXPORT-003: clean on fixture export (no gold co-location, no API keys).
VAL-HARNESS-003: dataset outputs / reports contain no raw API keys.

Also used by export fail-closed binding before publishing workspaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Agent/tree scan primitives from oracle G5
from swe_factory.oracle.docker_run import scan_agent_workspace_leak

# Known secret-ish patterns (never log matched values in full).
# VAL-LX-005 / VAL-XDEEP-006 / VAL-HARNESS-003: no raw tokens in ship/mine evidence.
_API_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-or-v1-[A-Za-z0-9]{16,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(
        r"(?:OPENROUTER_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GITHUB_TOKEN|GH_TOKEN)"
        r"\s*[=:]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}", re.IGNORECASE),
    # Classic GitHub OAuth / classic PAT tokens (never in ship tree).
    re.compile(r"\bgho_[A-Za-z0-9_]{10,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
)

_FORBIDDEN_NAME_RE = re.compile(
    r"(^|/)(gold\.patch|patch\.diff|solution\.patch|deletion_patch\.diff|"
    r"oracle_hidden|hidden_tests)($|/)",
    re.IGNORECASE,
)

_SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

_TEXT_SUFFIXES = {
    "",
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yml",
    ".yaml",
    ".toml",
    ".sh",
    ".patch",
    ".diff",
    ".js",
    ".ts",
    ".go",
    ".rs",
    ".env",
    ".example",
    ".cfg",
    ".ini",
    ".csv",
}

_MAX_FILE_BYTES = 512_000


@dataclass(frozen=True, slots=True)
class LeakScanResult:
    """Outcome of scanning an export tree or dataset directory."""

    clean: bool
    findings: tuple[str, ...] = ()
    files_scanned: int = 0
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "clean": self.clean,
            "findings": list(self.findings),
            "files_scanned": self.files_scanned,
            "details": dict(self.details),
        }


def _should_skip_dir(name: str) -> bool:
    return name in _SKIP_DIR_NAMES


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    if root.is_file():
        return [root]
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Skip paths under ignored dir segments
        if any(_should_skip_dir(part) for part in path.parts):
            continue
        files.append(path)
    return files


def _redact(snippet: str, *, max_len: int = 48) -> str:
    cleaned = re.sub(r"\s+", " ", snippet.strip())
    if len(cleaned) > max_len:
        return cleaned[:max_len] + "…"
    return cleaned


def scan_text_for_secrets(text: str, *, rel: str) -> list[str]:
    """Return findings for API keys / secret assignments in text."""
    findings: list[str] = []
    for pattern in _API_KEY_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append(f"secret-like pattern in {rel}: {_redact(match.group(0))!r}")
            break  # one finding per file from keys is enough
    return findings


def scan_path_for_leaks(
    path: Path,
    *,
    root: Path,
    gold_patch: str | None = None,
) -> list[str]:
    """Scan a single file path for path-name and content leaks."""
    findings: list[str] = []
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)

    normalized = rel.replace("\\", "/")
    if _FORBIDDEN_NAME_RE.search(normalized) and (
        path.name
        in {
            "gold.patch",
            "patch.diff",
            "solution.patch",
            "deletion_patch.diff",
        }
        or "oracle_hidden" in normalized.lower()
    ):
        findings.append(f"forbidden artifact path: {rel}")
        return findings

    try:
        size = path.stat().st_size
    except OSError:
        return findings
    if size > _MAX_FILE_BYTES:
        return findings
    if path.suffix.lower() not in _TEXT_SUFFIXES and path.suffix != "":
        # Still scan path names above; skip binary content
        return findings
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings

    findings.extend(scan_text_for_secrets(text, rel=rel))

    # Unified diffs in agent-visible auxiliary files (not tests/ source itself)
    if (
        "diff --git" in text
        and "+++ " in text
        and (
            path.suffix.lower() in {".patch", ".diff"}
            or path.name
            in {
                "hint.md",
                "solution.md",
                "answer.md",
                "notes.txt",
                "problem_statement.md",
                "task_meta.agent.json",
                "README.md",
                "prompt.md",
            }
        )
    ):
        findings.append(f"unified diff content in agent-visible file: {rel}")

    if gold_patch:
        # Gold body markers in prompt-like files
        markers = [
            ln.lstrip("+").strip()
            for ln in gold_patch.splitlines()
            if ln.startswith("+") and not ln.startswith("+++") and len(ln) > 8
        ][:5]
        promptish = path.name in {
            "problem_statement.md",
            "task_meta.agent.json",
            "README.md",
            "prompt.md",
            "hint.md",
            "notes.txt",
        }
        if promptish:
            for marker in markers:
                if marker and marker in text:
                    findings.append(f"gold body marker in agent file {rel}: {_redact(marker)!r}")
                    break
    return findings


def scan_export_tree(
    root: Path | str,
    *,
    gold_patches: dict[str, str] | None = None,
    agent_workspace_subdir: str = "tasks",
) -> LeakScanResult:
    """Scan an export/dataset directory for gold co-location and secrets.

    - Walks all text-ish files under ``root``.
    - For each ``tasks/<id>/`` agent workspace, also runs G5-style
      :func:`scan_agent_workspace_leak` when a matching gold patch is supplied
      or when a sibling full record is not needed.
    - ``tasks.jsonl`` gold fields are *not* treated as leaks (internal record);
      only agent-visible paths and secret patterns are.
    """
    base = Path(root)
    findings: list[str] = []
    scanned = 0
    gold_map = dict(gold_patches or {})

    for path in _iter_files(base):
        scanned += 1
        rel = str(path.relative_to(base)) if path.is_relative_to(base) else str(path)
        # Allow full task records in tasks.jsonl / gate_audit / audit files
        if path.name in {"tasks.jsonl", "gate_audit.jsonl", "ledger.jsonl"}:
            # Still scan for raw API keys
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            findings.extend(scan_text_for_secrets(text, rel=rel))
            continue

        findings.extend(scan_path_for_leaks(path, root=base))

    # Per-agent-workspace G5 scan
    agent_root = base / agent_workspace_subdir if agent_workspace_subdir else base
    if agent_root.is_dir():
        for child in sorted(agent_root.iterdir()):
            if not child.is_dir():
                continue
            gold = gold_map.get(child.name)
            g5 = scan_agent_workspace_leak(child, gold_patch=gold or "")
            for item in g5:
                findings.append(f"{child.name}: {item}")
            # Always forbid gold.patch sitting next to agent tree (even with empty gold)
            for forbidden in (
                "gold.patch",
                "patch.diff",
                "solution.patch",
                "deletion_patch.diff",
            ):
                if (child / forbidden).exists():
                    findings.append(f"forbidden agent artifact: {child.name}/{forbidden}")

    # De-dupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for item in findings:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return LeakScanResult(
        clean=len(unique) == 0,
        findings=tuple(unique),
        files_scanned=scanned,
        details={"root": str(base)},
    )


__all__ = [
    "LeakScanResult",
    "scan_export_tree",
    "scan_path_for_leaks",
    "scan_text_for_secrets",
]
