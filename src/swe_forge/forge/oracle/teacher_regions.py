"""Bounded source regions for teacher-authored oracle proposals.

Teacher differential and alternative-correct proposals need only replace a
candidate's recorded target region, not an entire large source file. The
materialized proposal remains a complete file before Docker executes it.
"""

from __future__ import annotations

from dataclasses import dataclass

from swe_forge.forge.models import Candidate


@dataclass(frozen=True)
class TeacherSource:
    """The prompt-sized source and deterministic full-file reconstruction data."""

    path: str
    source: str
    prefix: str
    suffix: str
    symbol: str

    @property
    def is_region(self) -> bool:
        return bool(self.prefix or self.suffix)

    def materialize(self, proposal: str) -> str:
        """Turn a proposed source region into a full target-file replacement."""
        replacement = proposal.strip("\n")
        if (
            self.source.endswith("\n")
            and replacement
            and not replacement.endswith("\n")
        ):
            replacement += "\n"
        return self.prefix + replacement + self.suffix


def select_teacher_source(
    candidate: Candidate, gold_sources: dict[str, str]
) -> TeacherSource | None:
    """Select a target region when provenance has validated source line bounds."""
    path = _primary_target(candidate.target.files, gold_sources)
    if not path:
        return None
    source = gold_sources.get(path, "")
    if not source.strip():
        return None

    region = _provenance_region(candidate, path, source)
    if region is not None:
        start, end, symbol = region
        lines = source.splitlines(keepends=True)
        return TeacherSource(
            path=path,
            source="".join(lines[start - 1 : end]),
            prefix="".join(lines[: start - 1]),
            suffix="".join(lines[end:]),
            symbol=symbol,
        )

    return TeacherSource(
        path=path,
        source=source,
        prefix="",
        suffix="",
        symbol=_target_symbol(candidate, path),
    )


def _primary_target(files: tuple[str, ...], gold_sources: dict[str, str]) -> str:
    for path in files:
        if path in gold_sources and gold_sources[path].strip():
            return path
    return next(iter(gold_sources), "")


def _target_symbol(candidate: Candidate, path: str) -> str:
    for file, symbol in zip(
        candidate.target.files, candidate.target.symbols, strict=False
    ):
        if file == path:
            return symbol
    return candidate.target.symbols[0] if candidate.target.symbols else ""


def _provenance_region(
    candidate: Candidate, path: str, source: str
) -> tuple[int, int, str] | None:
    raw_constituents = candidate.provenance.details.get("constituents")
    if not isinstance(raw_constituents, list):
        return None
    total_lines = len(source.splitlines())
    for raw in raw_constituents:
        if not isinstance(raw, dict) or raw.get("file") != path:
            continue
        start = raw.get("start_line")
        end = raw.get("end_line")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 1
            or end < start
            or end > total_lines
        ):
            continue
        symbol = raw.get("symbol")
        return start, end, str(symbol) if isinstance(symbol, str) else ""
    return None


def required_symbol(source: TeacherSource) -> str:
    """Return the method/function leaf expected in a proposal, if known."""
    return source.symbol.rsplit(".", 1)[-1] if source.symbol else ""


__all__ = ["TeacherSource", "required_symbol", "select_teacher_source"]
