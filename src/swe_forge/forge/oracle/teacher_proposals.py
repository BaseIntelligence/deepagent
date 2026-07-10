"""Parse fenced teacher proposals without retaining response content.

The caller consumes the returned source strings immediately to materialize a
proposal. This module never stores them in evidence or artifacts. It does,
however, preserve a trailing unclosed code fence as a received, truncated
proposal so proposal-count accounting cannot silently lose it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FencedProposal:
    """One fenced code proposal, including an explicit truncation marker."""

    content: str
    truncated: bool = False


def extract_fenced_proposals(text: str) -> list[FencedProposal]:
    """Return every fenced proposal, counting an unclosed final fence.

    A fenced response has the form `````language\n...`````; if its closing
    fence is absent, the residual content is still a received proposal and is
    marked truncated. This lets the caller classify a syntactically plausible
    but incomplete proposal as discarded rather than dropping it from totals.
    """
    proposals: list[FencedProposal] = []
    offset = 0
    while True:
        opening = text.find("```", offset)
        if opening < 0:
            break
        newline = text.find("\n", opening + 3)
        if newline < 0:
            break
        closing = text.find("```", newline + 1)
        if closing < 0:
            proposals.append(FencedProposal(text[newline + 1 :], truncated=True))
            break
        proposals.append(FencedProposal(text[newline + 1 : closing]))
        offset = closing + 3
    return proposals


__all__ = ["FencedProposal", "extract_fenced_proposals"]
