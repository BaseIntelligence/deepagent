"""PR filtering logic for SWE PR mining.

Provides configurable filters to exclude unsuitable PR candidates.
"""

import logging
from typing import Annotated

from pydantic import BaseModel, Field

from .enricher import EnrichedPullRequest

logger = logging.getLogger(__name__)


class FilterConfig(BaseModel):
    """Configuration for PR filtering.

    All filters are applied in sequence; a PR must pass all enabled filters.
    """

    exclude_bots: bool = True
    allowed_orgs: list[str] | None = None  # None = allow all orgs
    min_stars: Annotated[int, Field(ge=0)] = 0
    allowed_languages: list[str] = ["python"]
    max_files_changed: Annotated[int, Field(ge=1)] = 50


def filter_bot(pr: EnrichedPullRequest, exclude_bots: bool) -> bool:
    """Check if PR passes bot filter.

    Returns True if PR passes (should NOT be filtered out).
    Returns False if PR should be filtered out.

    Args:
        pr: The enriched PR to check
        exclude_bots: If True, exclude bot-authored PRs

    Returns:
        bool: True if passes filter, False if rejected
    """
    if not exclude_bots:
        return True
    if pr.is_bot:
        logger.debug(f"Filter rejected {pr.repo}#{pr.number}: bot author '{pr.user}'")
        return False
    return True


def filter_org(pr: EnrichedPullRequest, allowed_orgs: list[str] | None) -> bool:
    """Check if PR passes organization filter.

    Args:
        pr: The enriched PR to check
        allowed_orgs: List of allowed orgs, or None to allow all

    Returns:
        bool: True if passes filter, False if rejected
    """
    if allowed_orgs is None:
        return True

    org = pr.repo.split("/")[0]
    if org not in allowed_orgs:
        logger.debug(
            f"Filter rejected {pr.repo}#{pr.number}: org '{org}' not in allowed list"
        )
        return False
    return True


def filter_stars(pr: EnrichedPullRequest, min_stars: int) -> bool:
    """Check if PR passes stars filter.

    Args:
        pr: The enriched PR to check
        min_stars: Minimum stars required (0 = no filter)

    Returns:
        bool: True if passes filter, False if rejected
    """
    if min_stars <= 0:
        return True
    if pr.stars < min_stars:
        logger.debug(
            f"Filter rejected {pr.repo}#{pr.number}: stars {pr.stars} < {min_stars}"
        )
        return False
    return True


def filter_language(pr: EnrichedPullRequest, allowed_languages: list[str]) -> bool:
    """Check if PR passes language filter.

    Allows unknown language to pass (enrichment may have failed).

    Args:
        pr: The enriched PR to check
        allowed_languages: List of allowed languages (lowercase)

    Returns:
        bool: True if passes filter, False if rejected
    """
    if not allowed_languages:
        return True

    normalized = pr.language.lower()
    lang_unknown = normalized in ("", "unknown", "null")

    # Only reject known languages not in whitelist; unknown languages pass
    if not lang_unknown and normalized not in [
        lang.lower() for lang in allowed_languages
    ]:
        logger.debug(
            f"Filter rejected {pr.repo}#{pr.number}: language '{pr.language}' not in whitelist"
        )
        return False
    return True


def filter_file_count(pr: EnrichedPullRequest, max_files: int) -> bool:
    """Check if PR passes file count filter.

    Args:
        pr: The enriched PR to check
        max_files: Maximum number of files allowed

    Returns:
        bool: True if passes filter, False if rejected
    """
    if pr.files_changed > max_files:
        logger.debug(
            f"Filter rejected {pr.repo}#{pr.number}: files_changed {pr.files_changed} > {max_files}"
        )
        return False
    return True


def apply_filters(pr: EnrichedPullRequest, config: FilterConfig) -> bool:
    """Apply all configured filters to a PR.

    Filters are applied in order: bot -> org -> stars -> language -> file_count.
    A PR must pass all enabled filters to be accepted.

    Args:
        pr: The enriched PR to evaluate
        config: Filter configuration

    Returns:
        bool: True if PR passes all filters, False if rejected
    """
    if not filter_bot(pr, config.exclude_bots):
        return False

    if not filter_org(pr, config.allowed_orgs):
        return False

    if not filter_stars(pr, config.min_stars):
        return False

    if not filter_language(pr, config.allowed_languages):
        return False

    if not filter_file_count(pr, config.max_files_changed):
        return False

    logger.debug(f"Filter accepted {pr.repo}#{pr.number}")
    return True


def filter_prs(
    prs: list[EnrichedPullRequest], config: FilterConfig
) -> list[EnrichedPullRequest]:
    """Filter a batch of PRs.

    Args:
        prs: List of enriched PRs to filter
        config: Filter configuration

    Returns:
        list[EnrichedPullRequest]: PRs that passed all filters
    """
    return [pr for pr in prs if apply_filters(pr, config)]
