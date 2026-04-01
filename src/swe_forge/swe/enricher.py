"""PR enrichment logic for combining GH Archive events with GitHub API data."""

import asyncio
import logging
import re
from collections import Counter
from datetime import datetime

from pydantic import BaseModel, Field

from .gharchive import GhArchiveEvent
from .github_api import DiffTooLargeError, GitHubClient

logger = logging.getLogger(__name__)

BOT_PATTERNS = ["[bot]", "dependabot", "renovate", "github-actions", "pyup-bot"]

LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".m": "objective-c",
    ".mm": "objective-c++",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "powershell",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".sql": "sql",
    ".vue": "vue",
    ".svelte": "svelte",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".dart": "dart",
    ".pl": "perl",
    ".pm": "perl",
    ".jl": "julia",
    ".nim": "nim",
    ".cr": "crystal",
    ".d": "d",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    ".cob": "cobol",
    ".cbl": "cobol",
    ".vim": "vimscript",
    ".elm": "elm",
    ".idr": "idris",
    ".agda": "agda",
    ".purs": "purescript",
    ".dhall": "dhall",
    ".nix": "nix",
    ".lock": "lockfile",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".adoc": "asciidoc",
}


def is_bot(username: str) -> bool:
    """Check if a username belongs to a bot user."""
    return any(pattern in username.lower() for pattern in BOT_PATTERNS)


def detect_language(files: list[str]) -> str:
    """Detect primary programming language from file extensions.

    Returns the most common language or 'unknown' if no recognized extensions.
    """
    if not files:
        return "unknown"

    counts: Counter[str] = Counter()
    for filepath in files:
        parts = filepath.rsplit(".", 1)
        if len(parts) < 2:
            continue
        ext = "." + parts[1].lower()
        if ext in LANGUAGE_EXTENSIONS:
            counts[LANGUAGE_EXTENSIONS[ext]] += 1

    return counts.most_common(1)[0][0] if counts else "unknown"


def parse_files_from_diff(diff: str) -> list[str]:
    """Extract changed file paths from unified diff.

    Parses 'diff --git a/path b/path' headers to get file paths.

    Args:
        diff: Unified diff string.

    Returns:
        List of file paths (the 'b/' version representing new files).
    """
    files: list[str] = []
    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            # Format: diff --git a/path/to/file b/path/to/file
            parts = line.split(" ")
            if len(parts) >= 4:
                # Get the 'b/' version (new file path)
                path = parts[3].lstrip("b/")
                if path:
                    files.append(path)
    return files


def count_additions(diff: str) -> int:
    """Count addition lines in unified diff.

    Counts lines starting with '+' but not '+++' (file headers).

    Args:
        diff: Unified diff string.

    Returns:
        Number of addition lines.
    """
    return sum(
        1
        for line in diff.split("\n")
        if line.startswith("+") and not line.startswith("+++")
    )


def count_deletions(diff: str) -> int:
    """Count deletion lines in unified diff.

    Counts lines starting with '-' but not '---' (file headers).

    Args:
        diff: Unified diff string.

    Returns:
        Number of deletion lines.
    """
    return sum(
        1
        for line in diff.split("\n")
        if line.startswith("-") and not line.startswith("---")
    )


def extract_linked_issues(body: str) -> list[int]:
    """Extract issue numbers linked in PR body (fixes #N, closes #N, etc.)."""
    if not body:
        return []

    issues: set[int] = set()
    patterns = [
        r"(?i)fix(?:es|ed)?\s+#(\d+)",
        r"(?i)close(?:s|d)?\s+#(\d+)",
        r"(?i)resolve(?:s|d)?\s+#(\d+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, body):
            issues.add(int(match.group(1)))
    return sorted(issues)


class EnrichedPullRequest(BaseModel):
    """Enriched PR data from GH Archive + GitHub API."""

    id: str
    repo: str
    number: int

    title: str = ""
    body: str = ""
    user: str = ""
    state: str = "closed"

    base_commit: str = ""
    head_commit: str = ""
    merge_commit: str = ""
    base_ref: str = ""
    head_ref: str = ""

    created_at: datetime | None = None
    merged_at: datetime | None = None

    files_changed: int = Field(default=0, ge=0)
    additions: int = Field(default=0, ge=0)
    deletions: int = Field(default=0, ge=0)

    changed_files: list[str] = Field(default_factory=list)
    language: str = "unknown"
    stars: int = Field(default=0, ge=0)
    is_bot: bool = False
    linked_issues: list[int] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    # Store the PR diff to avoid duplicate API calls
    diff: str = Field(default="")


async def enrich_pr(
    event: GhArchiveEvent,
    github_client: GitHubClient,
) -> EnrichedPullRequest:
    """Enrich GH Archive event with GitHub API diff.

    Fetches only the PR diff (single API call) and extracts file info from it.
    Uses GH Archive event data for PR metadata (title, body, user, timestamps).
    """
    parts = event.repository.split("/", 1)
    owner = parts[0]
    repo_name = parts[1] if len(parts) > 1 else ""

    diff: str | None = None

    try:
        diff = await github_client.get_pr_diff(owner, repo_name, event.pull_number)
    except DiffTooLargeError:
        logger.info(
            f"Diff too large for {event.repository}#{event.pull_number}, using git clone fallback"
        )
        try:
            diff = await github_client.get_pr_diff_via_git(
                owner, repo_name, event.pull_number
            )
        except Exception as git_err:
            logger.warning(
                f"Git clone fallback also failed for {event.repository}#{event.pull_number}: {git_err}"
            )
    except Exception as e:
        logger.warning(
            f"GitHub API failed for {event.repository}#{event.pull_number}: {e}"
        )

    changed_paths = parse_files_from_diff(diff) if diff else []
    total_additions = count_additions(diff) if diff else 0
    total_deletions = count_deletions(diff) if diff else 0

    language = detect_language(changed_paths)
    if language == "unknown" and event.language_hint:
        language = event.language_hint

    user_login = event.user
    body_text = event.body or ""

    metadata: dict[str, str] = {
        "event_id": event.id,
        "event_type": event.event_type,
        "actor": event.actor,
        "action": event.action,
        "source": "gharchive",
        "has_api_data": str(bool(diff)).lower(),
    }
    if event.has_org:
        metadata["has_org"] = "true"

    return EnrichedPullRequest(
        id=event.id,
        repo=event.repository,
        number=event.pull_number,
        title=event.title,
        body=body_text,
        user=user_login,
        state="closed",
        base_commit=event.base_sha,
        head_commit="",
        merge_commit=event.merge_sha,
        base_ref=event.base_ref,
        head_ref=event.head_ref,
        created_at=event.created_at,
        merged_at=event.merged_at,
        files_changed=len(changed_paths),
        additions=total_additions,
        deletions=total_deletions,
        changed_files=changed_paths,
        language=language,
        stars=event.stars,
        is_bot=is_bot(user_login),
        linked_issues=extract_linked_issues(body_text),
        metadata=metadata,
        diff=diff or "",
    )


async def enrich_prs_batch(
    events: list[GhArchiveEvent],
    github_client: GitHubClient,
    concurrency: int = 10,
) -> list[EnrichedPullRequest]:
    """Enrich multiple PR events with concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)

    async def enrich_one(event: GhArchiveEvent) -> EnrichedPullRequest:
        async with semaphore:
            return await enrich_pr(event, github_client)

    return await asyncio.gather(*[enrich_one(e) for e in events])


def filter_bots(prs: list[EnrichedPullRequest]) -> list[EnrichedPullRequest]:
    """Filter out bot-authored PRs."""
    return [pr for pr in prs if not pr.is_bot]
