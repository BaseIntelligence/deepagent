"""PR enrichment logic for combining GH Archive events with GitHub API data."""

import asyncio
import logging
import re
from collections import Counter
from datetime import datetime

from pydantic import BaseModel, Field

from .gharchive import GhArchiveEvent
from .github_api import GitHubClient, PullRequest, PRFile

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


async def enrich_pr(
    event: GhArchiveEvent,
    github_client: GitHubClient,
) -> EnrichedPullRequest:
    """Enrich GH Archive event with GitHub API data.

    Fetches PR details and files in parallel, combining with event data.
    Falls back to event data on API failures.
    """
    parts = event.repository.split("/", 1)
    owner = parts[0]
    repo_name = parts[1] if len(parts) > 1 else ""

    pr_data: PullRequest | None = None
    pr_files: list[PRFile] = []

    try:
        pr_data, pr_files = await asyncio.gather(
            github_client.get_pr(owner, repo_name, event.pull_number),
            github_client.get_pr_files(owner, repo_name, event.pull_number),
        )
    except Exception as e:
        logger.warning(
            f"GitHub API failed for {event.repository}#{event.pull_number}: {e}"
        )

    changed_paths = [f.filename for f in pr_files]
    total_additions = sum(f.additions for f in pr_files)
    total_deletions = sum(f.deletions for f in pr_files)

    language = detect_language(changed_paths)
    if language == "unknown" and event.language_hint:
        language = event.language_hint

    user_login = pr_data.user_login if pr_data else event.user
    body_text = (pr_data.body if pr_data and pr_data.body else event.body) or ""

    metadata: dict[str, str] = {
        "event_id": event.id,
        "event_type": event.event_type,
        "actor": event.actor,
        "action": event.action,
        "source": "gharchive",
        "has_api_data": str(bool(pr_data)).lower(),
    }
    if event.has_org:
        metadata["has_org"] = "true"

    return EnrichedPullRequest(
        id=event.id,
        repo=event.repository,
        number=event.pull_number,
        title=pr_data.title if pr_data else event.title,
        body=body_text,
        user=user_login,
        state=pr_data.state if pr_data else "closed",
        base_commit=pr_data.base_sha if pr_data else event.base_sha,
        head_commit=pr_data.head_sha if pr_data else "",
        merge_commit=event.merge_sha if not pr_data else pr_data.head_sha,
        base_ref=pr_data.base_ref if pr_data else event.base_ref,
        head_ref=pr_data.head_ref if pr_data else event.head_ref,
        created_at=event.created_at,
        merged_at=pr_data.merged_at if pr_data else event.merged_at,
        files_changed=pr_data.changed_files if pr_data else len(changed_paths),
        additions=total_additions,
        deletions=total_deletions,
        changed_files=changed_paths,
        language=language,
        stars=pr_data.stars if pr_data else event.stars,
        is_bot=is_bot(user_login),
        linked_issues=extract_linked_issues(body_text),
        metadata=metadata,
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
