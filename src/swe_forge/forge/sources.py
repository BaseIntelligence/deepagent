"""Stage 0: contamination-resistant seed-repo registry with per-repo caps.

The registry is a small curated set of real, permissively-licensed, test-bearing
repositories (>=1 per supported language) each pinned to a concrete commit SHA
with its commit date, so the env-build checks out exactly that commit and never a
moving branch tip. Every entry carries a per-repo instance cap; the registry
hands out task instances through :meth:`SourceRegistry.acquire`, which enforces
the cap and tracks usage so a single repo can never seed more than its configured
number of tasks (guarding against repo-memorization / contamination).

The pinned commits below are real commit SHAs resolved from each repository's
upstream history; the env-build fetches and checks out exactly these SHAs.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from swe_forge.forge.models import InstanceGrant, RepoSpec


class SourceError(RuntimeError):
    """Base error for the source registry."""


class UnknownRepoError(SourceError):
    """Raised when a repo id is not present in the registry."""


class SourceRegistry:
    """An ordered, name-unique collection of :class:`RepoSpec` seed entries."""

    def __init__(self, specs: Iterable[RepoSpec]) -> None:
        self._specs: dict[str, RepoSpec] = {}
        for spec in specs:
            if spec.repo_id in self._specs:
                raise SourceError(
                    f"duplicate repo id in source registry: {spec.repo_id!r}"
                )
            self._specs[spec.repo_id] = spec

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[RepoSpec]:
        return iter(tuple(self._specs.values()))

    def repo_ids(self) -> tuple[str, ...]:
        """Return the registered repo ids in registration order."""
        return tuple(self._specs)

    def specs(self) -> tuple[RepoSpec, ...]:
        """Return the registered :class:`RepoSpec` entries in registration order."""
        return tuple(self._specs.values())

    def languages(self) -> tuple[str, ...]:
        """Return the distinct languages covered by the registry (sorted)."""
        return tuple(sorted({spec.language for spec in self._specs.values()}))

    def by_language(self, language: str) -> list[RepoSpec]:
        """Return every entry whose language matches ``language``."""
        target = language.strip().lower()
        return [spec for spec in self._specs.values() if spec.language == target]

    def has_language(self, language: str) -> bool:
        """Return ``True`` iff at least one entry covers ``language``."""
        return bool(self.by_language(language))

    def get(self, repo_id: str) -> RepoSpec:
        """Return the entry registered under ``repo_id`` or raise."""
        try:
            return self._specs[repo_id]
        except KeyError:
            raise UnknownRepoError(
                f"no repo {repo_id!r} in source registry; "
                f"known: {', '.join(self._specs) or '(none)'}"
            ) from None

    def acquire(self, repo_id: str) -> InstanceGrant:
        """Request one task instance from ``repo_id`` (enforces its cap)."""
        return self.get(repo_id).acquire()

    def reset_usage(self) -> None:
        """Reset usage counters for every entry."""
        for spec in self._specs.values():
            spec.reset_usage()

    def to_list(self) -> list[dict[str, object]]:
        """Return a JSON-serializable list of every entry's metadata."""
        return [spec.to_dict() for spec in self._specs.values()]


def _curated_specs() -> list[RepoSpec]:
    """Build fresh :class:`RepoSpec` instances for the curated seed set.

    Returns new objects on every call so each registry owns independent usage
    counters (acquiring instances from one registry never affects another).
    """
    return [
        RepoSpec(
            repo_id="pytest-dev/iniconfig",
            url="https://github.com/pytest-dev/iniconfig.git",
            commit="77db208ab4ae0cd2061d909fe222a1db72867850",
            commit_date="2026-02-25T11:10:21Z",
            language="python",
            license="MIT",
            instance_cap=5,
            default_branch="main",
            description="Brain-dead simple parsing of ini files (pure-Python, pytest).",
        ),
        RepoSpec(
            repo_id="sindresorhus/yocto-queue",
            url="https://github.com/sindresorhus/yocto-queue.git",
            commit="b07eac099753833b29d06c614149904445739776",
            commit_date="2025-11-11T06:30:03Z",
            language="javascript",
            license="MIT",
            instance_cap=4,
            default_branch="main",
            description="Tiny queue data structure (ESM JS/TS, zero runtime deps).",
        ),
        RepoSpec(
            repo_id="golang-jwt/jwt",
            url="https://github.com/golang-jwt/jwt.git",
            commit="e8e5b83ca9a5c5a3f287eda52c7bca78f9a6d176",
            commit_date="2026-05-26T21:30:25Z",
            language="go",
            license="MIT",
            instance_cap=6,
            default_branch="main",
            description="Go implementation of JSON Web Tokens (dependency-free).",
        ),
    ]


def build_source_registry() -> SourceRegistry:
    """Return a fresh curated source registry (>=1 usable repo per language)."""
    return SourceRegistry(_curated_specs())
