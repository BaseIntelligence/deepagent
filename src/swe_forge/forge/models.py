"""Core data model for the forge pipeline.

The pipeline is a pure data flow: each stage consumes the previous artifact and
produces the next, and every artifact is a plain dataclass that serializes to
JSON and carries the metadata the later stages and provenance need.

This module currently defines the Stage 0 :class:`RepoSpec` (a
contamination-resistant seed-repo record pinned to a concrete commit, with
per-repo instance-cap bookkeeping) and its :class:`InstanceGrant` result. Later
milestones extend it with the downstream artifacts (EnvImage, Candidate,
GeneratedSpec, OracleReport, CalibrationReport, ForgeTask, Provenance).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

#: Languages the forge pipeline supports end to end.
SUPPORTED_LANGUAGES: tuple[str, ...] = ("python", "javascript", "go")

#: A pinned commit must be a full 40-hex git object name, never a branch/tag or
#: an abbreviated ref, so the build checks out exactly one immutable commit.
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ModelError(ValueError):
    """Raised when a forge data-model object is constructed with invalid fields."""


def _parse_commit_date(value: str) -> None:
    """Validate ``value`` is an ISO-8601 timestamp; raise :class:`ModelError`."""
    candidate = value.strip()
    normalized = candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ModelError(
            f"commit_date must be an ISO-8601 timestamp; got {value!r}"
        ) from exc


@dataclass(frozen=True)
class InstanceGrant:
    """Outcome of requesting one task instance against a :class:`RepoSpec`.

    ``accepted`` is ``True`` when the request fell within the repo's instance
    cap. ``instance_index`` is the 1-based ordinal of an accepted instance (``0``
    when rejected); ``reason`` explains a rejection (empty on acceptance).
    ``cap``/``used``/``remaining`` snapshot the repo's bookkeeping *after* the
    request so the caller can audit that usage never exceeds the cap.
    """

    repo_id: str
    accepted: bool
    cap: int
    used: int
    remaining: int
    instance_index: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_id": self.repo_id,
            "accepted": self.accepted,
            "cap": self.cap,
            "used": self.used,
            "remaining": self.remaining,
            "instance_index": self.instance_index,
            "reason": self.reason,
        }


@dataclass
class RepoSpec:
    """A curated seed repository pinned to one immutable commit.

    Records the contamination metadata the pipeline needs (``commit`` SHA,
    ``commit_date``, ``language``, ``license``) plus a per-repo ``instance_cap``
    and a mutable ``used`` counter. Instances are handed out via :meth:`acquire`,
    which enforces the cap so a single repo can never seed more than its
    configured number of tasks (a guard against repo-memorization).
    """

    repo_id: str
    url: str
    commit: str
    commit_date: str
    language: str
    license: str
    instance_cap: int
    used: int = 0
    default_branch: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        for field_name in ("repo_id", "url", "commit", "commit_date", "license"):
            if not str(getattr(self, field_name)).strip():
                raise ModelError(f"RepoSpec.{field_name} must be non-empty")

        commit = self.commit.strip().lower()
        if not _FULL_SHA_RE.match(commit):
            raise ModelError(
                "RepoSpec.commit must be a full 40-hex git SHA "
                f"(a pinned commit, not a branch/tag/short ref); got {self.commit!r}"
            )
        self.commit = commit

        language = self.language.strip().lower()
        if language not in SUPPORTED_LANGUAGES:
            raise ModelError(
                f"RepoSpec.language must be one of {SUPPORTED_LANGUAGES}; "
                f"got {self.language!r}"
            )
        self.language = language

        _parse_commit_date(self.commit_date)

        if self.instance_cap < 1:
            raise ModelError(
                f"RepoSpec.instance_cap must be >= 1; got {self.instance_cap}"
            )
        if self.used < 0:
            raise ModelError(f"RepoSpec.used must be >= 0; got {self.used}")
        if self.used > self.instance_cap:
            raise ModelError(
                f"RepoSpec.used ({self.used}) must not exceed "
                f"instance_cap ({self.instance_cap})"
            )

    @property
    def remaining(self) -> int:
        """Instances still available before the cap is reached."""
        return self.instance_cap - self.used

    @property
    def at_cap(self) -> bool:
        """``True`` when no further instances may be acquired."""
        return self.used >= self.instance_cap

    def acquire(self) -> InstanceGrant:
        """Request one task instance, enforcing the per-repo cap.

        Returns an accepted :class:`InstanceGrant` and increments ``used`` while
        capacity remains; once the cap is reached every further request is
        rejected with a clear reason and ``used`` is left untouched (so it can
        never exceed ``instance_cap``).
        """
        if self.at_cap:
            return InstanceGrant(
                repo_id=self.repo_id,
                accepted=False,
                cap=self.instance_cap,
                used=self.used,
                remaining=self.remaining,
                reason=(
                    f"per-repo cap reached for {self.repo_id}: "
                    f"cap={self.instance_cap}, used={self.used}"
                ),
            )
        self.used += 1
        return InstanceGrant(
            repo_id=self.repo_id,
            accepted=True,
            cap=self.instance_cap,
            used=self.used,
            remaining=self.remaining,
            instance_index=self.used,
        )

    def reset_usage(self) -> None:
        """Reset the usage counter to zero (e.g. for a fresh pipeline run)."""
        self.used = 0

    def checkout_commands(self) -> list[str]:
        """Shell commands that check out exactly the pinned commit.

        Fetches and checks out the immutable SHA (not a branch tip) so a build
        is reproducible from the recorded commit alone.
        """
        return [
            "git init -q",
            f"git remote add origin {self.url}",
            f"git fetch -q --depth 1 origin {self.commit}",
            f"git checkout -q {self.commit}",
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_id": self.repo_id,
            "url": self.url,
            "commit": self.commit,
            "commit_date": self.commit_date,
            "language": self.language,
            "license": self.license,
            "instance_cap": self.instance_cap,
            "used": self.used,
            "remaining": self.remaining,
            "default_branch": self.default_branch,
            "description": self.description,
        }
