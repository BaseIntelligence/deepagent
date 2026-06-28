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
from dataclasses import dataclass, field
from datetime import datetime, timezone

#: Languages the forge pipeline supports end to end.
SUPPORTED_LANGUAGES: tuple[str, ...] = ("python", "javascript", "go")

#: The complete bug-generator menu (Stage 2). Every :class:`Candidate` must name
#: one of these; the concrete generators are added across the m3 milestone.
GENERATOR_NAMES: tuple[str, ...] = (
    "ast_mutation",
    "lm_authored",
    "pr_mirror",
    "function_removal",
    "bug_combination",
    "multi_file",
)

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


class BaselineNotGreenError(RuntimeError):
    """Raised when a downstream stage is asked to advance without a green baseline.

    Stage 1 (env build) establishes a green baseline as a *hard precondition* for
    every env-dependent downstream stage: no Candidate/mutation artifact may be
    produced from a repo that lacks a green :class:`EnvImage`. Downstream code
    enforces this by calling :func:`require_green_baseline`.
    """


@dataclass
class EnvImage:
    """Stage 1 artifact: one built Docker image per repo with a green baseline.

    Records the persisted image tag, the base image it was built from, the
    checked-out ``commit``, the install commands run so the repo's deps are
    present, the EXACT baseline test command proven green, and the baseline-green
    proof (``baseline_green`` + ``baseline_exit_code`` + a short ``baseline_summary``).
    A downstream stage may only proceed when ``baseline_green`` is ``True`` (see
    :func:`require_green_baseline`).
    """

    repo_id: str
    language: str
    image_tag: str
    base_image: str
    commit: str
    workspace_dir: str
    install_commands: list[str]
    baseline_test_command: str
    baseline_green: bool
    baseline_exit_code: int
    baseline_summary: str = ""
    prep_commands: list[str] = field(default_factory=list)
    built_at: str = ""
    provenance: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_id": self.repo_id,
            "language": self.language,
            "image_tag": self.image_tag,
            "base_image": self.base_image,
            "commit": self.commit,
            "workspace_dir": self.workspace_dir,
            "install_commands": list(self.install_commands),
            "baseline_test_command": self.baseline_test_command,
            "baseline_green": self.baseline_green,
            "baseline_exit_code": self.baseline_exit_code,
            "baseline_summary": self.baseline_summary,
            "prep_commands": list(self.prep_commands),
            "built_at": self.built_at,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EnvImage:
        """Reconstruct an :class:`EnvImage` from its :meth:`to_dict` form."""
        install = data.get("install_commands", [])
        prep = data.get("prep_commands", [])
        provenance = data.get("provenance", {})
        exit_code = data.get("baseline_exit_code", -1)
        return cls(
            repo_id=str(data["repo_id"]),
            language=str(data["language"]),
            image_tag=str(data["image_tag"]),
            base_image=str(data["base_image"]),
            commit=str(data.get("commit", "")),
            workspace_dir=str(data.get("workspace_dir", "")),
            install_commands=[str(c) for c in install]
            if isinstance(install, list)
            else [],
            baseline_test_command=str(data["baseline_test_command"]),
            baseline_green=bool(data.get("baseline_green", False)),
            baseline_exit_code=int(exit_code)
            if isinstance(exit_code, (int, str))
            else -1,
            baseline_summary=str(data.get("baseline_summary", "")),
            prep_commands=[str(c) for c in prep] if isinstance(prep, list) else [],
            built_at=str(data.get("built_at", "")),
            provenance=dict(provenance) if isinstance(provenance, dict) else {},
        )


def require_green_baseline(env_image: EnvImage | None) -> EnvImage:
    """Return ``env_image`` iff it proves a green baseline, else raise.

    The hard Stage-1 precondition: a downstream env-dependent stage calls this
    before producing any artifact, so a repo lacking a green :class:`EnvImage`
    can never advance past Stage 1. Raises :class:`BaselineNotGreenError` when
    the image is missing or its ``baseline_green`` proof is not set.
    """
    if env_image is None:
        raise BaselineNotGreenError(
            "no green baseline: env image is missing; cannot advance past Stage 1"
        )
    if not env_image.baseline_green:
        raise BaselineNotGreenError(
            f"no green baseline for {env_image.repo_id!r}: baseline_green is false "
            f"(exit {env_image.baseline_exit_code}); cannot advance past Stage 1"
        )
    return env_image


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class CandidateTarget:
    """The file(s) and symbol(s) a :class:`Candidate` mutates.

    ``files`` is the non-empty set of repo-relative paths the mutation touches;
    ``symbols`` names the targeted declarations (empty when a generator targets a
    whole file). :attr:`symbol` exposes the primary (first) symbol for the common
    single-symbol generators.
    """

    files: tuple[str, ...]
    symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.files or any(not str(f).strip() for f in self.files):
            raise ModelError("CandidateTarget.files must be a non-empty path list")

    @property
    def symbol(self) -> str:
        """The primary targeted symbol name (empty when none is set)."""
        return self.symbols[0] if self.symbols else ""

    def to_dict(self) -> dict[str, object]:
        return {
            "files": list(self.files),
            "symbols": list(self.symbols),
            "symbol": self.symbol,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CandidateTarget:
        files = data.get("files", [])
        symbols = data.get("symbols", [])
        return cls(
            files=tuple(str(f) for f in files) if isinstance(files, list) else (),
            symbols=tuple(str(s) for s in symbols) if isinstance(symbols, list) else (),
        )


@dataclass
class Provenance:
    """Auditable record attached to every generated artifact.

    Carries the fields the architecture mandates for reproducibility: the
    ``generator`` that produced the artifact, the ``seed`` that made it
    deterministic, the ``language``, a ``created_at`` timestamp, the
    ``tool_versions`` used, and a generator-specific ``details`` map (e.g. the
    mutation operator and the targeted span). Later stages extend ``details``
    with mutation/IRT/panel evidence.
    """

    generator: str
    seed: int
    language: str
    created_at: str = ""
    tool_versions: dict[str, str] = field(default_factory=dict)
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.generator).strip():
            raise ModelError("Provenance.generator must be non-empty")
        if self.language not in SUPPORTED_LANGUAGES:
            raise ModelError(
                f"Provenance.language must be one of {SUPPORTED_LANGUAGES}; "
                f"got {self.language!r}"
            )
        if not self.created_at:
            self.created_at = _utc_now_iso()

    def to_dict(self) -> dict[str, object]:
        return {
            "generator": self.generator,
            "seed": self.seed,
            "language": self.language,
            "created_at": self.created_at,
            "tool_versions": dict(self.tool_versions),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Provenance:
        versions = data.get("tool_versions", {})
        details = data.get("details", {})
        return cls(
            generator=str(data["generator"]),
            seed=int(data["seed"]) if isinstance(data["seed"], (int, str)) else 0,
            language=str(data["language"]),
            created_at=str(data.get("created_at", "")),
            tool_versions={str(k): str(v) for k, v in versions.items()}
            if isinstance(versions, dict)
            else {},
            details=dict(details) if isinstance(details, dict) else {},
        )


@dataclass
class Candidate:
    """Stage 2 artifact: one manufactured bug with its by-construction gold fix.

    A :class:`Candidate` pairs a forward ``mutation_patch`` (turns known-good code
    broken) with the inverse ``oracle_patch`` (the gold fix). Applying the
    mutation then the oracle to a pristine checkout restores every touched file
    byte-for-byte: this is the gold-by-construction guarantee. ``target`` records
    the touched file(s)/symbol(s), ``generator`` is one of :data:`GENERATOR_NAMES`,
    and ``difficulty_hint`` is a coarse a-priori label.
    """

    language: str
    generator: str
    target: CandidateTarget
    mutation_patch: str
    oracle_patch: str
    difficulty_hint: str
    provenance: Provenance

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ModelError(
                f"Candidate.language must be one of {SUPPORTED_LANGUAGES}; "
                f"got {self.language!r}"
            )
        if self.generator not in GENERATOR_NAMES:
            raise ModelError(
                f"Candidate.generator must be one of {GENERATOR_NAMES}; "
                f"got {self.generator!r}"
            )
        for field_name in ("mutation_patch", "oracle_patch", "difficulty_hint"):
            if not str(getattr(self, field_name)).strip():
                raise ModelError(f"Candidate.{field_name} must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "generator": self.generator,
            "target": self.target.to_dict(),
            "mutation_patch": self.mutation_patch,
            "oracle_patch": self.oracle_patch,
            "difficulty_hint": self.difficulty_hint,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Candidate:
        target = data["target"]
        provenance = data["provenance"]
        return cls(
            language=str(data["language"]),
            generator=str(data["generator"]),
            target=CandidateTarget.from_dict(target)
            if isinstance(target, dict)
            else CandidateTarget(files=()),
            mutation_patch=str(data["mutation_patch"]),
            oracle_patch=str(data["oracle_patch"]),
            difficulty_hint=str(data["difficulty_hint"]),
            provenance=Provenance.from_dict(provenance)
            if isinstance(provenance, dict)
            else Provenance(generator="", seed=0, language="python"),
        )
