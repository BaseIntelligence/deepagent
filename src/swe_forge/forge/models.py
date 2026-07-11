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

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swe_forge.forge.oracle.multifault import MultiFaultCompletenessEvidence

#: Languages the forge pipeline supports end to end.
SUPPORTED_LANGUAGES: tuple[str, ...] = ("python", "javascript", "go")

#: The solver-panel difficulty tiers (weak -> mid -> frontier), ordered from the
#: least to the most capable. Kept in sync with ``panel.VALID_TIERS``.
PANEL_TIERS: tuple[str, ...] = ("weak", "mid", "frontier")

#: Terminal band verdicts a :class:`CalibrationReport` may carry, plus the
#: ``pending`` pre-filter state (the band filter assigns the terminal verdict).
BAND_VERDICTS: tuple[str, ...] = ("keep", "drop", "pending")

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

#: The two terminal verdicts an oracle gate (and the whole pipeline) may reach.
ORACLE_VERDICTS: tuple[str, ...] = ("pass", "reject")

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
    #: Optional per-repo overrides (empty = fall back to the adapter defaults).
    #: ``baseline_install`` overrides the adapter's baseline install commands;
    #: ``baseline_test`` overrides the baseline/P2P test command;
    #: ``p2p_exclusions`` names fix-independent self-tests the env build must
    #: SKIP from the baseline/P2P suite (applied language-agnostically by the
    #: adapter). These let a repo whose green baseline differs from the bare
    #: adapter default join the registry without any ``if language ==`` in the
    #: stage code.
    baseline_install: tuple[str, ...] = ()
    baseline_test: str = ""
    p2p_exclusions: tuple[str, ...] = ()
    #: ``pr_mirror`` generation params: the upstream ``owner/name`` slug and the
    #: merged ``pr_number`` whose fault this entry reintroduces, plus the
    #: preferred ``pr_generator`` (defaults to ``pr_mirror``). A non-zero
    #: ``pr_number`` + a slug ``pr_repo`` marks an allowlist entry the pilot
    #: drives through the ``pr_mirror`` generator (see :prop:`has_pr_mirror`).
    pr_repo: str = ""
    pr_number: int = 0
    pr_generator: str = ""
    #: Optional SINGLE discriminating-assertion F2P selector for a ``pr_mirror``
    #: entry. When set, the pilot selects exactly these test name(s) as the
    #: isolated F2P (instead of the whole flipping test the ``p2p_exclusions``
    #: name), keeping the F2P a single precise assertion in semantic-correctness
    #: domains (URL/email validators) so the differential synthesizer's
    #: discriminators stay gold-green. Falls back to ``p2p_exclusions`` when unset.
    pr_f2p_names: tuple[str, ...] = ()
    #: When ``True`` the pilot also runs the structural generators
    #: (ast_mutation/function_removal/...) on this repo. Set on the diversified
    #: MODULAR seeds where a structural mutation can isolate a fault.
    structural_source: bool = False

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

        # Normalize the optional override containers to immutable tuples of
        # non-empty strings (tolerant of a list passed at construction).
        self.baseline_install = tuple(
            str(c).strip() for c in self.baseline_install if str(c).strip()
        )
        self.p2p_exclusions = tuple(
            str(e).strip() for e in self.p2p_exclusions if str(e).strip()
        )
        self.pr_f2p_names = tuple(
            str(n).strip() for n in self.pr_f2p_names if str(n).strip()
        )
        self.baseline_test = str(self.baseline_test).strip()
        self.pr_repo = str(self.pr_repo).strip()
        self.pr_generator = str(self.pr_generator).strip()

        if self.pr_number < 0:
            raise ModelError(f"RepoSpec.pr_number must be >= 0; got {self.pr_number}")
        if self.pr_generator and self.pr_generator not in GENERATOR_NAMES:
            raise ModelError(
                f"RepoSpec.pr_generator must be one of {GENERATOR_NAMES}; "
                f"got {self.pr_generator!r}"
            )
        if self.pr_number > 0 and "/" not in self.pr_repo:
            raise ModelError(
                "RepoSpec.pr_repo must be an 'owner/name' slug when pr_number is "
                f"set; got {self.pr_repo!r}"
            )

    @property
    def remaining(self) -> int:
        """Instances still available before the cap is reached."""
        return self.instance_cap - self.used

    @property
    def at_cap(self) -> bool:
        """``True`` when no further instances may be acquired."""
        return self.used >= self.instance_cap

    @property
    def has_pr_mirror(self) -> bool:
        """``True`` iff this entry is a ``pr_mirror`` allowlist source.

        An allowlist entry pins its own ``base_commit`` (the merged-PR state) and
        carries the upstream ``pr_repo`` slug + ``pr_number`` the ``pr_mirror``
        generator reconstructs the reverted fault from.
        """
        return self.pr_number > 0 and "/" in self.pr_repo

    @property
    def preferred_generator(self) -> str:
        """The generator to drive a ``pr_mirror`` entry (defaults to ``pr_mirror``)."""
        return self.pr_generator or "pr_mirror"

    def pr_params(self) -> dict[str, object]:
        """The ``pr_mirror`` generation params (``repo`` slug + ``pr_number``).

        Returned as the ``CandidatePlan.params`` payload the generator reads; the
        GitHub token is never part of this (it is read from the environment and
        never logged).
        """
        return {"repo": self.pr_repo, "pr_number": self.pr_number}

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

    def release(self, grant: InstanceGrant) -> None:
        """Release a previously accepted instance after its export is refused.

        Capacity is consumed by shipped tasks, not by a candidate that passed
        calibration but was refused during Stage 5 (for example by the leak
        audit). The checkpoint owns the one-shot lifecycle of a grant and calls
        this only for a terminal non-shipped export result.
        """
        if not grant.accepted:
            raise ModelError("cannot release a rejected instance grant")
        if grant.repo_id != self.repo_id:
            raise ModelError(
                f"instance grant repo {grant.repo_id!r} does not match "
                f"RepoSpec {self.repo_id!r}"
            )
        if self.used < 1:
            raise ModelError(
                f"RepoSpec.used underflow while releasing {self.repo_id!r}"
            )
        self.used -= 1

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
            "baseline_install": list(self.baseline_install),
            "baseline_test": self.baseline_test,
            "p2p_exclusions": list(self.p2p_exclusions),
            "pr_repo": self.pr_repo,
            "pr_number": self.pr_number,
            "pr_generator": self.pr_generator,
            "pr_f2p_names": list(self.pr_f2p_names),
            "structural_source": self.structural_source,
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
    # The original, unfiltered upstream/public regression command proven by the
    # env build.  Candidate-specific P2P derivation may narrow
    # ``baseline_test_command`` later, but it must never replace this command.
    # Alt-correct validates gold and every teacher proposal against this command
    # before it is allowed to inspect hidden F2P or filtered P2P results.
    original_public_test_command: str = ""
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
            "original_public_test_command": self.original_public_test_command,
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
            original_public_test_command=str(
                data.get("original_public_test_command", "")
            ),
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


def _coerce_int(value: object, default: int = 0) -> int:
    """Best-effort int coercion for ``from_dict`` loaders (tolerant of junk)."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, str)):
        try:
            return int(value)
        except ValueError:
            return default
    return default


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


@dataclass
class GeneratedSpec:
    """Stage 2 artifact: the agent-facing task description for a :class:`Candidate`.

    Built by *test-conditioned backtranslation*: the ``problem_statement`` is
    derived from the manufactured fault's FAIL->PASS (F2P) failure trace - the
    observable broken behavior of the hidden failing tests - NOT from the
    mutation/oracle diff. ``requirements`` is a non-empty list of expectations,
    each grounded in (and traceable to) a named F2P test. ``interface_block``
    enumerates the expected public symbol name(s)/signature(s) of the real target
    API so a correct solution is never failed merely for a naming difference.

    None of the three fields may leak the oracle/implementation body: the spec
    describes *what* must hold (from the tests) and the *signatures* to implement,
    never *how* (the gold code). A :class:`GeneratedSpec` is only ever emitted
    alongside a Candidate that passed its forward+inverse self-validation.
    """

    problem_statement: str
    requirements: list[str]
    interface_block: str
    provenance: Provenance

    def __post_init__(self) -> None:
        if not str(self.problem_statement).strip():
            raise ModelError("GeneratedSpec.problem_statement must be non-empty")
        if not self.requirements or any(
            not str(item).strip() for item in self.requirements
        ):
            raise ModelError(
                "GeneratedSpec.requirements must be a non-empty list of "
                "non-empty strings"
            )
        if not str(self.interface_block).strip():
            raise ModelError("GeneratedSpec.interface_block must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "problem_statement": self.problem_statement,
            "requirements": list(self.requirements),
            "interface_block": self.interface_block,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GeneratedSpec:
        requirements = data.get("requirements", [])
        provenance = data.get("provenance", {})
        return cls(
            problem_statement=str(data["problem_statement"]),
            requirements=[str(item) for item in requirements]
            if isinstance(requirements, list)
            else [],
            interface_block=str(data["interface_block"]),
            provenance=Provenance.from_dict(provenance)
            if isinstance(provenance, dict)
            else Provenance(generator="", seed=0, language="python"),
        )


@dataclass
class OracleTestFile:
    """One hidden test file recorded on an :class:`OracleReport`.

    ``path`` is the repo-relative path the test was written to (and selected by
    its ``fail_to_pass`` command); ``origin`` is ``"synthesized"`` (authored by
    the agentic generator) or ``"provided"`` (a caller-declared/intended test);
    ``content`` is the test body, kept so the later leak-audit/sanitize gate and
    any re-run have the source.
    """

    path: str
    content: str = ""
    origin: str = "synthesized"

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ModelError("OracleTestFile.path must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "content": self.content, "origin": self.origin}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> OracleTestFile:
        return cls(
            path=str(data["path"]),
            content=str(data.get("content", "")),
            origin=str(data.get("origin", "synthesized")),
        )


@dataclass(frozen=True)
class FinalMutationEvidence:
    """Mutation-adequacy evidence bound to the exact shipped hidden suite.

    Mutation counts are meaningful only for the tests that were actually present
    when the tool completed. The final mutation remeasurement records the
    canonical hidden-suite fingerprint together with its replacement counts so
    every downstream consumer can reject stale evidence after later oracle gates
    add or remove tests.
    """

    suite_fingerprint: str
    mutants_total: int
    mutants_killed: int
    threshold: float
    tool: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.suite_fingerprint):
            raise ModelError(
                "FinalMutationEvidence.suite_fingerprint must be a SHA-256 hex digest"
            )
        if self.mutants_total < 0:
            raise ModelError("FinalMutationEvidence.mutants_total must be >= 0")
        if not 0 <= self.mutants_killed <= self.mutants_total:
            raise ModelError(
                "FinalMutationEvidence.mutants_killed must satisfy 0 <= killed <= total"
            )
        if not 0.0 < self.threshold <= 1.0:
            raise ModelError(
                "FinalMutationEvidence.threshold must be in (0, 1]; "
                f"got {self.threshold}"
            )

    @property
    def kill_ratio(self) -> float:
        return self.mutants_killed / self.mutants_total if self.mutants_total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "suite_fingerprint": self.suite_fingerprint,
            "mutants_total": self.mutants_total,
            "mutants_killed": self.mutants_killed,
            "threshold": self.threshold,
            "tool": self.tool,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FinalMutationEvidence:
        return cls(
            suite_fingerprint=str(data["suite_fingerprint"]),
            mutants_total=_coerce_int(data.get("mutants_total", 0)),
            mutants_killed=_coerce_int(data.get("mutants_killed", 0)),
            threshold=_coerce_float(data.get("threshold", 0.0)),
            tool=str(data.get("tool", "")),
        )


@dataclass
class OracleReport:
    """Stage 3 artifact: the oracle-hardening verdict + evidence for a Candidate.

    A task ships only if its hidden test suite is *adequate*; the oracle gates
    (establish -> flakiness -> mutation -> differential -> alt-correct -> leak)
    each contribute evidence to this report and the pipeline sets the terminal
    ``verdict``. The establish gate populates ``fail_to_pass`` (the hidden tests
    that FAIL on the broken tree and PASS on the gold tree), ``pass_to_pass``
    (the regression suite that stays green on both trees), and ``test_files``
    (the synthesized/provided test bodies); later gates fill ``flakiness_runs``,
    ``mutants_total``/``mutants_killed``, ``differential_pass``,
    ``alt_correct_accepted``, and ``leak_audit``.

    Invariant: a ``reject`` verdict always carries at least one attributable
    ``reason``; a ``pass`` verdict carries none.
    """

    language: str
    generator: str
    verdict: str
    reasons: list[str] = field(default_factory=list)
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    test_files: list[OracleTestFile] = field(default_factory=list)
    flakiness_runs: int = 0
    mutants_total: int = 0
    mutants_killed: int = 0
    final_mutation_evidence: FinalMutationEvidence | None = None
    multifault_evidence: MultiFaultCompletenessEvidence | None = None
    differential_pass: bool = False
    alt_correct_accepted: bool = False
    leak_audit: str = ""
    provenance: Provenance | None = None
    details: dict[str, object] = field(default_factory=dict)
    # Raw teacher alternatives are audit-only evidence.  They deliberately do
    # not participate in ``to_dict`` because that payload is used by the public
    # task, dataset, report, and generation-manifest surfaces.  A protected
    # writer may use ``to_protected_dict`` to persist them outside those surfaces.
    protected_alt_correct_audit: dict[str, object] | None = field(
        default=None, repr=False
    )
    # Concrete-transport receipts authorize teacher gate evidence. Like raw
    # alternatives, receipt secrets never enter public serialization.
    protected_teacher_transport_receipts: list[dict[str, object]] = field(
        default_factory=list, repr=False
    )

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ModelError(
                f"OracleReport.language must be one of {SUPPORTED_LANGUAGES}; "
                f"got {self.language!r}"
            )
        if self.verdict not in ORACLE_VERDICTS:
            raise ModelError(
                f"OracleReport.verdict must be one of {ORACLE_VERDICTS}; "
                f"got {self.verdict!r}"
            )
        if self.verdict == "reject" and not self.reasons:
            raise ModelError(
                "OracleReport.verdict 'reject' requires at least one attributable "
                "reason"
            )
        if self.verdict == "pass" and self.reasons:
            raise ModelError(
                "OracleReport.verdict 'pass' must carry no reasons; got "
                f"{self.reasons!r}"
            )
        if not (
            0 <= self.mutants_killed <= self.mutants_total or self.mutants_total == 0
        ):
            raise ModelError(
                "OracleReport.mutants_killed must satisfy 0 <= killed <= total"
            )

    @property
    def is_pass(self) -> bool:
        """``True`` iff the terminal verdict is ``pass``."""
        return self.verdict == "pass"

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "generator": self.generator,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "fail_to_pass": list(self.fail_to_pass),
            "pass_to_pass": list(self.pass_to_pass),
            "test_files": [tf.to_dict() for tf in self.test_files],
            "flakiness_runs": self.flakiness_runs,
            "mutants_total": self.mutants_total,
            "mutants_killed": self.mutants_killed,
            "final_mutation_evidence": self.final_mutation_evidence.to_dict()
            if self.final_mutation_evidence
            else None,
            "multifault_evidence": self.multifault_evidence.to_dict()
            if self.multifault_evidence
            else None,
            "differential_pass": self.differential_pass,
            "alt_correct_accepted": self.alt_correct_accepted,
            "leak_audit": self.leak_audit,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "details": dict(self.details),
        }

    def to_protected_dict(self) -> dict[str, object]:
        """Serialize this report with private alt-correct proposal evidence.

        This method is exclusively for the protected audit store.  The default
        :meth:`to_dict` remains the only serialization permitted for agent-facing
        task workspaces, datasets, benchmark reports, and manifests.
        """
        data = self.to_dict()
        data["protected_alt_correct_audit"] = (
            dict(self.protected_alt_correct_audit)
            if self.protected_alt_correct_audit is not None
            else None
        )
        data["protected_teacher_transport_receipts"] = [
            dict(receipt) for receipt in self.protected_teacher_transport_receipts
        ]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> OracleReport:
        test_files = data.get("test_files", [])
        provenance = data.get("provenance")
        reasons = data.get("reasons", [])
        f2p = data.get("fail_to_pass", [])
        p2p = data.get("pass_to_pass", [])
        details = data.get("details", {})
        final_mutation_evidence = data.get("final_mutation_evidence")
        multifault_evidence = data.get("multifault_evidence")
        if multifault_evidence is not None and not isinstance(
            multifault_evidence, dict
        ):
            raise ModelError(
                "OracleReport.multifault_evidence must be an object or null"
            )
        if isinstance(multifault_evidence, dict):
            from swe_forge.forge.oracle.multifault import MultiFaultCompletenessEvidence

            parsed_multifault_evidence = MultiFaultCompletenessEvidence.from_dict(
                multifault_evidence
            )
        else:
            parsed_multifault_evidence = None
        return cls(
            language=str(data["language"]),
            generator=str(data.get("generator", "")),
            verdict=str(data["verdict"]),
            reasons=[str(r) for r in reasons] if isinstance(reasons, list) else [],
            fail_to_pass=[str(c) for c in f2p] if isinstance(f2p, list) else [],
            pass_to_pass=[str(c) for c in p2p] if isinstance(p2p, list) else [],
            test_files=[
                OracleTestFile.from_dict(tf)
                for tf in test_files
                if isinstance(tf, dict)
            ]
            if isinstance(test_files, list)
            else [],
            flakiness_runs=_coerce_int(data.get("flakiness_runs", 0)),
            mutants_total=_coerce_int(data.get("mutants_total", 0)),
            mutants_killed=_coerce_int(data.get("mutants_killed", 0)),
            final_mutation_evidence=FinalMutationEvidence.from_dict(
                final_mutation_evidence
            )
            if isinstance(final_mutation_evidence, dict)
            else None,
            multifault_evidence=parsed_multifault_evidence,
            differential_pass=bool(data.get("differential_pass", False)),
            alt_correct_accepted=bool(data.get("alt_correct_accepted", False)),
            leak_audit=str(data.get("leak_audit", "")),
            provenance=Provenance.from_dict(provenance)
            if isinstance(provenance, dict)
            else None,
            details=dict(details) if isinstance(details, dict) else {},
        )

    @classmethod
    def from_protected_dict(cls, data: dict[str, object]) -> OracleReport:
        """Restore a report from a protected audit payload, if structurally safe."""
        report = cls.from_dict(data)
        audit = data.get("protected_alt_correct_audit")
        if audit is not None and not isinstance(audit, dict):
            raise ModelError("protected_alt_correct_audit must be an object or null")
        report.protected_alt_correct_audit = (
            dict(audit) if isinstance(audit, dict) else None
        )
        receipts = data.get("protected_teacher_transport_receipts", [])
        if not isinstance(receipts, list) or not all(
            isinstance(receipt, dict) for receipt in receipts
        ):
            raise ModelError(
                "protected_teacher_transport_receipts must be a list of objects"
            )
        report.protected_teacher_transport_receipts = [
            dict(receipt) for receipt in receipts
        ]
        return report


def _coerce_float(value: object, default: float = 0.0) -> float:
    """Best-effort float coercion for ``from_dict`` loaders (tolerant of junk)."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return default
    return default


@dataclass
class ModelSolveRecord:
    """One panel model's calibration summary on a :class:`CalibrationReport`.

    ``solves`` of ``k`` independent rollouts succeeded (a *solve* = the model's
    patch passed the FULL hidden test set in Docker), and ``pass_at_k`` is the
    resulting pass-rate. The pass@k invariant is enforced here so the data model
    itself guards it: ``solves == 0`` forces ``pass_at_k == 0`` and any solve
    forces ``pass_at_k > 0``.
    """

    model: str
    tier: str
    k: int
    solves: int
    pass_at_k: float

    def __post_init__(self) -> None:
        if not str(self.model).strip():
            raise ModelError("ModelSolveRecord.model must be non-empty")
        if self.tier not in PANEL_TIERS:
            raise ModelError(
                f"ModelSolveRecord.tier must be one of {PANEL_TIERS}; got {self.tier!r}"
            )
        if self.k < 0:
            raise ModelError(f"ModelSolveRecord.k must be >= 0; got {self.k}")
        if not (0 <= self.solves <= self.k):
            raise ModelError(
                "ModelSolveRecord.solves must satisfy 0 <= solves <= k "
                f"({self.solves} vs k={self.k})"
            )
        self.pass_at_k = float(self.pass_at_k)
        if not (0.0 <= self.pass_at_k <= 1.0):
            raise ModelError(
                f"ModelSolveRecord.pass_at_k must be in [0, 1]; got {self.pass_at_k}"
            )
        if self.solves == 0 and self.pass_at_k != 0.0:
            raise ModelError(
                "ModelSolveRecord with solves == 0 must have pass_at_k == 0.0; "
                f"got {self.pass_at_k}"
            )
        if self.solves >= 1 and self.pass_at_k <= 0.0:
            raise ModelError(
                "ModelSolveRecord with solves >= 1 must have pass_at_k > 0; "
                f"got {self.pass_at_k}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "tier": self.tier,
            "k": self.k,
            "solves": self.solves,
            "pass_at_k": self.pass_at_k,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ModelSolveRecord:
        return cls(
            model=str(data["model"]),
            tier=str(data["tier"]),
            k=_coerce_int(data.get("k", 0)),
            solves=_coerce_int(data.get("solves", 0)),
            pass_at_k=_coerce_float(data.get("pass_at_k", 0.0)),
        )


@dataclass
class CalibrationReport:
    """Stage 4 artifact: the calibration (difficulty + discrimination) verdict.

    Records the per-model panel outcome (``models``: one
    :class:`ModelSolveRecord` per validated model), the 2-parameter IRT fit over
    that per-model/per-rollout solve matrix (``irt_difficulty`` and
    ``irt_discrimination``, both matrix-derived), and the band ``band_verdict``.

    The IRT fields are populated by the calibration fit; ``band_verdict`` starts
    ``"pending"`` and is set to its terminal ``"keep"``/``"drop"`` value by the
    band filter (with an attributable ``reason``). A ``ForgeTask`` may only be
    built from a ``keep`` report.
    """

    language: str
    models: list[ModelSolveRecord]
    k: int
    irt_difficulty: float
    irt_discrimination: float
    band_verdict: str = "pending"
    reasons: list[str] = field(default_factory=list)
    difficulty_hint: str = ""
    provenance: Provenance | None = None
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ModelError(
                f"CalibrationReport.language must be one of {SUPPORTED_LANGUAGES}; "
                f"got {self.language!r}"
            )
        if self.band_verdict not in BAND_VERDICTS:
            raise ModelError(
                f"CalibrationReport.band_verdict must be one of {BAND_VERDICTS}; "
                f"got {self.band_verdict!r}"
            )
        if self.k < 0:
            raise ModelError(f"CalibrationReport.k must be >= 0; got {self.k}")
        self.irt_difficulty = float(self.irt_difficulty)
        self.irt_discrimination = float(self.irt_discrimination)
        if not math.isfinite(self.irt_difficulty):
            raise ModelError(
                f"CalibrationReport.irt_difficulty must be finite; "
                f"got {self.irt_difficulty}"
            )
        if not math.isfinite(self.irt_discrimination):
            raise ModelError(
                f"CalibrationReport.irt_discrimination must be finite; "
                f"got {self.irt_discrimination}"
            )

    @property
    def is_keep(self) -> bool:
        """``True`` iff the terminal band verdict is ``keep``."""
        return self.band_verdict == "keep"

    def tier_pass_rates(self) -> dict[str, float]:
        """Mean ``pass_at_k`` per tier present in ``models`` (empty tiers omitted)."""
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for rec in self.models:
            sums[rec.tier] = sums.get(rec.tier, 0.0) + rec.pass_at_k
            counts[rec.tier] = counts.get(rec.tier, 0) + 1
        return {tier: sums[tier] / counts[tier] for tier in sums}

    def frontier_pass_at_k(self) -> float:
        """The strongest frontier model's ``pass_at_k`` (``0.0`` if no frontier)."""
        frontier = [r.pass_at_k for r in self.models if r.tier == "frontier"]
        return max(frontier) if frontier else 0.0

    def set_band_verdict(self, verdict: str, reason: str) -> None:
        """Set the terminal band verdict (``keep``/``drop``) with a reason."""
        if verdict not in ("keep", "drop"):
            raise ModelError(
                f"terminal band_verdict must be 'keep' or 'drop'; got {verdict!r}"
            )
        if not str(reason).strip():
            raise ModelError("a band verdict requires a non-empty reason")
        self.band_verdict = verdict
        self.reasons = [reason]

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "models": [m.to_dict() for m in self.models],
            "k": self.k,
            "irt_difficulty": self.irt_difficulty,
            "irt_discrimination": self.irt_discrimination,
            "band_verdict": self.band_verdict,
            "reasons": list(self.reasons),
            "difficulty_hint": self.difficulty_hint,
            "frontier_pass_at_k": self.frontier_pass_at_k(),
            "tier_pass_rates": self.tier_pass_rates(),
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CalibrationReport:
        models = data.get("models", [])
        provenance = data.get("provenance")
        reasons = data.get("reasons", [])
        details = data.get("details", {})
        return cls(
            language=str(data["language"]),
            models=[
                ModelSolveRecord.from_dict(m) for m in models if isinstance(m, dict)
            ]
            if isinstance(models, list)
            else [],
            k=_coerce_int(data.get("k", 0)),
            irt_difficulty=_coerce_float(data.get("irt_difficulty", 0.0)),
            irt_discrimination=_coerce_float(data.get("irt_discrimination", 0.0)),
            band_verdict=str(data.get("band_verdict", "pending")),
            reasons=[str(r) for r in reasons] if isinstance(reasons, list) else [],
            difficulty_hint=str(data.get("difficulty_hint", "")),
            provenance=Provenance.from_dict(provenance)
            if isinstance(provenance, dict)
            else None,
            details=dict(details) if isinstance(details, dict) else {},
        )


class ExportGateError(ModelError):
    """Raised when a :class:`ForgeTask` is assembled from non-shippable gates.

    The architecture export invariant (S3 data model): a ``ForgeTask`` may only be
    created when ``OracleReport.verdict == 'pass'`` AND
    ``CalibrationReport.band_verdict == 'keep'``. The gate is enforced here, at
    assembly, so a half-built shippable object can never reach the writer (an
    oracle pass alone is necessary but NOT sufficient).
    """


@dataclass
class ForgeTask:
    """Stage 5 artifact: the shippable task = the whole verified pipeline bundle.

    A :class:`ForgeTask` bundles the env image, the by-construction
    :class:`Candidate` (mutation + gold), the agent-facing :class:`GeneratedSpec`,
    the :class:`OracleReport` (the 100%-verifiable contract), and the
    :class:`CalibrationReport` (the hard-for-LLMs evidence), plus the
    export-ready ``fail_to_pass``/``pass_to_pass`` selection commands (the FULL
    hidden ``test_files[]`` suite, NOT just the original F2P) and full
    :class:`Provenance`.

    Invariant (enforced fail-fast in :meth:`__post_init__`): the task can exist
    ONLY when ``oracle_report.verdict == 'pass'`` AND
    ``calibration_report.band_verdict == 'keep'``. A rejected or
    calibration-dropped candidate raises :class:`ExportGateError` at assembly and
    never reaches the writer.
    """

    task_id: str
    repo: str
    repo_url: str
    base_commit: str
    language: str
    generator: str
    candidate: Candidate
    spec: GeneratedSpec
    oracle_report: OracleReport
    calibration_report: CalibrationReport
    env_image: EnvImage
    install_commands: list[str]
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    provenance: Provenance
    created_at: str = ""

    def __post_init__(self) -> None:
        if not str(self.task_id).strip():
            raise ModelError("ForgeTask.task_id must be non-empty")
        if self.language not in SUPPORTED_LANGUAGES:
            raise ModelError(
                f"ForgeTask.language must be one of {SUPPORTED_LANGUAGES}; "
                f"got {self.language!r}"
            )
        # Fail-fast export gate: oracle pass is necessary but NOT sufficient --
        # calibration must also keep the candidate (architecture S3 invariant).
        if self.oracle_report.verdict != "pass":
            raise ExportGateError(
                "ForgeTask refused: OracleReport.verdict is "
                f"{self.oracle_report.verdict!r} (reasons="
                f"{list(self.oracle_report.reasons)}); a rejected candidate is "
                "never assembled into a shippable task"
            )
        if self.calibration_report.band_verdict != "keep":
            raise ExportGateError(
                "ForgeTask refused: oracle passed but CalibrationReport.band_verdict "
                f"is {self.calibration_report.band_verdict!r}; an oracle pass is "
                "necessary but a calibration 'keep' is also required to ship"
            )
        if not self.fail_to_pass:
            raise ModelError(
                "ForgeTask.fail_to_pass must be non-empty (the hidden F2P suite)"
            )
        if not self.created_at:
            self.created_at = _utc_now_iso()

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "repo": self.repo,
            "repo_url": self.repo_url,
            "base_commit": self.base_commit,
            "language": self.language,
            "generator": self.generator,
            "candidate": self.candidate.to_dict(),
            "spec": self.spec.to_dict(),
            "oracle_report": self.oracle_report.to_dict(),
            "calibration_report": self.calibration_report.to_dict(),
            "env_image": self.env_image.to_dict(),
            "install_commands": list(self.install_commands),
            "fail_to_pass": list(self.fail_to_pass),
            "pass_to_pass": list(self.pass_to_pass),
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ForgeTask:
        install = data.get("install_commands", [])
        f2p = data.get("fail_to_pass", [])
        p2p = data.get("pass_to_pass", [])
        return cls(
            task_id=str(data["task_id"]),
            repo=str(data.get("repo", "")),
            repo_url=str(data.get("repo_url", "")),
            base_commit=str(data.get("base_commit", "")),
            language=str(data["language"]),
            generator=str(data.get("generator", "")),
            candidate=Candidate.from_dict(data["candidate"]),  # type: ignore[arg-type]
            spec=GeneratedSpec.from_dict(data["spec"]),  # type: ignore[arg-type]
            oracle_report=OracleReport.from_dict(data["oracle_report"]),  # type: ignore[arg-type]
            calibration_report=CalibrationReport.from_dict(
                data["calibration_report"]  # type: ignore[arg-type]
            ),
            env_image=EnvImage.from_dict(data["env_image"]),  # type: ignore[arg-type]
            install_commands=[str(c) for c in install]
            if isinstance(install, list)
            else [],
            fail_to_pass=[str(c) for c in f2p] if isinstance(f2p, list) else [],
            pass_to_pass=[str(c) for c in p2p] if isinstance(p2p, list) else [],
            provenance=Provenance.from_dict(data["provenance"]),  # type: ignore[arg-type]
            created_at=str(data.get("created_at", "")),
        )
