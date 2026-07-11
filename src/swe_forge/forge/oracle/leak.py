"""Leak audit + sanitize gate: no oracle/solution content in the shipped tree.

The sixth and final oracle-hardening gate (architecture S6, Stage 3.6). The
earlier gates (establish -> flakiness -> mutation -> differential -> alt-correct)
prove the hidden suite is a *deterministic, mutation-adequate, gold-unique,
non-over-fit* FAIL->PASS contract. This gate proves the **agent-facing tree** the
solver will actually receive contains NO answer: no oracle/solution content (the
gold ``oracle_patch`` text, restored gold source, or hidden-test answers) and
none of the hidden F2P/P2P test *bodies* (only the statement + requirements +
interface published by the spec are ever visible).

It wires the repository's existing, battle-tested leak tooling into the pipeline:

* :func:`swe_forge.synthetic.leak_auditor.audit_patch_leaks` - the oracle-snippet
  + forbidden-artifact static scan.
* :func:`swe_forge.synthetic.sanitizer.sanitize_tree` - strips build/cache
  artifacts that should never ship and could carry gold.

Flow (*the teacher proposes, deterministic execution disposes* - here the
disposer is a deterministic static audit, no LLM):

1. **Materialize** the agent-facing tree: the *broken* tree (forward
   ``mutation_patch`` applied) exported from the candidate's green ``EnvImage``,
   with the internal harness artifacts (``.git``, the patch-staging dir) excluded
   and the hidden tests removed (synthesized hidden tests are never part of the
   checkout anyway).
2. **Normalize**: always strip build/cache artifacts via ``sanitize_tree`` (a
   ``.pyc`` or ``build/`` copy is never shipped and could carry gold).
3. **Audit**: run ``audit_patch_leaks`` (oracle-snippet + forbidden-artifact) and
   additionally confirm no hidden-test path/body is present.
4. **Decide**: a clean tree passes (``leak_audit == "clean"``). On a detected
   leak the gate strips it when it is a *safely-removable standalone artifact*
   (a stray ``oracle.patch``/``solution.patch``, a hidden-test file copy, a
   build/cache artifact) and re-audits; if the tree comes back clean it passes,
   recording the detected markers (``leak_audit == "sanitized: ..."``). Otherwise
   (oracle/answer content embedded in a legitimate source file that cannot be
   removed without breaking the repo) it **rejects** with an attributable
   ``leak_detected`` reason listing the residual markers.
"""

from __future__ import annotations

import base64
import contextlib
import io
import re
import shlex
import shutil
import tarfile
import tempfile
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from swe_forge.forge.adapters import LanguageAdapter, build_default_registry
from swe_forge.forge.models import (
    Candidate,
    EnvImage,
    OracleReport,
    OracleTestFile,
    Provenance,
    require_green_baseline,
)
from swe_forge.forge.oracle.establish import (
    DockerOracleRecipe,
    TreeState,
    _PATCH_DIR,
)
from swe_forge.synthetic.leak_auditor import audit_patch_leaks
from swe_forge.synthetic.sanitizer import sanitize_tree

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from swe_forge.execution.docker_client import DockerClient

#: Attributable reject reason prefix (stable key the contract/CLI gate on).
REASON_LEAK = "leak_detected"

#: Minimum stripped-line length treated as a "significant" leak snippet. Mirrors
#: the threshold the underlying ``leak_auditor`` uses so the two scans agree.
_MIN_SNIPPET_LEN = 24

#: Default forbidden standalone artifact filenames (the gold/answer must never
#: ship as a loose file). Mirrors ``leak_auditor.audit_patch_leaks``'s defaults.
DEFAULT_FORBIDDEN_FILENAMES: tuple[str, ...] = (
    "removed.patch",
    "oracle.patch",
    "solution.patch",
    "gold.patch",
)

# Binary/oversized files are skipped by the content scan (same spirit as the
# underlying auditor) to keep the audit fast and deterministic.
_SKIP_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip", ".pyc", ".pyo"}
)
_MAX_SCAN_BYTES = 1_000_000


class LeakError(RuntimeError):
    """Raised for an unrecoverable failure while driving the leak gate."""


# --------------------------------------------------------------------------- #
# Findings / audit value types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LeakFinding:
    """One detected leak in the agent-facing tree.

    ``marker`` is a stable kind key (``oracle_snippet`` | ``forbidden_artifact`` |
    ``build_cache_artifact`` | ``hidden_test_file`` | ``hidden_test_body`` |
    ``other``); ``path`` is the repo-relative file it was found in (empty when not
    file-scoped); ``removable`` is ``True`` only for a *standalone* artifact that
    can be deleted without breaking the repo (an embedded snippet in a legitimate
    source file is NOT removable). ``detail`` is a short human-readable note.
    """

    path: str
    marker: str
    removable: bool = False
    detail: str = ""

    def describe(self) -> str:
        """A stable, log-safe one-line description (``marker: path``)."""
        return f"{self.marker}: {self.path}" if self.path else self.marker

    def summary(self) -> dict[str, object]:
        return {
            "path": self.path,
            "marker": self.marker,
            "removable": self.removable,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LeakAudit:
    """The outcome of one static scan over the agent-facing tree."""

    findings: tuple[LeakFinding, ...] = ()

    @property
    def is_clean(self) -> bool:
        """``True`` iff the tree carries no detectable oracle/answer leak."""
        return not self.findings

    def markers(self) -> list[str]:
        """Stable, sorted, de-duplicated marker descriptions for reporting."""
        return sorted({f.describe() for f in self.findings})

    def removable_findings(self) -> tuple[LeakFinding, ...]:
        return tuple(f for f in self.findings if f.removable)

    def summary(self) -> dict[str, object]:
        return {
            "clean": self.is_clean,
            "findings": [f.summary() for f in self.findings],
            "markers": self.markers(),
        }


@dataclass
class LeakOutcome:
    """The result of the leak gate (folded into an :class:`OracleReport`)."""

    verdict: str
    reasons: list[str]
    leak_audit: str
    detected: bool = False
    sanitized: bool = False
    removed: list[str] = field(default_factory=list)
    normalized: list[str] = field(default_factory=list)
    findings_before: list[str] = field(default_factory=list)
    findings_after: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    @property
    def is_pass(self) -> bool:
        return self.verdict == "pass"


# --------------------------------------------------------------------------- #
# Static scan helpers (deterministic; no Docker, no LLM)
# --------------------------------------------------------------------------- #
def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _significant_lines(text: str, *, min_len: int = _MIN_SNIPPET_LEN) -> list[str]:
    """Return the stripped, sufficiently-long lines of ``text`` (leak snippets)."""
    out: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if len(stripped) >= min_len:
            out.append(stripped)
    return out


# Ubiquitous test-framework scaffolding / import lines carry no answer content
# and appear verbatim in a repo's own shipped tests (e.g. every node:test file
# opens with ``const test = require("node:test");``). Matching them would falsely
# flag a legitimate baseline test as a hidden-test-body leak, so they are excluded
# from the hidden-test body scan.
_BOILERPLATE_LINE_RE = re.compile(
    r"^(?:"
    r"import\b"
    r"|from\b.+\bimport\b"
    r"|package\b"
    r"|(?:const|let|var)\s+[\w{},*\s]+=\s*require\("
    r"|require\("
    r"|export\b"
    r"|module\.exports\b"
    r"|['\"]use strict['\"]"
    r")"
)


def _is_boilerplate_line(line: str) -> bool:
    """``True`` for import/require/package scaffolding that carries no answer."""
    return bool(_BOILERPLATE_LINE_RE.match(line.strip()))


def _scan_targets(root: Path) -> list[Path]:
    """Files worth scanning for embedded content (skip binary/oversized)."""
    targets: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in _SKIP_SUFFIXES:
            continue
        try:
            if path.stat().st_size > _MAX_SCAN_BYTES:
                continue
        except OSError:
            continue
        targets.append(path)
    return targets


def _classify_patch_finding(finding: str) -> LeakFinding:
    """Map a ``leak_auditor.audit_patch_leaks`` finding string to a structured one."""
    forbidden = "Forbidden artifact present: "
    cache = "Leaky build/cache artifact present: "
    snippet = "Oracle snippet appears in "
    if finding.startswith(forbidden):
        path = finding[len(forbidden) :].strip()
        return LeakFinding(path=path, marker="forbidden_artifact", removable=True)
    if finding.startswith(cache):
        path = finding[len(cache) :].strip()
        return LeakFinding(path=path, marker="build_cache_artifact", removable=True)
    if finding.startswith(snippet):
        path = finding[len(snippet) :].strip()
        return LeakFinding(
            path=path,
            marker="oracle_snippet",
            removable=False,
            detail="restored gold/oracle content present in a shipped source file",
        )
    return LeakFinding(path="", marker="other", removable=False, detail=finding)


def _scan_hidden_tests(
    root: Path, hidden_test_files: Sequence[OracleTestFile]
) -> list[LeakFinding]:
    """Detect any hidden-test path or body present in the agent-facing tree."""
    findings: list[LeakFinding] = []
    body_lines: dict[str, list[str]] = {}
    for tf in hidden_test_files:
        rel = tf.path
        if (root / rel).is_file():
            findings.append(
                LeakFinding(
                    path=rel,
                    marker="hidden_test_file",
                    removable=True,
                    detail="hidden test body present in the agent-facing tree",
                )
            )
        lines = _significant_lines(tf.content)
        if lines:
            body_lines[rel] = [
                _norm_ws(line) for line in lines if not _is_boilerplate_line(line)
            ]
            if not body_lines[rel]:
                del body_lines[rel]

    if not body_lines:
        return findings

    own_paths = {tf.path for tf in hidden_test_files}
    for path in _scan_targets(root):
        rel = str(path.relative_to(root))
        if rel in own_paths:
            continue  # the hidden-test file itself is handled above
        try:
            text = _norm_ws(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        for test_rel, lines in body_lines.items():
            # Agent-visible regression tests can legitimately share a generic
            # assertion with a generated hidden test. A hidden-test *body* leak
            # means its complete meaningful body was copied into the agent tree,
            # so require every non-boilerplate line rather than rejecting on a
            # single incidental overlap. A one-line hidden test remains fully
            # protected, and a copied multi-line test still fails closed.
            if lines and all(line and line in text for line in lines):
                findings.append(
                    LeakFinding(
                        path=rel,
                        marker="hidden_test_body",
                        removable=False,
                        detail=f"hidden test {test_rel} body content appears here",
                    )
                )
                break
    return findings


def audit_agent_tree(
    root: Path | str,
    *,
    oracle_patch: str,
    hidden_test_files: Sequence[OracleTestFile] = (),
    forbidden_filenames: Sequence[str] | None = None,
) -> LeakAudit:
    """Statically audit the agent-facing tree for oracle/answer leaks.

    Reuses :func:`audit_patch_leaks` for the oracle-snippet + forbidden-artifact
    scan and supplements it with hidden-test path/body detection. Returns a
    :class:`LeakAudit` whose findings are classified by removability so the gate
    can decide between sanitize and reject.
    """
    root_path = Path(root).resolve()
    forbidden = list(forbidden_filenames or DEFAULT_FORBIDDEN_FILENAMES)

    base = audit_patch_leaks(
        root_path, oracle_patch=oracle_patch, forbidden_filenames=forbidden
    )
    findings: list[LeakFinding] = [_classify_patch_finding(f) for f in base.findings]
    findings.extend(_scan_hidden_tests(root_path, hidden_test_files))

    # De-duplicate while keeping a deterministic order (by marker then path).
    seen: set[tuple[str, str]] = set()
    unique: list[LeakFinding] = []
    for finding in sorted(findings, key=lambda f: (f.marker, f.path)):
        key = (finding.marker, finding.path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return LeakAudit(findings=tuple(unique))


def normalize_agent_tree(root: Path | str) -> list[str]:
    """Strip build/cache artifacts that must never ship (reuses ``sanitize_tree``).

    These are normalized away *before* the leak audit so incidental build noise
    (a ``__pycache__`` from a P2P run, a ``build/`` dir) is never mistaken for a
    planted leak. Returns the repo-relative paths removed.
    """
    root_path = Path(root).resolve()
    result = sanitize_tree(root_path)
    removed: list[str] = []
    for path in result.removed_paths:
        with contextlib.suppress(ValueError):
            removed.append(str(path.resolve().relative_to(root_path)))
    return sorted(removed)


def sanitize_leaks(root: Path | str, audit: LeakAudit) -> list[str]:
    """Delete the safely-removable leak artifacts the audit found.

    Removable findings are standalone artifacts (a stray ``oracle.patch``, a
    hidden-test file copy, a build/cache artifact) that can be deleted without
    breaking the repo. Embedded oracle/answer content in a legitimate source file
    is NEVER deleted here (that path leads to a reject). Returns the paths removed.
    """
    root_path = Path(root).resolve()
    removed: list[str] = []
    for finding in audit.removable_findings():
        if not finding.path:
            continue
        target = (root_path / finding.path).resolve()
        if root_path not in (target, *target.parents):
            continue  # never follow a path outside the tree
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
        removed.append(finding.path)
    return sorted(set(removed))


def assess_leak(
    before: LeakAudit,
    after: LeakAudit,
    *,
    removed: Sequence[str] = (),
    normalized: Sequence[str] = (),
) -> LeakOutcome:
    """Decide the gate verdict from the pre/post-sanitize audits.

    Clean before -> pass (``leak_audit == "clean"``). Detected then clean after
    sanitize -> pass listing the detected markers (``leak_audit ==
    "sanitized: ..."``). A residual leak that cannot be safely removed -> reject
    citing it (``leak_audit == "leak: ..."``).
    """
    details: dict[str, object] = {
        "stage": "leak",
        "before": before.summary(),
        "after": after.summary(),
        "normalized": list(normalized),
        "removed": list(removed),
    }

    if before.is_clean:
        return LeakOutcome(
            verdict="pass",
            reasons=[],
            leak_audit="clean",
            detected=False,
            sanitized=False,
            normalized=list(normalized),
            findings_before=[],
            findings_after=[],
            details=details,
        )

    before_markers = before.markers()
    if after.is_clean:
        return LeakOutcome(
            verdict="pass",
            reasons=[],
            leak_audit="sanitized: " + "; ".join(before_markers),
            detected=True,
            sanitized=True,
            removed=list(removed),
            normalized=list(normalized),
            findings_before=before_markers,
            findings_after=[],
            details=details,
        )

    residual = after.markers()
    reason = (
        f"{REASON_LEAK}: the agent-facing tree still contains oracle/solution "
        f"content after sanitization ({residual}); it cannot be safely removed "
        "without breaking the repo -> reject"
    )
    return LeakOutcome(
        verdict="reject",
        reasons=[reason],
        leak_audit="leak: " + "; ".join(residual),
        detected=True,
        sanitized=bool(removed),
        removed=list(removed),
        normalized=list(normalized),
        findings_before=before_markers,
        findings_after=residual,
        details=details,
    )


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def _tool_versions(extra: dict[str, str] | None = None) -> dict[str, str]:
    versions: dict[str, str] = {}
    with contextlib.suppress(metadata.PackageNotFoundError):
        versions["litellm"] = metadata.version("litellm")
    if extra:
        versions.update(extra)
    return versions


def build_leak_report(
    candidate: Candidate,
    prior_report: OracleReport,
    outcome: LeakOutcome,
    *,
    env_image: EnvImage | None = None,
    extra_details: dict[str, object] | None = None,
) -> OracleReport:
    """Fold a :class:`LeakOutcome` into the running :class:`OracleReport`.

    Carries the establish + flakiness + mutation + differential + alt-correct
    fields forward unchanged, sets ``leak_audit``, and sets the terminal verdict
    (``pass`` when the shipped tree is clean - possibly after stripping a
    removable artifact; ``reject`` with an attributable ``leak_detected`` reason
    when an unremovable leak remains).
    """
    details: dict[str, object] = dict(prior_report.details)
    details["leak"] = outcome.details
    if env_image is not None:
        details.setdefault("env_image", env_image.image_tag)
    if extra_details:
        details.update(extra_details)

    base_prov = prior_report.provenance
    provenance = Provenance(
        generator=candidate.generator,
        seed=candidate.provenance.seed,
        language=candidate.language,
        tool_versions=dict(base_prov.tool_versions) if base_prov else _tool_versions(),
        details={
            "stage": "oracle.leak",
            "detected": outcome.detected,
            "sanitized": outcome.sanitized,
            "removed": list(outcome.removed),
            "normalized": list(outcome.normalized),
            "findings_before": list(outcome.findings_before),
            "findings_after": list(outcome.findings_after),
        },
    )

    return OracleReport(
        language=prior_report.language,
        generator=prior_report.generator,
        verdict=outcome.verdict,
        reasons=list(outcome.reasons),
        fail_to_pass=list(prior_report.fail_to_pass),
        pass_to_pass=list(prior_report.pass_to_pass),
        test_files=list(prior_report.test_files),
        flakiness_runs=prior_report.flakiness_runs,
        mutants_total=prior_report.mutants_total,
        mutants_killed=prior_report.mutants_killed,
        final_mutation_evidence=prior_report.final_mutation_evidence,
        multifault_evidence=prior_report.multifault_evidence,
        differential_pass=prior_report.differential_pass,
        alt_correct_accepted=prior_report.alt_correct_accepted,
        leak_audit=outcome.leak_audit,
        provenance=provenance,
        details=details,
        protected_alt_correct_audit=prior_report.protected_alt_correct_audit,
        protected_teacher_transport_receipts=list(
            prior_report.protected_teacher_transport_receipts
        ),
    )


# --------------------------------------------------------------------------- #
# Agent-facing tree provider (Docker-backed in production)
# --------------------------------------------------------------------------- #
@runtime_checkable
class AgentTreeProvider(Protocol):
    """Materializes the agent-facing tree as a local directory to audit.

    ``open()`` is an async context manager yielding the path to a local copy of
    the broken (mutation-applied) tree with harness artifacts and hidden tests
    excluded; the directory is cleaned up on exit.
    """

    @property
    def language(self) -> str: ...

    def open(self) -> "contextlib.AbstractAsyncContextManager[Path]": ...


class DockerAgentTreeProvider:
    """Exports the broken tree from a candidate's ``EnvImage`` to a local dir.

    Opens a fresh ``--rm`` :class:`~swe_forge.execution.sandbox.DockerSandbox` on
    the green EnvImage (the repo already checked out, pristine = gold), applies
    the forward ``mutation_patch`` (the state the solver receives), removes any
    hidden-test files, and streams a ``tar`` of the workspace (excluding ``.git``
    and the patch-staging dir) out as base64, extracting it to a temp directory
    the gate audits. The temp dir is removed on exit even on failure.
    """

    def __init__(
        self,
        candidate: Candidate,
        env_image: EnvImage,
        *,
        hidden_test_paths: Sequence[str] = (),
        command_timeout: float = 600.0,
        docker_client: "DockerClient | None" = None,
    ) -> None:
        self._candidate = candidate
        self._env_image = env_image
        self._hidden_test_paths = list(hidden_test_paths)
        self._timeout = command_timeout
        self._docker_client = docker_client

    @property
    def language(self) -> str:
        return self._candidate.language

    @contextlib.asynccontextmanager
    async def open(self) -> "AsyncIterator[Path]":
        from swe_forge.execution.docker_client import DockerClient
        from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

        client = self._docker_client or DockerClient()
        config = SandboxConfig(
            name="swe-forge-oracle-leak",
            image=self._env_image.image_tag,
            workspace_dir=self._env_image.workspace_dir,
            command_timeout=self._timeout,
        )
        sandbox = DockerSandbox(client, config)
        tmp_root = Path(tempfile.mkdtemp(prefix="swe-forge-leak-"))
        try:
            async with sandbox:
                recipe = DockerOracleRecipe(
                    sandbox,
                    language=self._candidate.language,
                    workspace_dir=self._env_image.workspace_dir,
                    mutation_patch=self._candidate.mutation_patch,
                    oracle_patch=self._candidate.oracle_patch,
                    p2p_command=self._env_image.baseline_test_command,
                    command_timeout=self._timeout,
                )
                # The solver receives the BROKEN tree (mutation applied).
                await recipe.set_state(TreeState.BROKEN)
                for rel in self._hidden_test_paths:
                    await sandbox.run_command(f"rm -f {shlex.quote(rel)}")
                await self._export_tree(sandbox, tmp_root)
                yield tmp_root
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    async def _export_tree(self, sandbox: object, dest: Path) -> None:
        workspace = self._env_image.workspace_dir
        cmd = (
            f"cd {shlex.quote(workspace)} && "
            f"tar -czf - --exclude=./.git --exclude=./{_PATCH_DIR} . | base64 -w0"
        )
        result = await sandbox.run_command(cmd, timeout=self._timeout)  # type: ignore[attr-defined]
        if result.exit_code != 0:
            raise LeakError(
                "failed to export the agent-facing tree from the sandbox: "
                f"{(result.stderr or result.stdout or '').strip()[:500]}"
            )
        try:
            payload = base64.b64decode(result.stdout)
        except (ValueError, TypeError) as exc:
            raise LeakError(f"could not decode the exported tree: {exc}") from exc
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            archive.extractall(dest, filter="data")


# --------------------------------------------------------------------------- #
# Top-level gate
# --------------------------------------------------------------------------- #
async def run_leak_gate(
    candidate: Candidate,
    env_image: EnvImage,
    prior_report: OracleReport,
    *,
    provider: AgentTreeProvider | None = None,
    adapter: LanguageAdapter | None = None,
    sanitize: bool = True,
    forbidden_filenames: Collection[str] | None = None,
    docker_client: "DockerClient | None" = None,
    command_timeout: float = 600.0,
) -> OracleReport:
    """Run the leak audit + sanitize gate over the candidate's agent-facing tree.

    A green baseline is a hard precondition and the prior gate (alt-correct) must
    have passed. Materializes the broken tree (via ``provider``, Docker-backed by
    default), normalizes build/cache artifacts, audits for oracle/solution and
    hidden-test leaks, sanitizes removable artifacts when ``sanitize`` is set, and
    returns the extended :class:`OracleReport` (with ``leak_audit`` set and the
    terminal verdict).
    """
    require_green_baseline(env_image)
    if prior_report.verdict != "pass":
        raise LeakError(
            "leak gate requires a passing prior (alt-correct) report; got verdict "
            f"{prior_report.verdict!r}"
        )

    if adapter is None:
        adapter = build_default_registry().get(candidate.language)

    hidden_test_files = list(prior_report.test_files)
    if provider is None:
        provider = DockerAgentTreeProvider(
            candidate,
            env_image,
            hidden_test_paths=[tf.path for tf in hidden_test_files],
            command_timeout=command_timeout,
            docker_client=docker_client,
        )

    forbidden = (
        list(forbidden_filenames)
        if forbidden_filenames is not None
        else list(DEFAULT_FORBIDDEN_FILENAMES)
    )

    async with provider.open() as tree_root:
        normalized = normalize_agent_tree(tree_root)
        before = audit_agent_tree(
            tree_root,
            oracle_patch=candidate.oracle_patch,
            hidden_test_files=hidden_test_files,
            forbidden_filenames=forbidden,
        )
        removed: list[str] = []
        after = before
        if not before.is_clean and sanitize:
            removed = sanitize_leaks(tree_root, before)
            after = audit_agent_tree(
                tree_root,
                oracle_patch=candidate.oracle_patch,
                hidden_test_files=hidden_test_files,
                forbidden_filenames=forbidden,
            )
        outcome = assess_leak(before, after, removed=removed, normalized=normalized)

    return build_leak_report(candidate, prior_report, outcome, env_image=env_image)
