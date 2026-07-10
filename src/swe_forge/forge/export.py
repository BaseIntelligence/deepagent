"""Stage 5 export: assemble + ship a verified :class:`ForgeTask`.

This is the final pipeline stage. It turns the verified artifacts of the earlier
stages (the by-construction :class:`Candidate`, the agent-facing
:class:`GeneratedSpec`, the 100%-verifiable :class:`OracleReport`, and the
hard-for-LLMs :class:`CalibrationReport`) into a shippable benchmark task and
emits it three ways:

* a self-contained ``tasks/<id>/`` workspace (``workspace.yaml`` + gold
  ``patch.diff`` + mutation ``deletion_patch.diff`` + a non-empty ``tests/`` of
  the hidden suite + an executable, robust ``evaluate.sh``), and
* one record per kept task appended to the ``jsonl`` and ``parquet`` datasets
  (reusing :mod:`swe_forge.export.jsonl` / :mod:`swe_forge.export.parquet`).

The non-negotiable invariants this module enforces:

* **Fail-fast export gate.** A task is assembled ONLY when
  ``OracleReport.verdict == 'pass'`` AND ``CalibrationReport.band_verdict ==
  'keep'``. An oracle pass alone is necessary but NOT sufficient -- the caller
  passes the calibration verdict via
  :func:`~swe_forge.forge.oracle.pipeline.ensure_oracle_exportable`, so an
  oracle-pass + calibration-drop candidate is refused at assembly (never a
  half-built shippable object).
* **Full hidden suite.** Survivor-killing tests synthesized by the
  differential/mutation gates live in ``OracleReport.test_files[]`` (NOT only
  ``fail_to_pass``), so the exported ``tests/`` AND the generated ``evaluate.sh``
  enforce the FULL ``test_files[]`` set, not just the original F2P.
* **No gold leak.** The gold patch + hidden tests ship in the benchmark-only
  forge location, never inside the agent-facing repo tree; the exported tree
  passes a leak audit (oracle-snippet, hidden-test body, forbidden/cache
  artifact) AND a ``.git`` history check (gold must NOT be recoverable via
  ``git show HEAD~1``: a shipped repo tree is re-init'd to a single orphan
  commit). A planted leak blocks shipping.
* **Deterministic + idempotent.** Ids are a stable function of (repo, generator,
  seed, target); re-export with overwrite reproduces the workspace without
  duplicating dataset rows; a failed mid-write leaves no partial ``tasks/<id>/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path, PurePosixPath

import yaml  # type: ignore[import-untyped]

from swe_forge.forge.adapters import LanguageAdapter, build_default_registry
from swe_forge.forge.models import (
    CalibrationReport,
    Candidate,
    EnvImage,
    ExportGateError,
    ForgeTask,
    GeneratedSpec,
    OracleReport,
    OracleTestFile,
    Provenance,
    _utc_now_iso,
)
from swe_forge.forge.oracle.leak import audit_agent_tree
from swe_forge.forge.oracle.pipeline import ExportRefusedError, ensure_oracle_exportable
from swe_forge.forge import publication
from swe_forge.forge.publication import (
    PublicationEntry,
    PublicationError,
    canonical_task_payload,
    publish_generation,
)
from swe_forge.swe.models import (
    SweTask,
    SweTaskStatus,
    validate_repo_name,
)
from swe_forge.synthetic.models import LeakAuditResult

#: Standard in-container paths (mirrors the existing workspace export layout).
REPO_DIR = "/workspace/repo"
FORGE_DIR = "/workspace/forge"
TESTS_DIR = f"{FORGE_DIR}/tests"

#: The benchmark-only (NOT agent-visible) files/dirs in an exported task dir.
#: These carry the gold/solution + hidden suite and are excluded from the leak
#: audit's agent-facing view (patch/test diffs excluded by design).
_BENCHMARK_ONLY_NAMES = frozenset(
    {
        "patch.diff",
        "deletion_patch.diff",
        "test_patch.diff",
        "evaluate.sh",
        "run_tests.sh",
        "provenance.json",
    }
)
_BENCHMARK_ONLY_DIRS = frozenset({"tests"})

#: Deterministic git identity for the orphan-commit re-init (never the host user).
_GIT_ENV = {
    "GIT_AUTHOR_NAME": "swe-forge",
    "GIT_AUTHOR_EMAIL": "forge@local",
    "GIT_COMMITTER_NAME": "swe-forge",
    "GIT_COMMITTER_EMAIL": "forge@local",
}

#: Per-direct-export store.  It lives beside, not inside, the task-root
#: enumeration surface so generic task consumers never mistake it for a task.
_DIRECT_STORE_DIR = ".forge-task-publications"
_DIRECT_GENERATIONS_DIR = "generations"
_DIRECT_CURRENT_LINK = "current"


class ExportError(RuntimeError):
    """Raised for an unrecoverable failure while exporting a task."""


# --------------------------------------------------------------------------- #
# Deterministic id
# --------------------------------------------------------------------------- #
def _slug(value: str, *, max_len: int = 48) -> str:
    """A stable, filesystem-safe lowercase slug of ``value``."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned[:max_len] or "task"


def forge_task_id(
    repo: str,
    generator: str,
    seed: int,
    target_files: Sequence[str],
    target_symbols: Sequence[str] = (),
) -> str:
    """A stable, unique task id for a ``(repo, generator, seed, target)`` tuple.

    Deterministic: the same inputs always yield the same id (so re-exporting a
    task overwrites in place rather than duplicating). Unique: distinct tuples map
    to distinct ids via a sha256 digest over the canonicalized inputs.
    """
    payload = "\n".join(
        [
            repo or "",
            generator or "",
            str(seed),
            ",".join(sorted(str(f) for f in target_files)),
            ",".join(sorted(str(s) for s in target_symbols)),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{_slug(repo)}__{generator}__{digest}"


def _repo_slug(repo_url: str, fallback_id: str) -> str:
    """Derive a dataset-friendly ``owner/repo`` slug from a clone URL.

    A GitHub-style URL maps to its ``owner/repo``; anything else falls back to a
    sanitized ``forge/<id>`` so the dataset's repo field is always a valid
    ``owner/repo`` (the dataset model validates this format).
    """
    url = (repo_url or "").strip()
    match = re.search(r"github\.com[/:]+([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if match:
        candidate = f"{match.group(1)}/{match.group(2)}"
        try:
            return validate_repo_name(candidate)
        except ValueError:
            pass
    return f"forge/{_slug(fallback_id or url or 'repo')}"


# --------------------------------------------------------------------------- #
# Full hidden suite (test_files[] -> selection commands)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HiddenRun:
    """One hidden test: its repo-relative path, body, and selection command."""

    path: str
    content: str
    command: str


def canonical_hidden_test_path(path: str) -> str:
    """Return a canonical, workspace-contained POSIX hidden-test path.

    Hidden tests are copied to ``tasks/<id>/tests/<path>`` and then rendered
    into ``evaluate.sh``.  Accepting an absolute path, traversal component, or
    non-canonical spelling would let a malicious oracle artifact write outside
    the workspace or make the evaluator address a different file.  Validate at
    the export boundary rather than relying on a previous oracle gate.
    """
    raw = str(path)
    if not raw or raw.strip() != raw:
        raise ExportError(
            "hidden test path must be a non-empty canonical relative path"
        )
    if "\\" in raw:
        raise ExportError(f"hidden test path must use POSIX separators: {raw!r}")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or not candidate.parts:
        raise ExportError(f"hidden test path must be relative: {raw!r}")
    if any(part in ("", ".", "..") for part in candidate.parts):
        raise ExportError(f"hidden test path may not traverse directories: {raw!r}")
    canonical = candidate.as_posix()
    if raw != canonical:
        raise ExportError(f"hidden test path must be canonical: {raw!r}")
    return canonical


def validate_hidden_test_paths(test_files: Sequence[OracleTestFile]) -> None:
    """Reject unsafe or duplicate hidden-test destinations before any write."""
    seen: set[str] = set()
    for test_file in test_files:
        path = canonical_hidden_test_path(test_file.path)
        if path in seen:
            raise ExportError(f"duplicate hidden test path: {path!r}")
        seen.add(path)


def hidden_runs(
    adapter: LanguageAdapter,
    fail_to_pass: Sequence[str],
    test_files: Sequence[OracleTestFile],
) -> list[HiddenRun]:
    """Map the FULL hidden ``test_files[]`` set to runnable selection commands.

    Each test file keeps its established F2P selection command when one references
    its path; any other hidden test (e.g. a mutation/differential-gate
    survivor-killing test recorded only in ``test_files``) gets a selection
    command via the adapter so the whole hidden suite is enforced -- not just the
    original F2P.
    """
    validate_hidden_test_paths(test_files)
    runs: list[HiddenRun] = []
    for tf in test_files:
        if not tf.content:
            continue
        path = canonical_hidden_test_path(tf.path)
        command: str | None = None
        for cmd in fail_to_pass:
            if path and path in cmd and shlex.quote(path) == path:
                command = cmd
                break
        if command is None:
            command = adapter.test_command((path,))
        runs.append(HiddenRun(path=path, content=tf.content, command=command))
    return runs


def build_full_fail_to_pass(
    adapter: LanguageAdapter,
    fail_to_pass: Sequence[str],
    test_files: Sequence[OracleTestFile],
) -> list[str]:
    """The deduplicated selection commands covering the FULL hidden suite."""
    commands = list(fail_to_pass)
    for run in hidden_runs(adapter, fail_to_pass, test_files):
        if run.command not in commands:
            commands.append(run.command)
    return list(dict.fromkeys(commands))


# --------------------------------------------------------------------------- #
# Assembly (fail-fast gate)
# --------------------------------------------------------------------------- #
def _tool_versions(base: dict[str, str] | None = None) -> dict[str, str]:
    versions: dict[str, str] = dict(base or {})
    if "litellm" not in versions:
        try:
            versions["litellm"] = metadata.version("litellm")
        except metadata.PackageNotFoundError:
            pass
    return versions


def _build_provenance(
    candidate: Candidate,
    oracle_report: OracleReport,
    calibration_report: CalibrationReport,
) -> Provenance:
    base_prov = candidate.provenance
    raw_teacher_gates = oracle_report.details.get("teacher_gates")
    teacher_gates = (
        dict(raw_teacher_gates) if isinstance(raw_teacher_gates, dict) else {}
    )
    details: dict[str, object] = {
        "generator": candidate.generator,
        "seed": base_prov.seed,
        "oracle_verdict": oracle_report.verdict,
        "band_verdict": calibration_report.band_verdict,
        "mutants_total": oracle_report.mutants_total,
        "mutants_killed": oracle_report.mutants_killed,
        "final_mutation_suite_fingerprint": (
            oracle_report.final_mutation_evidence.suite_fingerprint
            if oracle_report.final_mutation_evidence
            else ""
        ),
        "final_mutation_threshold": (
            oracle_report.final_mutation_evidence.threshold
            if oracle_report.final_mutation_evidence
            else 0.0
        ),
        "flakiness_runs": oracle_report.flakiness_runs,
        "differential_pass": oracle_report.differential_pass,
        "alt_correct_accepted": oracle_report.alt_correct_accepted,
        "teacher_gates": teacher_gates,
        "leak_audit": oracle_report.leak_audit,
        # Keep the final per-constituent proof in the shipped provenance. A
        # report consumer can therefore audit every leave-one-broken verdict
        # without access to hidden tests or inverse patch bodies.
        "multifault_completeness": (
            oracle_report.multifault_evidence.to_dict()
            if oracle_report.multifault_evidence
            else None
        ),
        "irt_difficulty": calibration_report.irt_difficulty,
        "irt_discrimination": calibration_report.irt_discrimination,
        "frontier_pass_at_k": calibration_report.frontier_pass_at_k(),
        "tier_pass_rates": calibration_report.tier_pass_rates(),
        "panel": [m.to_dict() for m in calibration_report.models],
    }
    return Provenance(
        generator=candidate.generator,
        seed=base_prov.seed,
        language=candidate.language,
        created_at=_utc_now_iso(),
        tool_versions=_tool_versions(base_prov.tool_versions),
        details=details,
    )


def assemble_forge_task(
    *,
    candidate: Candidate,
    spec: GeneratedSpec,
    oracle_report: OracleReport,
    calibration_report: CalibrationReport,
    env_image: EnvImage,
    repo_url: str,
    base_commit: str = "",
    repo: str | None = None,
    task_id: str | None = None,
    adapter: LanguageAdapter | None = None,
) -> ForgeTask:
    """Assemble a shippable :class:`ForgeTask`, enforcing the export gate fail-fast.

    The architecture invariant is checked BEFORE any object is built: the caller
    passes the calibration verdict to
    :func:`~swe_forge.forge.oracle.pipeline.ensure_oracle_exportable` so an
    oracle-pass + calibration-drop candidate is refused at assembly (raises
    :class:`ExportRefusedError`); :class:`ForgeTask` re-checks the same invariant
    (raising :class:`ExportGateError`) so a half-built shippable object can never
    reach the writer.
    """
    # Fail-fast: oracle pass is necessary but NOT sufficient -- pass the band
    # verdict so an oracle-pass + calibration-drop candidate is refused here.
    ensure_oracle_exportable(
        oracle_report,
        candidate=candidate,
        calibration_kept=calibration_report.is_keep,
    )

    if adapter is None:
        adapter = build_default_registry().get(candidate.language)

    base_commit = base_commit or env_image.commit
    repo = repo or _repo_slug(repo_url, env_image.repo_id)
    seed = candidate.provenance.seed
    if task_id is None:
        task_id = forge_task_id(
            repo,
            candidate.generator,
            seed,
            candidate.target.files,
            candidate.target.symbols,
        )
    _validate_task_id(task_id)

    full_f2p = build_full_fail_to_pass(
        adapter, oracle_report.fail_to_pass, oracle_report.test_files
    )
    pass_to_pass = list(oracle_report.pass_to_pass) or [env_image.baseline_test_command]

    return ForgeTask(
        task_id=task_id,
        repo=repo,
        repo_url=repo_url,
        base_commit=base_commit,
        language=candidate.language,
        generator=candidate.generator,
        candidate=candidate,
        spec=spec,
        oracle_report=oracle_report,
        calibration_report=calibration_report,
        env_image=env_image,
        install_commands=list(env_image.install_commands),
        fail_to_pass=full_f2p,
        pass_to_pass=pass_to_pass,
        provenance=_build_provenance(candidate, oracle_report, calibration_report),
    )


# --------------------------------------------------------------------------- #
# Dataset record (reuse the jsonl/parquet SweTask path)
# --------------------------------------------------------------------------- #
def forge_task_to_swe_task(task: ForgeTask) -> SweTask:
    """Convert a :class:`ForgeTask` to the dataset record type (one row each).

    Reuses the repository's :class:`SweTask` dataset schema so the existing
    jsonl/parquet exporters round-trip the record losslessly (map fields
    ``install_config``/``meta`` and list fields ``fail_to_pass``/``pass_to_pass``
    deserialize back to dict/list values).
    """
    test_files = [
        {"path": tf.path, "content": tf.content}
        for tf in task.oracle_report.test_files
        if tf.content
    ]
    install_config = {
        "install_commands": json.dumps(list(task.install_commands)),
        "language": task.language,
        "base_image": task.env_image.base_image,
    }
    cal = task.calibration_report
    meta = {
        "generator": task.generator,
        "seed": str(task.candidate.provenance.seed),
        "repo_url": task.repo_url,
        "strategy": task.generator,
        "oracle_verdict": task.oracle_report.verdict,
        "band_verdict": cal.band_verdict,
        "irt_difficulty": repr(cal.irt_difficulty),
        "irt_discrimination": repr(cal.irt_discrimination),
        "frontier_pass_at_k": repr(cal.frontier_pass_at_k()),
        "image_tag": task.env_image.image_tag,
        "final_mutation_suite_fingerprint": (
            task.oracle_report.final_mutation_evidence.suite_fingerprint
            if task.oracle_report.final_mutation_evidence
            else ""
        ),
        "created_at": task.created_at,
    }
    return SweTask(
        id=task.task_id,
        repo=task.repo,
        base_commit=task.base_commit,
        language=task.language,
        patch=task.candidate.oracle_patch,
        deletion_patch=task.candidate.mutation_patch,
        fail_to_pass=list(task.fail_to_pass),
        pass_to_pass=list(task.pass_to_pass),
        generated_test_files=test_files,
        install_config=install_config,
        meta=meta,
        source_type=f"synthetic_{task.generator}",
        prompt=task.spec.problem_statement,
        dataset_prompt=task.spec.problem_statement,
        status=SweTaskStatus.EXPORTED,
    )


def export_dataset(
    tasks: Sequence[ForgeTask],
    jsonl_path: Path | str,
    parquet_path: Path | str,
) -> int:
    """Write the kept set to jsonl + parquet (one record per task, full regen).

    Always rewrites both files from the complete kept set so re-export never
    duplicates rows; an empty kept set yields a 0-line jsonl and a valid 0-row
    parquet with the correct schema.
    """
    from swe_forge.export.jsonl import export_jsonl
    from swe_forge.export.parquet import export_parquet

    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ExportError("refusing dataset export with duplicate task ids")
    records = [forge_task_to_swe_task(task) for task in tasks]
    export_jsonl(records, jsonl_path, append=False)
    export_parquet(records, parquet_path)
    return len(records)


# --------------------------------------------------------------------------- #
# evaluate.sh (executable, self-contained, robust apply, FULL hidden suite)
# --------------------------------------------------------------------------- #
def _render_evaluate_script(
    *,
    repo_url: str,
    base_commit: str,
    install_commands: Sequence[str],
    pass_to_pass: Sequence[str],
    runs: Sequence[HiddenRun],
    language: str,
) -> str:
    """Render a forge ``evaluate.sh`` scoring a task ``{"score": 0|1}``.

    Self-contained: clones+checks out the base commit (when the repo is absent),
    applies the mutation/deletion then gold patches with a ``git apply --3way``
    fallback (context-drift tolerant), materializes each hidden test at its
    repo-relative path only around its own run (write -> run -> remove) so the
    whole-suite regression (P2P) never sees the hidden suite, and enforces the
    FULL hidden ``test_files[]`` set in both phases. Python runs defeat CPython's
    second-resolution ``.pyc`` cache (``PYTHONDONTWRITEBYTECODE`` + purge).
    """
    if language == "python":
        purge = (
            'find . -name "__pycache__" -type d -prune -exec rm -rf {} + '
            '2>/dev/null; find . -name "*.pyc" -delete 2>/dev/null; true'
        )
    else:
        purge = "true"

    install_block = (
        "\n".join(f"  {cmd} || true" for cmd in install_commands)
        or "  echo 'no install commands'"
    )

    def f2p_phase(expect_pass: bool) -> str:
        if not runs:
            return "  echo 'no hidden tests'"
        lines: list[str] = []
        for run in runs:
            parent = os.path.dirname(run.path)
            source = f'"$FORGE_PATH/tests"/{shlex.quote(run.path)}'
            destination = f'"$REPO_PATH"/{shlex.quote(run.path)}'
            mkdir = (
                f'  mkdir -p "$REPO_PATH"/{shlex.quote(parent)} 2>/dev/null || true\n'
                if parent
                else ""
            )
            cond = (
                f"if ! {run.command}; then"
                if expect_pass
                else f"if {run.command}; then"
            )
            should = "PASS after patch" if expect_pass else "FAIL before patch"
            lines.append(
                f"  {purge}\n"
                f"{mkdir}"
                f"  cp {source} {destination}\n"
                f"  {cond}\n"
                f"    echo {shlex.quote(f'FAIL: hidden test should {should}: {run.command}')}\n"
                f"    SCORE=0\n"
                f"  fi\n"
                f"  rm -f {destination}"
            )
        return "\n".join(lines)

    def p2p_phase() -> str:
        if not pass_to_pass:
            return "  echo 'no pass_to_pass'"
        lines: list[str] = []
        for cmd in pass_to_pass:
            lines.append(
                f"  {purge}\n"
                f"  if ! {cmd}; then\n"
                f'    echo "FAIL: pass_to_pass should PASS: {cmd}"\n'
                f"    SCORE=0\n"
                f"  fi"
            )
        return "\n".join(lines)

    f2p_before = f2p_phase(expect_pass=False)
    f2p_after = f2p_phase(expect_pass=True)
    p2p = p2p_phase()

    return f"""#!/bin/bash
# evaluate.sh - SWE-Forge task evaluator (synthetic FAIL->PASS).
# Emits {{"score": 1}} iff: the broken (mutation) tree FAILS the full hidden suite
# and PASSES regression, and the gold-patched tree PASSES the full hidden suite
# AND regression. Otherwise {{"score": 0}}.
set -o pipefail
export PYTHONDONTWRITEBYTECODE=1

TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_PATH="${{1:-{REPO_DIR}}}"
FORGE_PATH="$TASK_DIR"
SCORE=1

echo "=== SWE-Forge Evaluator ==="
echo "Task dir: $TASK_DIR"
echo "Repo path: $REPO_PATH"

# -- Setup: clone + checkout the base commit (skip when already present) -----
if [ ! -d "$REPO_PATH/.git" ]; then
  echo "Cloning repository..."
  rm -rf "$REPO_PATH"
git clone {shlex.quote(repo_url)} "$REPO_PATH" 2>/dev/null
fi
cd "$REPO_PATH" || {{ echo '{{"score": 0}}'; exit 0; }}
git checkout {shlex.quote(base_commit)} --force 2>/dev/null || true
git clean -fdx 2>/dev/null || true

# -- Apply the mutation/deletion patch (robust: straight apply -> --3way) ----
if [ -s "$FORGE_PATH/deletion_patch.diff" ]; then
  echo "Applying mutation/deletion patch..."
  if ! git apply --whitespace=nowarn "$FORGE_PATH/deletion_patch.diff" 2>/dev/null; then
    if ! git apply --3way --whitespace=nowarn "$FORGE_PATH/deletion_patch.diff" 2>/dev/null; then
      echo "ERROR: could not apply deletion patch"
      echo '{{"score": 0}}'
      exit 0
    fi
  fi
fi

# -- Install dependencies -----------------------------------------------------
echo "=== Installing dependencies ==="
{install_block}

# -- Phase 1: broken tree (hidden suite FAILs, regression PASSes) -------------
echo "=== Phase 1: before gold patch ==="
{f2p_before}
{p2p}
if [ "$SCORE" -eq 0 ]; then
  echo "Phase 1 FAILED - aborting"
  echo '{{"score": 0}}'
  exit 0
fi
echo "Phase 1 PASSED"

# -- Apply the gold patch (robust: straight apply -> --3way) -----------------
echo "=== Applying gold patch ==="
if ! git apply --whitespace=nowarn "$FORGE_PATH/patch.diff" 2>/dev/null; then
  if ! git apply --3way --whitespace=nowarn "$FORGE_PATH/patch.diff" 2>/dev/null; then
    echo "ERROR: could not apply gold patch"
    echo '{{"score": 0}}'
    exit 0
  fi
fi
echo "Gold patch applied"

# -- Phase 2: gold tree (hidden suite PASSes, regression PASSes) -------------
echo "=== Phase 2: after gold patch ==="
{f2p_after}
{p2p}

echo ""
if [ "$SCORE" -eq 1 ]; then
  echo "=== RESULT: PASS ==="
  echo '{{"score": 1}}'
else
  echo "=== RESULT: FAIL ==="
  echo '{{"score": 0}}'
fi
"""


def _write_evaluate(task: ForgeTask, task_dir: Path, runs: Sequence[HiddenRun]) -> None:
    script = _render_evaluate_script(
        repo_url=task.repo_url,
        base_commit=task.base_commit,
        install_commands=task.install_commands,
        pass_to_pass=task.pass_to_pass,
        runs=runs,
        language=task.language,
    )
    path = task_dir / "evaluate.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


# --------------------------------------------------------------------------- #
# Workspace writer
# --------------------------------------------------------------------------- #
def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _workspace_data(task: ForgeTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "repo": {
            "url": task.repo_url,
            "base_commit": task.base_commit,
            "path": REPO_DIR,
        },
        "language": task.language,
        "prompt": task.spec.problem_statement,
        "requirements": list(task.spec.requirements),
        "interface": task.spec.interface_block,
        "environment": {
            "image": task.env_image.image_tag,
            "base_image": task.env_image.base_image,
            "repo_path": REPO_DIR,
            "forge_path": FORGE_DIR,
            "tests_path": TESTS_DIR,
        },
        "install": {
            "commands": list(task.install_commands),
            "working_dir": REPO_DIR,
        },
        "tests": {
            "fail_to_pass": list(task.fail_to_pass),
            "pass_to_pass": list(task.pass_to_pass),
            "working_dir": REPO_DIR,
        },
        # Solution + hidden tests live under the benchmark-only forge_path, never
        # inside the agent repo subtree (repo_path).
        "solution": {
            "path": FORGE_DIR,
            "patch_file": "patch.diff",
            "deletion_patch_file": "deletion_patch.diff",
            "tests_path": TESTS_DIR,
        },
        "synthetic": {
            "source_type": f"synthetic_{task.generator}",
            "deletion_patch_file": "deletion_patch.diff",
            "strategy": task.generator,
            "generator": task.generator,
        },
        "meta": {
            "generator": task.generator,
            "band_verdict": task.calibration_report.band_verdict,
            "oracle_verdict": task.oracle_report.verdict,
            "final_mutation_suite_fingerprint": (
                task.oracle_report.final_mutation_evidence.suite_fingerprint
                if task.oracle_report.final_mutation_evidence
                else ""
            ),
            "created_at": task.created_at,
        },
    }


def write_workspace(
    task: ForgeTask, task_dir: Path, adapter: LanguageAdapter
) -> list[HiddenRun]:
    """Write the full ``tasks/<id>/`` workspace into ``task_dir`` (must exist)."""
    runs = hidden_runs(
        adapter, task.oracle_report.fail_to_pass, task.oracle_report.test_files
    )

    workspace_path = task_dir / "workspace.yaml"
    with workspace_path.open("w", encoding="utf-8") as handle:
        yaml.dump(
            _workspace_data(task),
            handle,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    (task_dir / "patch.diff").write_text(
        _ensure_trailing_newline(task.candidate.oracle_patch), encoding="utf-8"
    )
    (task_dir / "deletion_patch.diff").write_text(
        _ensure_trailing_newline(task.candidate.mutation_patch), encoding="utf-8"
    )

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        target = tests_dir / run.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_ensure_trailing_newline(run.content), encoding="utf-8")
    if not any(tests_dir.rglob("*")):
        # A shipped task always has a non-empty hidden suite; guard the contract.
        raise ExportError(
            f"refusing to export {task.task_id!r}: no hidden test files to ship"
        )

    (task_dir / "provenance.json").write_text(
        json.dumps(task.provenance.to_dict(), indent=2), encoding="utf-8"
    )

    _write_evaluate(task, task_dir, runs)
    return runs


# --------------------------------------------------------------------------- #
# Orphan-git re-init + leak audit (incl. the .git history vector)
# --------------------------------------------------------------------------- #
def reinit_orphan_git(repo_dir: Path | str) -> None:
    """Re-init ``repo_dir`` as a single ORPHAN/root commit (strip gold history).

    forge builds the broken tree as a synthetic commit ON TOP of the pinned GOLD
    commit, and an EnvImage keeps the repo's full ``.git`` history -- so gold is
    one ``git show HEAD~1`` / ``git checkout HEAD~1`` away. Any agent-facing tree
    we ship must therefore drop that history: we delete ``.git`` and re-init a
    fresh repo whose ONLY commit is the broken baseline, leaving exactly one
    ``HEAD`` (so ``git diff HEAD`` patch capture still works) while ``git show
    HEAD~1`` fails and gold is unrecoverable from history.
    """
    repo = Path(repo_dir)
    git = shutil.which("git")
    if git is None:
        raise ExportError("git is required to strip .git history but was not found")
    shutil.rmtree(repo / ".git", ignore_errors=True)
    env = {**os.environ, **_GIT_ENV}
    subprocess.run([git, "init", "-q"], cwd=repo, check=True, env=env)
    subprocess.run([git, "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(
        [git, "commit", "-q", "-m", "broken-baseline", "--allow-empty"],
        cwd=repo,
        check=True,
        env=env,
    )


def audit_git_history(root: Path | str) -> list[str]:
    """Report any ``.git`` under ``root`` from which gold is recoverable.

    Gold is recoverable when a shipped repo's history has more than one commit
    (``git show HEAD~1`` succeeds): forge's broken tree sits on top of the gold
    commit, so a non-orphan history leaks the answer. A single orphan/root commit
    (or no ``.git`` at all) is clean.
    """
    root_path = Path(root).resolve()
    git = shutil.which("git")
    findings: list[str] = []
    for git_dir in root_path.rglob(".git"):
        if not git_dir.is_dir():
            continue
        repo = git_dir.parent
        rel = repo.relative_to(root_path)
        if git is None:
            findings.append(f"git_history: unverifiable .git present at {rel}")
            continue
        result = subprocess.run(
            [git, "-C", str(repo), "rev-list", "--count", "HEAD"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            continue  # no HEAD / not a real repo -> nothing to recover
        try:
            commits = int(result.stdout.strip())
        except ValueError:
            continue
        if commits > 1:
            findings.append(
                f"git_history: gold recoverable via .git history at {rel} "
                f"({commits} commits; 'git show HEAD~1' succeeds)"
            )
    return findings


def _copy_agent_visible(src: Path, dst: Path) -> None:
    """Copy the AGENT-VISIBLE subset of an exported task dir to ``dst``.

    Excludes the benchmark-only solution/harness files (gold + mutation patches,
    the hidden ``tests/`` suite, the scorer scripts) so the leak audit scans only
    what the solver would actually see (statement/requirements/interface +, when
    shipped, the broken repo tree).
    """
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_dir():
            if path.name in _BENCHMARK_ONLY_DIRS:
                continue
            shutil.copytree(path, dst / path.name)
        else:
            if path.name in _BENCHMARK_ONLY_NAMES:
                continue
            shutil.copy2(path, dst / path.name)


def audit_exported_workspace(
    task_dir: Path | str,
    *,
    oracle_patch: str,
    test_files: Sequence[OracleTestFile],
) -> LeakAuditResult:
    """Leak-audit an exported ``tasks/<id>/`` tree (file scan + .git history).

    Combines two vectors: (1) a static content scan over the agent-visible files
    (no gold/oracle patch line, no hidden-test body, no forbidden/cache artifact),
    reusing the oracle gate's :func:`audit_agent_tree`; and (2) a ``.git`` history
    check that gold is not recoverable from any shipped repo's commit history.
    Clean -> ``passed == True``, ``findings == []``, ``risk_score == 0.0``.
    """
    task_path = Path(task_dir).resolve()
    findings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="swe-forge-export-audit-") as tmp:
        view = Path(tmp) / "agent_view"
        _copy_agent_visible(task_path, view)
        audit = audit_agent_tree(
            view, oracle_patch=oracle_patch, hidden_test_files=test_files
        )
        findings.extend(audit.markers())
    findings.extend(audit_git_history(task_path))
    risk = min(1.0, len(findings) / 10.0)
    return LeakAuditResult(passed=not findings, risk_score=risk, findings=findings)


# --------------------------------------------------------------------------- #
# Single-task export (private stage writer + recoverable direct publication)
# --------------------------------------------------------------------------- #
@dataclass
class TaskExportResult:
    """Outcome of exporting one task's workspace."""

    task_id: str
    status: str  # "shipped" | "skipped" | "refused" | "source_rejected" | "cap_rejected" | "failed"
    path: Path | None = None
    reason: str = ""
    leak_findings: list[str] = field(default_factory=list)

    @property
    def shipped(self) -> bool:
        return self.status == "shipped"

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "path": str(self.path) if self.path else None,
            "reason": self.reason,
            "leak_findings": list(self.leak_findings),
        }


def _validate_staged_workspace(task: ForgeTask, task_dir: Path) -> None:
    """Reject an incomplete staged workspace before any public selection."""
    required = (
        task_dir / "workspace.yaml",
        task_dir / "patch.diff",
        task_dir / "deletion_patch.diff",
        task_dir / "evaluate.sh",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ExportError(
            f"staged workspace is missing required files: {', '.join(sorted(missing))}"
        )
    if not os.access(task_dir / "evaluate.sh", os.X_OK):
        raise ExportError("staged workspace evaluate.sh is not executable")

    try:
        workspace = yaml.safe_load((task_dir / "workspace.yaml").read_text("utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ExportError(f"staged workspace.yaml is unreadable: {exc}") from exc
    if not isinstance(workspace, dict) or workspace.get("task_id") != task.task_id:
        raise ExportError("staged workspace.yaml does not match its task id")

    tests_dir = task_dir / "tests"
    if not tests_dir.is_dir() or not any(
        path.is_file() for path in tests_dir.rglob("*")
    ):
        raise ExportError("staged workspace has no hidden test files")


def _write_staged_workspace(
    task: ForgeTask,
    task_dir: Path,
    *,
    adapter: LanguageAdapter | None = None,
    broken_tree: Path | str | None = None,
) -> TaskExportResult:
    """Write and audit one workspace only inside a caller-owned private stage.

    This intentionally has no public overwrite or publication behavior.  Batch
    export and :class:`PilotCheckpoint` call it beneath their already-private
    generation stage, while direct export selects the completed stage through its
    own task-scoped immutable generation pointer.
    """
    if os.path.lexists(task_dir):
        if not task_dir.is_dir() or task_dir.is_symlink() or any(task_dir.iterdir()):
            return TaskExportResult(
                task_id=task.task_id,
                status="failed",
                reason=f"staged workspace path is not an empty directory: {task_dir}",
            )
    else:
        task_dir.mkdir()

    if adapter is None:
        adapter = build_default_registry().get(task.language)

    try:
        write_workspace(task, task_dir, adapter)
        if broken_tree is not None:
            repo_dst = task_dir / "repo"
            shutil.copytree(broken_tree, repo_dst)
            reinit_orphan_git(repo_dst)
        _validate_staged_workspace(task, task_dir)

        audit = audit_exported_workspace(
            task_dir,
            oracle_patch=task.candidate.oracle_patch,
            test_files=task.oracle_report.test_files,
        )
        if not audit.passed:
            shutil.rmtree(task_dir, ignore_errors=True)
            return TaskExportResult(
                task_id=task.task_id,
                status="refused",
                reason="leak audit failed: " + "; ".join(audit.findings),
                leak_findings=list(audit.findings),
            )
        return TaskExportResult(task_id=task.task_id, status="shipped", path=task_dir)
    except Exception as exc:  # noqa: BLE001 - cleanup then report, never partial dir
        shutil.rmtree(task_dir, ignore_errors=True)
        return TaskExportResult(
            task_id=task.task_id,
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
        )


def _direct_store_root(tasks_root: Path, task_id: str) -> Path:
    return (
        tasks_root.parent
        / _DIRECT_STORE_DIR
        / _direct_store_namespace(tasks_root)
        / task_id
    )


def _validate_task_id(task_id: str) -> None:
    """Require every export ID to be one safe path component."""
    if (
        task_id in ("", ".", "..")
        or "/" in task_id
        or "\\" in task_id
        or Path(task_id).is_absolute()
    ):
        raise ExportError(f"unsafe export task id: {task_id!r}")


def _direct_store_ancestors(tasks_root: Path) -> tuple[Path, Path]:
    """Return the store base and root-specific namespace beneath its parent."""
    base = tasks_root.parent / _DIRECT_STORE_DIR
    return base, base / _direct_store_namespace(tasks_root)


def _direct_store_namespace(tasks_root: Path) -> str:
    """Return a stable private-store namespace for this exact task root."""
    root = tasks_root.resolve(strict=False)
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]


def _direct_facade_target(tasks_root: Path, task_id: str) -> str:
    namespace = _direct_store_namespace(tasks_root)
    return f"../{_DIRECT_STORE_DIR}/{namespace}/{task_id}/{_DIRECT_CURRENT_LINK}"


def _read_selected_direct_workspace(
    tasks_root: Path, task: ForgeTask, final_dir: Path
) -> Path | None:
    """Return the complete direct generation selected by ``current`` if present.

    Only the expected facade symlink is managed.  Legacy files, directories,
    arbitrary symlinks, and broken selected pointers fail closed rather than
    being replaced or migrated destructively.
    """
    task_id = task.task_id
    expected_facade = _direct_facade_target(tasks_root, task_id)
    final_exists = os.path.lexists(final_dir)
    if final_exists:
        if not final_dir.is_symlink() or os.readlink(final_dir) != expected_facade:
            raise ExportError(
                f"refusing to replace unmanaged legacy workspace {final_dir}"
            )

    store_base, namespace = _direct_store_ancestors(tasks_root)
    for ancestor in (store_base, namespace):
        if os.path.lexists(ancestor) and (
            not ancestor.is_dir() or ancestor.is_symlink()
        ):
            raise ExportError(f"direct export store ancestor is invalid: {ancestor}")

    store = _direct_store_root(tasks_root, task_id)
    if os.path.lexists(store) and (not store.is_dir() or store.is_symlink()):
        raise ExportError(f"refusing to use unmanaged direct export store {store}")
    current = store / _DIRECT_CURRENT_LINK
    if not current.is_symlink():
        if os.path.lexists(current):
            raise ExportError(f"direct export pointer is not a symlink: {current}")
        if final_exists:
            raise ExportError(
                f"managed direct facade has no selected generation: {final_dir}"
            )
        return None

    target = os.readlink(current)
    generation_name = target.removeprefix(f"{_DIRECT_GENERATIONS_DIR}/")
    if target != f"{_DIRECT_GENERATIONS_DIR}/{generation_name}" or not re.fullmatch(
        r"[0-9a-f]{32}", generation_name
    ):
        raise ExportError(f"direct export pointer has an invalid target: {current}")

    generations = store / _DIRECT_GENERATIONS_DIR
    if not generations.is_dir() or generations.is_symlink():
        raise ExportError(f"direct export generations store is invalid: {generations}")
    raw_workspace = generations / generation_name
    if not raw_workspace.is_dir() or raw_workspace.is_symlink():
        raise ExportError(
            f"direct export pointer does not select an immutable generation: {current}"
        )
    try:
        workspace = current.resolve(strict=True)
        expected_workspace = raw_workspace.resolve(strict=True)
    except OSError as exc:
        raise ExportError(f"direct export pointer is broken: {current}") from exc
    if workspace != expected_workspace:
        raise ExportError(
            f"direct export pointer escapes its generation store: {current}"
        )
    if not (workspace / "workspace.yaml").is_file():
        raise ExportError(
            f"direct export pointer does not select a complete workspace: {current}"
        )

    try:
        workspace_data = yaml.safe_load(
            (workspace / "workspace.yaml").read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as exc:
        raise ExportError(
            f"selected direct workspace.yaml is unreadable: {exc}"
        ) from exc
    if not isinstance(workspace_data, dict) or workspace_data.get("task_id") != task_id:
        raise ExportError(f"direct export pointer selects the wrong task: {current}")
    _validate_staged_workspace(task, workspace)

    if final_exists and final_dir.resolve(strict=True) != workspace:
        raise ExportError(
            f"direct export facade resolves outside its selected workspace: {final_dir}"
        )
    return workspace


def _direct_pointer_selects(store: Path, generation_id: str) -> bool:
    """Whether ``current`` already names exactly this committed generation."""
    current = store / _DIRECT_CURRENT_LINK
    return (
        current.is_symlink()
        and os.readlink(current) == f"{_DIRECT_GENERATIONS_DIR}/{generation_id}"
    )


def _install_direct_facade(tasks_root: Path, task_id: str, final_dir: Path) -> None:
    """Create the task's stable facade after a complete generation is selected."""
    target = _direct_facade_target(tasks_root, task_id)
    if os.path.lexists(final_dir):
        if final_dir.is_symlink() and os.readlink(final_dir) == target:
            return
        raise ExportError(f"refusing to replace unmanaged legacy workspace {final_dir}")

    temporary = tasks_root / f".{task_id}.facade-{uuid.uuid4().hex}"
    try:
        os.symlink(target, temporary)
        os.replace(temporary, final_dir)
        publication._fsync_path(tasks_root)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_new_empty_direct_store(store: Path) -> None:
    """Discard only a brand-new store when staging failed before any commit."""
    generations = store / _DIRECT_GENERATIONS_DIR
    current = store / _DIRECT_CURRENT_LINK
    if current.is_symlink() or any(
        generations.iterdir() if generations.exists() else ()
    ):
        return
    shutil.rmtree(store, ignore_errors=True)
    parent = store.parent
    try:
        parent.rmdir()
    except OSError:
        pass


def _fsync_direct_store_ancestors(tasks_root: Path, store: Path) -> None:
    """Persist newly-created store ancestry before selecting a generation."""
    for path in (store.parent, store.parent.parent, tasks_root.parent):
        publication._fsync_path(path)


def export_forge_task(
    task: ForgeTask,
    tasks_root: Path | str,
    *,
    overwrite: bool = False,
    adapter: LanguageAdapter | None = None,
    broken_tree: Path | str | None = None,
) -> TaskExportResult:
    """Publish one direct workspace through a recoverable immutable generation.

    The visible ``tasks/<id>`` is a managed symlink to a task-scoped ``current``
    pointer.  A successor is fully written, audited, fsynced, and renamed into
    the immutable generation store before the pointer replacement selects it.
    Thus overwrite failures preserve the prior workspace byte-for-byte, while
    abandoned staging and unselected generations remain invisible on restart.
    """
    try:
        ensure_oracle_exportable(
            task.oracle_report,
            candidate=task.candidate,
            calibration_kept=task.calibration_report.is_keep,
        )
    except ExportRefusedError as exc:
        return TaskExportResult(
            task_id=task.task_id,
            status="refused",
            reason=str(exc),
        )

    tasks_path = Path(tasks_root)
    if tasks_path.is_symlink():
        return TaskExportResult(
            task_id=task.task_id,
            status="failed",
            reason="refusing direct export through a managed batch tasks facade",
        )
    try:
        _validate_task_id(task.task_id)
    except ExportError as exc:
        return TaskExportResult(
            task_id=task.task_id,
            status="failed",
            reason=str(exc),
        )

    final_dir = tasks_path / task.task_id
    store = _direct_store_root(tasks_path, task.task_id)
    final_exists = os.path.lexists(final_dir)
    expected_facade = _direct_facade_target(tasks_path, task.task_id)
    if (
        final_exists
        and not overwrite
        and (not final_dir.is_symlink() or os.readlink(final_dir) != expected_facade)
    ):
        return TaskExportResult(
            task_id=task.task_id,
            status="skipped",
            path=final_dir,
            reason="task workspace already exists; overwrite not requested",
        )

    stage: Path | None = None
    pointer_tmp: Path | None = None
    store_created = False
    generation_committed = False

    try:
        selected = _read_selected_direct_workspace(tasks_path, task, final_dir)
        if final_exists and not overwrite:
            return TaskExportResult(
                task_id=task.task_id,
                status="skipped",
                path=final_dir,
                reason="task workspace already exists; overwrite not requested",
            )
        if selected is not None and not os.path.lexists(final_dir):
            _install_direct_facade(tasks_path, task.task_id, final_dir)
        if selected is None and os.path.lexists(final_dir):
            raise ExportError(
                f"unmanaged direct workspace cannot be published: {final_dir}"
            )

        tasks_root_existed = tasks_path.exists()
        tasks_path.mkdir(parents=True, exist_ok=True)
        if not tasks_root_existed:
            publication._fsync_path(tasks_path.parent)
        store_base, namespace = _direct_store_ancestors(tasks_path)
        if not store_base.exists():
            store_base.mkdir()
            publication._fsync_path(store_base.parent)
        if not namespace.exists():
            namespace.mkdir()
            publication._fsync_path(store_base)
        if not store.exists():
            store.mkdir()
            store_created = True
            _fsync_direct_store_ancestors(tasks_path, store)
        generations = store / _DIRECT_GENERATIONS_DIR
        if os.path.lexists(generations):
            if not generations.is_dir() or generations.is_symlink():
                raise ExportError(
                    f"direct export generations store is invalid: {generations}"
                )
        else:
            generations.mkdir()
            publication._fsync_path(store)
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=store))
        staged = _write_staged_workspace(
            task,
            stage,
            adapter=adapter,
            broken_tree=broken_tree,
        )
        if not staged.shipped:
            shutil.rmtree(stage, ignore_errors=True)
            stage = None
            if store_created:
                _remove_new_empty_direct_store(store)
            return staged

        publication._fsync_tree(stage)
        generation_id = uuid.uuid4().hex
        final_generation = generations / generation_id
        os.replace(stage, final_generation)
        stage = None
        generation_committed = True
        publication._fsync_path(generations)

        pointer_tmp = store / f".current-{uuid.uuid4().hex}"
        os.symlink(f"{_DIRECT_GENERATIONS_DIR}/{generation_id}", pointer_tmp)
        publication._fsync_path(store)
        try:
            os.replace(pointer_tmp, store / _DIRECT_CURRENT_LINK)
        except OSError:
            # The pointer replacement is the irreversible commit point.  Some
            # filesystems/test doubles may report an error after completing the
            # rename, in which case expose the selected complete successor
            # rather than falsely claiming that the old generation survived.
            if not _direct_pointer_selects(store, generation_id):
                raise
        pointer_tmp = None
        publication._fsync_path(store)
        if selected is None:
            # First write has no predecessor.  Overwrites retain their stable
            # facade throughout, so no failure-prone operation follows commit.
            _install_direct_facade(tasks_path, task.task_id, final_dir)
        return TaskExportResult(task_id=task.task_id, status="shipped", path=final_dir)
    except Exception as exc:  # noqa: BLE001 - preserve selected output on every failure
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        if pointer_tmp is not None:
            pointer_tmp.unlink(missing_ok=True)
        if store_created and not generation_committed:
            _remove_new_empty_direct_store(store)
        return TaskExportResult(
            task_id=task.task_id,
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
        )


# --------------------------------------------------------------------------- #
# Batch export (mixed batch ships only the qualified subset)
# --------------------------------------------------------------------------- #
@dataclass
class ExportRequest:
    """One candidate's artifacts to attempt to export (assembly + writing)."""

    candidate: Candidate
    spec: GeneratedSpec
    oracle_report: OracleReport
    calibration_report: CalibrationReport
    env_image: EnvImage
    repo_url: str
    base_commit: str = ""
    repo: str | None = None
    task_id: str | None = None
    broken_tree: Path | str | None = None

    def _fallback_id(self) -> str:
        repo = self.repo or _repo_slug(self.repo_url, self.env_image.repo_id)
        return forge_task_id(
            repo,
            self.candidate.generator,
            self.candidate.provenance.seed,
            self.candidate.target.files,
            self.candidate.target.symbols,
        )


@dataclass
class BatchExportResult:
    """Aggregate outcome of a batch export."""

    out_dir: Path
    tasks_dir: Path
    jsonl_path: Path
    parquet_path: Path
    results: list[TaskExportResult] = field(default_factory=list)

    @property
    def shipped(self) -> list[TaskExportResult]:
        return [r for r in self.results if r.status == "shipped"]

    @property
    def kept(self) -> list[TaskExportResult]:
        return [r for r in self.results if r.status in ("shipped", "skipped")]

    @property
    def refused(self) -> list[TaskExportResult]:
        return [
            r
            for r in self.results
            if r.status in ("refused", "source_rejected", "cap_rejected", "failed")
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "out_dir": str(self.out_dir),
            "tasks_dir": str(self.tasks_dir),
            "jsonl_path": str(self.jsonl_path),
            "parquet_path": str(self.parquet_path),
            "shipped_count": len(self.shipped),
            "kept_count": len(self.kept),
            "refused_count": len(self.refused),
            "results": [r.to_dict() for r in self.results],
        }


def export_batch(
    requests: Sequence[ExportRequest],
    out_dir: Path | str,
    *,
    overwrite: bool = False,
    replace_existing: bool = False,
    jsonl_name: str = "dataset.jsonl",
    parquet_name: str = "dataset.parquet",
    adapter: LanguageAdapter | None = None,
) -> BatchExportResult:
    """Export a batch: ship only the qualified subset; one refusal never aborts.

    Each request is assembled through the fail-fast gate; an oracle-reject or
    calibration-drop is recorded as refused (with the violated gate) and skipped
    without aborting its qualified siblings. Qualified tasks are written and the
    jsonl/parquet datasets are regenerated from the FULL kept set (one record per
    kept task, no append-duplication). An all-unqualified (or empty) batch still
    writes valid empty artifacts. ``replace_existing`` is only for a caller that
    has re-certified a same-id task and needs a complete successor generation;
    ordinary exports retain their conflicting-id refusal.
    """
    out_path = Path(out_dir)
    tasks_dir = out_path / "tasks"
    jsonl_path = out_path / jsonl_name
    parquet_path = out_path / parquet_name

    results_by_index: dict[int, TaskExportResult] = {}
    unique_entries: list[PublicationEntry] = []
    duplicate_of: dict[int, int] = {}
    canonical_by_id: dict[str, str] = {}
    entry_index_by_id: dict[str, int] = {}

    # Assemble and deduplicate the whole request set BEFORE opening a generation
    # for write.  A same-id conflict is therefore a no-mutation error.
    for index, request in enumerate(requests):
        try:
            task = assemble_forge_task(
                candidate=request.candidate,
                spec=request.spec,
                oracle_report=request.oracle_report,
                calibration_report=request.calibration_report,
                env_image=request.env_image,
                repo_url=request.repo_url,
                base_commit=request.base_commit,
                repo=request.repo,
                task_id=request.task_id,
                adapter=adapter,
            )
        except (ExportRefusedError, ExportGateError, ExportError) as exc:
            results_by_index[index] = TaskExportResult(
                task_id=request.task_id or request._fallback_id(),
                status="refused",
                reason=str(exc),
            )
            continue

        payload = canonical_task_payload(task)
        previous_payload = canonical_by_id.get(task.task_id)
        if previous_payload is not None:
            if previous_payload != payload:
                raise ExportError(
                    f"conflicting duplicate task_id {task.task_id!r}; "
                    "all same-id payloads must be identical"
                )
            duplicate_of[index] = entry_index_by_id[task.task_id]
            continue
        canonical_by_id[task.task_id] = payload
        entry_index_by_id[task.task_id] = index
        unique_entries.append(PublicationEntry(index=index, task=task))

    def _write_workspace(task: ForgeTask, stage_tasks: Path) -> TaskExportResult:
        request = next(
            item for item in unique_entries if item.task.task_id == task.task_id
        )
        source_request = requests[request.index]
        return _write_staged_workspace(
            task,
            stage_tasks / task.task_id,
            adapter=adapter,
            broken_tree=source_request.broken_tree,
        )

    try:
        _generation, outcomes = publish_generation(
            out_path,
            unique_entries,
            workspace_writer=_write_workspace,
            dataset_writer=export_dataset,
            overwrite=overwrite,
            replace_existing=replace_existing,
        )
    except PublicationError as exc:
        raise ExportError(str(exc)) from exc

    for outcome in outcomes:
        results_by_index[outcome.index] = TaskExportResult(
            task_id=outcome.task_id,
            status=outcome.status,
            path=tasks_dir / outcome.task_id if outcome.kept else None,
            reason=outcome.reason,
            leak_findings=list(outcome.leak_findings),
        )
    for duplicate_index, original_index in duplicate_of.items():
        original = results_by_index.get(original_index)
        if original is None:
            raise ExportError("duplicate task export lost its original result")
        results_by_index[duplicate_index] = TaskExportResult(
            task_id=original.task_id,
            status="deduplicated",
            path=original.path,
            reason=f"identical to request at index {original_index}",
        )

    results = [results_by_index[index] for index in range(len(requests))]

    return BatchExportResult(
        out_dir=out_path,
        tasks_dir=tasks_dir,
        jsonl_path=jsonl_path,
        parquet_path=parquet_path,
        results=results,
    )
