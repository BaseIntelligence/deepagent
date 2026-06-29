"""Headline A verification path: run each shipped task's ``evaluate.sh`` in Docker.

For every exported ``tasks/<id>/`` this module runs the self-contained
``evaluate.sh`` inside the task's Docker image and confirms the synthetic
FAIL->PASS contract end to end:

* **Per task (VAL-EXPORT-009):** ``evaluate.sh`` Phase 1 proves the broken
  (mutation) tree FAILS the hidden suite while the regression (P2P) stays green,
  Phase 2 proves the gold-patched tree PASSES the hidden suite AND regression,
  and the script emits a final ``{"score": 1}`` (exit 0). A final score of 1 is
  therefore the complete Headline A proof for that task (Phase 1 aborts with
  ``{"score": 0}`` if the broken state does not fail or regression breaks).
* **Aggregate (VAL-EXPORT-010):** gold == 100% across the whole shipped set --
  ``count(score == 1) == count(tasks/*/)``; a single non-1 gold score is a
  release blocker.
* **Determinism (VAL-EXPORT-011):** >=2 independent ``--rm`` runs of the same
  task reproduce ``{"score": 1}`` (the score never flips across fresh
  containers).

The Docker invocation is injectable (``runner=``) so the aggregation and
determinism logic is unit-tested offline; the real ``docker run --rm`` path is
exercised by the integration test and the manual proof. Containers use unique
``--rm`` names and are force-removed on timeout (guaranteed teardown, no orphan
containers, no off-limits resources touched).
"""

from __future__ import annotations

import re
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

import yaml  # type: ignore[import-untyped]

from swe_forge.forge.export import FORGE_DIR, REPO_DIR

#: Independent fresh-container runs per task. >=2 proves determinism (no flip).
DEFAULT_DETERMINISM_RUNS = 2
#: Per-run wall-clock ceiling (clone + install + two phases of the hidden suite).
DEFAULT_TIMEOUT = 1800.0

_SCORE_RE = re.compile(r'\{"score":\s*([01])\}')
_PHASE1_MARKER = "Phase 1 PASSED"
_NAME_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_.-]")


class GoldEvalError(RuntimeError):
    """Raised when a task workspace cannot be evaluated (missing image/script)."""


class DockerExec(NamedTuple):
    """The captured result of one ``docker run`` of an ``evaluate.sh``."""

    exit_code: int
    stdout: str
    stderr: str


#: A pluggable Docker runner: ``(task_dir, *, image, name, timeout, extra_args)``.
DockerRunner = Callable[..., DockerExec]


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
def parse_score(output: str) -> int | None:
    """Return the LAST ``{"score": N}`` value in ``output`` (``None`` if absent).

    ``evaluate.sh`` may print an intermediate ``{"score": 0}`` on an early abort,
    so the final score is the last match.
    """
    matches = _SCORE_RE.findall(output or "")
    if not matches:
        return None
    return int(matches[-1])


def phase1_passed(output: str) -> bool:
    """True iff ``evaluate.sh`` reported ``Phase 1 PASSED`` (broken-fail + P2P)."""
    return _PHASE1_MARKER in (output or "")


# --------------------------------------------------------------------------- #
# Image resolution + container naming
# --------------------------------------------------------------------------- #
def resolve_eval_image(task_dir: Path | str) -> str:
    """Resolve the Docker image for a task from its ``workspace.yaml``.

    Prefers the built ``EnvImage`` tag (``environment.image``) and falls back to
    the language ``base_image``. Raises :class:`GoldEvalError` when neither is set.
    """
    task_dir = Path(task_dir)
    workspace = task_dir / "workspace.yaml"
    if not workspace.is_file():
        raise GoldEvalError(f"no workspace.yaml in {task_dir}")
    data = yaml.safe_load(workspace.read_text(encoding="utf-8")) or {}
    env = data.get("environment", {}) if isinstance(data, dict) else {}
    image = ""
    if isinstance(env, dict):
        image = str(env.get("image") or env.get("base_image") or "").strip()
    if not image:
        raise GoldEvalError(
            f"workspace.yaml in {task_dir} declares no environment image"
        )
    return image


def _container_name(prefix: str, task_id: str) -> str:
    slug = _NAME_SANITIZE_RE.sub("-", task_id)[:32].strip("-_.") or "task"
    return f"{prefix}-{slug}-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# Default Docker runner (throwaway --rm container, forced teardown on timeout)
# --------------------------------------------------------------------------- #
def run_evaluate_container(
    task_dir: Path,
    *,
    image: str,
    name: str,
    timeout: float = DEFAULT_TIMEOUT,
    extra_args: Sequence[str] = (),
) -> DockerExec:
    """Run a task's ``evaluate.sh`` in a throwaway ``--rm`` container.

    The task dir is mounted read-only at the in-container forge path and the
    self-contained script is invoked with the standard repo path. ``--rm``
    guarantees teardown on normal exit; a timeout force-removes the container by
    its unique name so no orphan is left on the shared host.
    """
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "-v",
        f"{Path(task_dir).resolve()}:{FORGE_DIR}:ro",
        *extra_args,
        image,
        "bash",
        f"{FORGE_DIR}/evaluate.sh",
        REPO_DIR,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return DockerExec(proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as exc:
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
        return DockerExec(
            124,
            stdout,
            f"timeout after {timeout}s; container '{name}' force-removed\n{stderr}",
        )
    except FileNotFoundError as exc:  # docker binary missing
        raise GoldEvalError(f"docker not available: {exc}") from exc


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


# --------------------------------------------------------------------------- #
# Result records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvalRun:
    """One ``evaluate.sh`` run of a task in a fresh container."""

    task_id: str
    run_index: int
    score: int | None
    phase1_passed: bool
    exit_code: int
    container_name: str
    stdout: str = ""
    stderr: str = ""

    @property
    def gold(self) -> bool:
        return self.score == 1

    def to_dict(self) -> dict[str, object]:
        return {
            "run_index": self.run_index,
            "score": self.score,
            "phase1_passed": self.phase1_passed,
            "exit_code": self.exit_code,
            "container_name": self.container_name,
        }


@dataclass(frozen=True)
class TaskGoldResult:
    """The aggregate of N fresh-container runs of one task's ``evaluate.sh``."""

    task_id: str
    task_dir: Path
    image: str
    runs: list[EvalRun]

    @property
    def scores(self) -> list[int | None]:
        return [r.score for r in self.runs]

    @property
    def gold(self) -> bool:
        """Every run scored exactly ``{"score": 1}`` (Headline A for this task)."""
        return bool(self.runs) and all(r.score == 1 for r in self.runs)

    @property
    def deterministic(self) -> bool:
        """No flip across runs: every fresh container produced the same score."""
        return len({r.score for r in self.runs}) <= 1

    @property
    def phase1_all(self) -> bool:
        """Every run reported ``Phase 1 PASSED`` (broken-fail + regression-green)."""
        return bool(self.runs) and all(r.phase1_passed for r in self.runs)

    @property
    def final_score(self) -> int | None:
        return self.runs[-1].score if self.runs else None

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "image": self.image,
            "gold": self.gold,
            "deterministic": self.deterministic,
            "phase1_passed": self.phase1_all,
            "scores": self.scores,
            "runs": [r.to_dict() for r in self.runs],
        }


@dataclass(frozen=True)
class GoldEvalReport:
    """The Headline A roll-up over every shipped ``tasks/<id>/``."""

    tasks_dir: Path
    results: list[TaskGoldResult]

    @property
    def shipped_count(self) -> int:
        return len(self.results)

    @property
    def gold_count(self) -> int:
        return sum(1 for r in self.results if r.gold)

    @property
    def gold_rate(self) -> float:
        return self.gold_count / self.shipped_count if self.results else 0.0

    @property
    def all_gold(self) -> bool:
        """count(gold) == count(shipped tasks), i.e. gold == 100% (VAL-EXPORT-010)."""
        return bool(self.results) and self.gold_count == self.shipped_count

    @property
    def deterministic(self) -> bool:
        """Every shipped task reproduced its score across runs (VAL-EXPORT-011)."""
        return all(r.deterministic for r in self.results)

    @property
    def non_gold(self) -> list[TaskGoldResult]:
        return [r for r in self.results if not r.gold]

    @property
    def flipped(self) -> list[TaskGoldResult]:
        return [r for r in self.results if not r.deterministic]

    @property
    def passed(self) -> bool:
        """Headline A holds: gold == 100% AND no task flipped across runs."""
        return self.all_gold and self.deterministic

    def to_dict(self) -> dict[str, object]:
        return {
            "tasks_dir": str(self.tasks_dir),
            "shipped_count": self.shipped_count,
            "gold_count": self.gold_count,
            "gold_rate": self.gold_rate,
            "all_gold": self.all_gold,
            "deterministic": self.deterministic,
            "passed": self.passed,
            "non_gold": [r.task_id for r in self.non_gold],
            "flipped": [r.task_id for r in self.flipped],
            "results": [r.to_dict() for r in self.results],
        }


# --------------------------------------------------------------------------- #
# Task discovery + evaluation
# --------------------------------------------------------------------------- #
def _has_task_dirs(root: Path) -> bool:
    return any(p.is_dir() and (p / "evaluate.sh").is_file() for p in root.iterdir())


def resolve_tasks_root(path: Path | str) -> Path:
    """Resolve the directory that directly contains the ``<id>/`` task dirs.

    Accepts either that directory or an export ``out_dir`` whose ``tasks/``
    subdirectory holds the workspaces.
    """
    path = Path(path)
    if not path.is_dir():
        raise GoldEvalError(f"not a directory: {path}")
    sub = path / "tasks"
    if sub.is_dir() and not _has_task_dirs(path):
        return sub
    return path


def discover_task_dirs(tasks_root: Path | str) -> list[Path]:
    """All immediate ``<id>/`` dirs under ``tasks_root`` that hold an evaluate.sh."""
    tasks_root = Path(tasks_root)
    return sorted(
        p for p in tasks_root.iterdir() if p.is_dir() and (p / "evaluate.sh").is_file()
    )


def evaluate_task_gold(
    task_dir: Path | str,
    *,
    runs: int = DEFAULT_DETERMINISM_RUNS,
    image: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    extra_args: Sequence[str] = (),
    runner: DockerRunner | None = None,
    name_prefix: str = "swe-forge-goldeval",
) -> TaskGoldResult:
    """Run one task's ``evaluate.sh`` ``runs`` times in fresh ``--rm`` containers."""
    task_dir = Path(task_dir)
    if not (task_dir / "evaluate.sh").is_file():
        raise GoldEvalError(f"no evaluate.sh in {task_dir}")
    image = image or resolve_eval_image(task_dir)
    run_fn = runner or run_evaluate_container
    eval_runs: list[EvalRun] = []
    for i in range(max(1, runs)):
        name = _container_name(name_prefix, task_dir.name)
        result = run_fn(
            task_dir,
            image=image,
            name=name,
            timeout=timeout,
            extra_args=tuple(extra_args),
        )
        eval_runs.append(
            EvalRun(
                task_id=task_dir.name,
                run_index=i,
                score=parse_score(result.stdout),
                phase1_passed=phase1_passed(result.stdout),
                exit_code=result.exit_code,
                container_name=name,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
    return TaskGoldResult(
        task_id=task_dir.name, task_dir=task_dir, image=image, runs=eval_runs
    )


def run_gold_eval(
    tasks_dir: Path | str,
    *,
    runs: int = DEFAULT_DETERMINISM_RUNS,
    image: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    extra_args: Sequence[str] = (),
    runner: DockerRunner | None = None,
    name_prefix: str = "swe-forge-goldeval",
) -> GoldEvalReport:
    """Run ``evaluate.sh`` over every shipped task and roll up the Headline A proof.

    Each task is evaluated ``runs`` times (>=2 proves determinism); a single
    ``image`` override applies to every task (otherwise each task's own
    ``workspace.yaml`` image is used, so a multi-language set scores correctly).
    """
    tasks_root = resolve_tasks_root(tasks_dir)
    task_dirs = discover_task_dirs(tasks_root)
    results = [
        evaluate_task_gold(
            d,
            runs=runs,
            image=image,
            timeout=timeout,
            extra_args=extra_args,
            runner=runner,
            name_prefix=name_prefix,
        )
        for d in task_dirs
    ]
    return GoldEvalReport(tasks_dir=tasks_root, results=results)
