"""Stage 5 benchmark report: provenance audit + the two-headline roll-up.

This module turns an exported pilot directory (``tasks/<id>/`` workspaces plus the
``dataset.jsonl`` / ``dataset.parquet`` datasets) into two artifacts:

* a **provenance audit** -- per-task *completeness* (every shipped task records the
  mandated provenance fields) and *consistency with the gates* (each task's
  recorded oracle/band verdicts and calibration numbers agree with why it was
  kept), and
* a **benchmark report** (Markdown + machine-parseable JSON) carrying the two
  mission headlines and the supporting panel/IRT/breakdown evidence.

The assertions this serves (the feature's ``fulfills`` set):

* **VAL-EXPORT-014** -- provenance complete for 100% of shipped tasks
  (generator, seed, mutation-adequacy counts, IRT params, the per-``{model,tier}``
  panel solve matrix covering weak->mid->frontier, language, tool versions, and a
  parseable timestamp).
* **VAL-EXPORT-015** -- provenance internally consistent with the gates: every
  shipped task records ``oracle == pass`` and ``band == keep`` with
  ``0 < frontier_pass_rate <= band_high`` and ``discrimination >= min``.
* **VAL-EXPORT-016** -- HEADLINE A: the report shows gold == 100% across the
  shipped set (from the independently re-measured ``evaluate.sh`` scores).
* **VAL-EXPORT-017** -- per-model solve-rates (tier ordering weak <= mid <=
  frontier < gold), an IRT difficulty/discrimination summary, and a
  generator + language breakdown whose counts sum to the shipped total.
* **VAL-EXPORT-018** -- HEADLINE B: a stated frontier threshold and the measured
  aggregate frontier solve-rate, strictly below the threshold yet > 0.
* **VAL-EXPORT-019** -- the shipped-task count reconciles with the jsonl line
  count, the parquet row count, and the ``tasks/*/`` directory count, and the
  JSON form parses with every required key populated.

The benchmark report consumes the gold-eval result (Headline A) as an injected
:class:`~swe_forge.forge.gold_eval.GoldEvalReport` (or its serialized dict), so
this module's aggregation/reconciliation/audit logic is unit-tested offline while
the real Docker ``evaluate.sh`` path is exercised by gold-eval.
"""

from __future__ import annotations

import json
import math
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from swe_forge.forge.calibrate.filter import (
    DEFAULT_BAND_HIGH,
    BandFilterConfig,
)
from swe_forge.forge.gold_eval import (
    DEFAULT_DETERMINISM_RUNS,
    EvalRun,
    GoldEvalError,
    GoldEvalReport,
    TaskGoldResult,
    discover_task_dirs,
    resolve_tasks_root,
)
from swe_forge.forge.models import (
    GENERATOR_NAMES,
    PANEL_TIERS,
    SUPPORTED_LANGUAGES,
    ModelSolveRecord,
)
from swe_forge.forge.oracle.mutation import DEFAULT_KILL_THRESHOLD
from swe_forge.forge.oracle.multifault import (
    MULTIFAULT_GENERATORS,
    MultiFaultCompletenessEvidence,
    MultiFaultError,
)
from swe_forge.forge.oracle.teacher_evidence import teacher_gate_evidence_issues
from swe_forge.forge.publication import protected_teacher_receipts_path

#: Default frontier threshold for HEADLINE B. The keep band includes
#: ``band_high`` itself, so the independently stated headline threshold must sit
#: strictly above that edge. This preserves a robust ``aggregate < threshold``
#: guarantee even when every kept task lands exactly on the inclusive keep edge;
#: it does not alter the calibration keep band.
DEFAULT_FRONTIER_THRESHOLD = DEFAULT_BAND_HIGH + 0.01

#: A tiny tolerance for the (approximately) nondecreasing per-tier ordering, so
#: floating-point noise never trips weak <= mid <= frontier.
_TIER_ORDER_TOLERANCE = 1e-9

#: The standard dataset file names an export ``out_dir`` carries.
DEFAULT_JSONL_NAME = "dataset.jsonl"
DEFAULT_PARQUET_NAME = "dataset.parquet"


class ReportError(RuntimeError):
    """Raised when a report cannot be built (missing tasks dir / unreadable input)."""


# --------------------------------------------------------------------------- #
# Per-task provenance view
# --------------------------------------------------------------------------- #
def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_timestamp(value: object) -> bool:
    """True iff ``value`` is a parseable ISO-8601 timestamp string."""
    if not isinstance(value, str) or not value.strip():
        return False
    candidate = value.strip()
    normalized = candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
    try:
        datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return True


@dataclass
class TaskProvenance:
    """The provenance of one shipped ``tasks/<id>/`` as the report reads it.

    Loaded from the task's ``provenance.json`` (the authoritative per-task record
    written at export), with ``workspace.yaml`` consulted only as a fallback for
    the language/verdict fields. ``raw`` keeps the original provenance dict so the
    completeness check can detect a missing/empty field rather than a defaulted
    one.
    """

    task_id: str
    language: str
    generator: str
    seed: object
    created_at: object
    tool_versions: dict[str, object]
    mutants_total: object
    mutants_killed: object
    irt_difficulty: object
    irt_discrimination: object
    frontier_pass_at_k: object
    oracle_verdict: str
    band_verdict: str
    panel: list[dict[str, object]]
    protected_teacher_transport_receipts: object = field(default=None, repr=False)
    raw: dict[str, object] = field(default_factory=dict)
    details: dict[str, object] = field(default_factory=dict)

    @classmethod
    def load(cls, task_dir: Path | str) -> TaskProvenance:
        """Read one task dir's provenance (``provenance.json`` + ``workspace.yaml``)."""
        task_path = Path(task_dir)
        prov: dict[str, object] = {}
        prov_file = task_path / "provenance.json"
        if prov_file.is_file():
            try:
                loaded = json.loads(prov_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ReportError(
                    f"invalid provenance.json in {task_path.name}: {exc}"
                ) from exc
            if isinstance(loaded, dict):
                prov = loaded

        details_raw = prov.get("details", {})
        details: dict[str, object] = (
            dict(details_raw) if isinstance(details_raw, dict) else {}
        )

        workspace: dict[str, object] = {}
        ws_file = task_path / "workspace.yaml"
        if ws_file.is_file():
            loaded_ws = yaml.safe_load(ws_file.read_text(encoding="utf-8")) or {}
            if isinstance(loaded_ws, dict):
                workspace = loaded_ws
        ws_meta = workspace.get("meta", {})
        ws_meta = ws_meta if isinstance(ws_meta, dict) else {}

        tool_versions_raw = prov.get("tool_versions", {})
        tool_versions: dict[str, object] = (
            dict(tool_versions_raw) if isinstance(tool_versions_raw, dict) else {}
        )

        panel_raw = details.get("panel", [])
        panel: list[dict[str, object]] = (
            [dict(rec) for rec in panel_raw if isinstance(rec, dict)]
            if isinstance(panel_raw, list)
            else []
        )
        protected_receipts: object = None
        resolved = task_path.resolve()
        if resolved.parent.name == "tasks":
            receipt_path = protected_teacher_receipts_path(
                resolved.parent.parent, task_path.name
            )
            try:
                metadata = receipt_path.lstat()
                if (
                    stat.S_ISREG(metadata.st_mode)
                    and not stat.S_ISLNK(metadata.st_mode)
                    and not metadata.st_mode & 0o077
                ):
                    protected_receipts = json.loads(
                        receipt_path.read_text(encoding="utf-8")
                    )
            except (OSError, json.JSONDecodeError):
                protected_receipts = None

        def _pick(*keys: str, default: object = "") -> object:
            for source in (details, prov, ws_meta):
                for key in keys:
                    if key in source and source[key] not in (None, ""):
                        return source[key]
            return default

        return cls(
            task_id=task_path.name,
            language=str(_pick("language") or workspace.get("language", "")),
            generator=str(_pick("generator")),
            seed=_pick("seed", default=None),
            created_at=_pick("created_at", default=None),
            tool_versions=tool_versions,
            mutants_total=details.get("mutants_total"),
            mutants_killed=details.get("mutants_killed"),
            irt_difficulty=details.get("irt_difficulty"),
            irt_discrimination=details.get("irt_discrimination"),
            frontier_pass_at_k=details.get("frontier_pass_at_k"),
            oracle_verdict=str(_pick("oracle_verdict")),
            band_verdict=str(_pick("band_verdict")),
            panel=panel,
            protected_teacher_transport_receipts=protected_receipts,
            raw=prov,
            details=details,
        )

    # -- typed accessors (best-effort numeric coercion for the aggregates) --- #
    def _float(self, value: object) -> float | None:
        if _is_number(value):
            return float(value)  # type: ignore[arg-type]
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    @property
    def difficulty(self) -> float | None:
        return self._float(self.irt_difficulty)

    @property
    def discrimination(self) -> float | None:
        return self._float(self.irt_discrimination)

    @property
    def frontier_rate(self) -> float | None:
        return self._float(self.frontier_pass_at_k)

    def panel_records(self) -> list[ModelSolveRecord]:
        """The panel entries that parse as valid :class:`ModelSolveRecord`s."""
        records: list[ModelSolveRecord] = []
        for entry in self.panel:
            try:
                records.append(ModelSolveRecord.from_dict(entry))
            except Exception:  # noqa: BLE001 - a malformed entry is a finding, not a crash
                continue
        return records

    def panel_tiers(self) -> set[str]:
        tiers: set[str] = set()
        for entry in self.panel:
            tier = entry.get("tier")
            if isinstance(tier, str):
                tiers.add(tier)
        return tiers


# --------------------------------------------------------------------------- #
# VAL-EXPORT-014: provenance completeness
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CompletenessFinding:
    """The missing/invalid provenance fields detected for one shipped task."""

    task_id: str
    missing: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"task_id": self.task_id, "missing": list(self.missing)}


@dataclass
class ProvenanceCompletenessResult:
    """Roll-up of the per-task provenance completeness check (VAL-EXPORT-014)."""

    checked: int
    findings: list[CompletenessFinding]
    kill_threshold: float

    @property
    def complete(self) -> int:
        return self.checked - len(self.findings)

    @property
    def passed(self) -> bool:
        """0 tasks with any missing field (and at least one task checked)."""
        return self.checked > 0 and not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "checked": self.checked,
            "complete": self.complete,
            "passed": self.passed,
            "kill_threshold": self.kill_threshold,
            "findings": [f.to_dict() for f in self.findings],
        }


def _completeness_missing(prov: TaskProvenance, kill_threshold: float) -> list[str]:
    missing: list[str] = []

    if prov.generator not in GENERATOR_NAMES:
        missing.append("generator")
    if not (isinstance(prov.seed, int) and not isinstance(prov.seed, bool)):
        missing.append("seed")
    if prov.language not in SUPPORTED_LANGUAGES:
        missing.append("language")
    if not prov.tool_versions:
        missing.append("tool_versions")
    if not _parse_timestamp(prov.created_at):
        missing.append("created_at")

    total = prov.mutants_total
    killed = prov.mutants_killed
    if not (isinstance(total, int) and not isinstance(total, bool) and total > 0):
        missing.append("mutants_total")
    elif not (isinstance(killed, int) and not isinstance(killed, bool)):
        missing.append("mutants_killed")
    elif not (0 <= killed <= total):
        missing.append("mutants_killed")
    elif killed / total < kill_threshold:
        missing.append("mutants_killed<threshold")

    if not (_is_number(prov.irt_difficulty) and math.isfinite(prov.difficulty or 0.0)):
        missing.append("irt_difficulty")
    if not (
        _is_number(prov.irt_discrimination)
        and math.isfinite(prov.discrimination or 0.0)
    ):
        missing.append("irt_discrimination")

    records = prov.panel_records()
    if not records:
        missing.append("panel")
    else:
        present = {rec.tier for rec in records}
        for tier in PANEL_TIERS:
            if tier not in present:
                missing.append(f"panel:{tier}")
    if prov.frontier_rate is None:
        missing.append("frontier_pass_at_k")
    missing.extend(
        teacher_gate_evidence_issues(
            prov.details,
            candidate=prov.details.get("candidate_transport_fingerprint"),
            protected_receipts=prov.protected_teacher_transport_receipts,
        )
    )
    missing.extend(_multifault_evidence_issues(prov))

    return missing


def _multifault_evidence_issues(prov: TaskProvenance) -> list[str]:
    """Return completeness failures for a multi-fault task's shipped proof."""

    if prov.generator not in MULTIFAULT_GENERATORS:
        return []
    raw = prov.details.get("multifault_completeness")
    if not isinstance(raw, dict):
        return ["multifault_completeness"]
    try:
        evidence = MultiFaultCompletenessEvidence.from_dict(raw)
    except (KeyError, TypeError, ValueError, MultiFaultError):
        return ["multifault_completeness"]
    if not evidence.p2p_command.strip():
        return ["multifault_completeness:p2p_command"]
    issues: list[str] = []
    expected = list(range(len(evidence.constituents)))
    actual = [record.index for record in evidence.constituents]
    if actual != expected:
        issues.append("multifault_completeness:indexes")
    for record in evidence.constituents:
        if (
            record.verdict != "pass"
            or not record.other_inverse_patches_applied
            or not record.p2p_passed
            or not record.failed_f2p_test_ids
        ):
            issues.append(f"multifault_completeness:constituent:{record.index}")
    return issues


def check_provenance_completeness(
    provenances: Sequence[TaskProvenance],
    *,
    kill_threshold: float = DEFAULT_KILL_THRESHOLD,
) -> ProvenanceCompletenessResult:
    """Check every shipped task records the mandated provenance (VAL-EXPORT-014)."""
    findings = [
        CompletenessFinding(task_id=prov.task_id, missing=missing)
        for prov in provenances
        if (missing := _completeness_missing(prov, kill_threshold))
    ]
    return ProvenanceCompletenessResult(
        checked=len(provenances), findings=findings, kill_threshold=kill_threshold
    )


# --------------------------------------------------------------------------- #
# VAL-EXPORT-015: provenance consistency with the gates
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConsistencyFinding:
    """The gate-consistency violations detected for one shipped task."""

    task_id: str
    issues: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"task_id": self.task_id, "issues": list(self.issues)}


@dataclass
class ProvenanceConsistencyResult:
    """Roll-up of the per-task gate-consistency check (VAL-EXPORT-015)."""

    checked: int
    findings: list[ConsistencyFinding]
    band_high: float
    discrimination_threshold: float

    @property
    def consistent(self) -> int:
        return self.checked - len(self.findings)

    @property
    def passed(self) -> bool:
        return self.checked > 0 and not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "checked": self.checked,
            "consistent": self.consistent,
            "passed": self.passed,
            "band_high": self.band_high,
            "discrimination_threshold": self.discrimination_threshold,
            "findings": [f.to_dict() for f in self.findings],
        }


def _consistency_issues(prov: TaskProvenance, config: BandFilterConfig) -> list[str]:
    issues: list[str] = []
    if prov.oracle_verdict != "pass":
        issues.append(f"oracle_verdict={prov.oracle_verdict!r} (expected 'pass')")
    if prov.band_verdict != "keep":
        issues.append(f"band_verdict={prov.band_verdict!r} (expected 'keep')")

    frontier = prov.frontier_rate
    if frontier is None:
        issues.append("frontier_pass_rate missing")
    elif not (0.0 < frontier <= config.band_high):
        issues.append(
            f"frontier_pass_rate {frontier:.4f} outside the keep band "
            f"(0, {config.band_high:.4f}]"
        )

    discrimination = prov.discrimination
    if discrimination is None:
        issues.append("irt_discrimination missing")
    elif discrimination < config.discrimination_threshold:
        issues.append(
            f"irt_discrimination {discrimination:.4f} < keep threshold "
            f"{config.discrimination_threshold:.4f}"
        )
    issues.extend(
        teacher_gate_evidence_issues(
            prov.details,
            candidate=prov.details.get("candidate_transport_fingerprint"),
            protected_receipts=prov.protected_teacher_transport_receipts,
        )
    )
    issues.extend(_multifault_evidence_issues(prov))
    return issues


def check_provenance_consistency(
    provenances: Sequence[TaskProvenance],
    *,
    config: BandFilterConfig | None = None,
) -> ProvenanceConsistencyResult:
    """Check each shipped task's provenance agrees with its gates (VAL-EXPORT-015)."""
    cfg = config or BandFilterConfig()
    findings = [
        ConsistencyFinding(task_id=prov.task_id, issues=issues)
        for prov in provenances
        if (issues := _consistency_issues(prov, cfg))
    ]
    return ProvenanceConsistencyResult(
        checked=len(provenances),
        findings=findings,
        band_high=cfg.band_high,
        discrimination_threshold=cfg.discrimination_threshold,
    )


# --------------------------------------------------------------------------- #
# Panel aggregation (per-model + per-tier solve-rates)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelAggregate:
    """One panel model's solve-rate pooled across the shipped set."""

    model: str
    tier: str
    tasks: int
    k_total: int
    solves_total: int

    @property
    def solve_rate(self) -> float:
        return self.solves_total / self.k_total if self.k_total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "tier": self.tier,
            "tasks": self.tasks,
            "k_total": self.k_total,
            "solves_total": self.solves_total,
            "solve_rate": self.solve_rate,
        }


def aggregate_panel(
    provenances: Sequence[TaskProvenance],
) -> tuple[list[ModelAggregate], dict[str, float]]:
    """Pool the per-task panel matrices into per-model and per-tier solve-rates.

    Returns the per-model aggregates (sorted weak->mid->frontier then by model id)
    and the per-tier pooled solve-rate map ``{tier: solves/k}`` over every record
    of that tier across the shipped set.
    """
    tier_of: dict[str, str] = {}
    model_tasks: dict[str, int] = {}
    model_k: dict[str, int] = {}
    model_solves: dict[str, int] = {}
    tier_k: dict[str, int] = {}
    tier_solves: dict[str, int] = {}

    for prov in provenances:
        for rec in prov.panel_records():
            tier_of.setdefault(rec.model, rec.tier)
            model_tasks[rec.model] = model_tasks.get(rec.model, 0) + 1
            model_k[rec.model] = model_k.get(rec.model, 0) + rec.k
            model_solves[rec.model] = model_solves.get(rec.model, 0) + rec.solves
            tier_k[rec.tier] = tier_k.get(rec.tier, 0) + rec.k
            tier_solves[rec.tier] = tier_solves.get(rec.tier, 0) + rec.solves

    tier_rank = {tier: i for i, tier in enumerate(PANEL_TIERS)}
    aggregates = [
        ModelAggregate(
            model=model,
            tier=tier_of[model],
            tasks=model_tasks[model],
            k_total=model_k[model],
            solves_total=model_solves[model],
        )
        for model in tier_of
    ]
    aggregates.sort(key=lambda a: (tier_rank.get(a.tier, len(PANEL_TIERS)), a.model))

    tier_rates = {
        tier: (tier_solves[tier] / tier_k[tier] if tier_k[tier] else 0.0)
        for tier in tier_k
    }
    return aggregates, tier_rates


def _tier_ordering_ok(tier_rates: dict[str, float]) -> bool:
    """True iff present tiers are (approximately) nondecreasing weak->mid->frontier."""
    ordered = [tier_rates[tier] for tier in PANEL_TIERS if tier in tier_rates]
    return all(
        ordered[i] <= ordered[i + 1] + _TIER_ORDER_TOLERANCE
        for i in range(len(ordered) - 1)
    )


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


# --------------------------------------------------------------------------- #
# Gold (HEADLINE A) summary
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GoldSummary:
    """The Headline A gold roll-up the report surfaces (VAL-EXPORT-016)."""

    measured: bool
    gold_count: int
    shipped_count: int
    all_gold: bool
    deterministic: bool
    results: tuple[TaskGoldResult, ...] = ()

    @property
    def gold_rate(self) -> float:
        return self.gold_count / self.shipped_count if self.strict_proof else 0.0

    @property
    def strict_proof(self) -> bool:
        """Whether the retained records prove every task twice, strictly."""
        return (
            self.measured
            and self.shipped_count > 0
            and self.shipped_count == len(self.results)
            and self.gold_count == self.shipped_count
            and self.all_gold
            and self.deterministic
            and len({result.task_id for result in self.results}) == len(self.results)
            and all(
                len(result.runs) >= DEFAULT_DETERMINISM_RUNS
                and result.gold
                and result.deterministic
                for result in self.results
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "measured": self.measured,
            "gold_count": self.gold_count,
            "shipped_count": self.shipped_count,
            "gold_rate": self.gold_rate,
            "all_gold": self.all_gold,
            "deterministic": self.deterministic,
            "strict_proof": self.strict_proof,
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def unmeasured(cls) -> GoldSummary:
        return cls(
            measured=False,
            gold_count=0,
            shipped_count=0,
            all_gold=False,
            deterministic=False,
            results=(),
        )

    @classmethod
    def from_gold_eval(cls, report: GoldEvalReport | dict[str, object]) -> GoldSummary:
        """Build the summary from a :class:`GoldEvalReport` or its serialized dict."""
        if isinstance(report, GoldEvalReport):
            return cls(
                measured=True,
                gold_count=report.gold_count,
                shipped_count=report.shipped_count,
                all_gold=report.all_gold,
                deterministic=report.deterministic,
                results=tuple(report.results),
            )
        parsed = _parse_gold_results(report.get("results"))
        gold_count = sum(1 for result in parsed if result.gold)
        shipped = len(parsed)
        return cls(
            measured=True,
            gold_count=gold_count,
            shipped_count=shipped,
            all_gold=bool(parsed) and gold_count == shipped,
            deterministic=bool(parsed)
            and all(result.deterministic for result in parsed),
            results=tuple(parsed),
        )


def _parse_gold_results(value: object) -> list[TaskGoldResult]:
    """Decode saved Headline A proof records, rejecting aggregate-only claims."""
    if not isinstance(value, list):
        return []
    results: list[TaskGoldResult] = []
    for item in value:
        if not isinstance(item, dict):
            return []
        task_id = item.get("task_id")
        image = item.get("image")
        raw_runs = item.get("runs")
        if not isinstance(task_id, str) or not task_id or not isinstance(image, str):
            return []
        if not isinstance(raw_runs, list):
            return []
        runs: list[EvalRun] = []
        for raw_run in raw_runs:
            if not isinstance(raw_run, dict):
                return []
            run_task_id = raw_run.get("task_id", task_id)
            run_index = raw_run.get("run_index")
            score = raw_run.get("score")
            phase1 = raw_run.get("phase1_passed")
            exit_code = raw_run.get("exit_code")
            container_name = raw_run.get("container_name")
            if (
                not isinstance(run_task_id, str)
                or run_task_id != task_id
                or not isinstance(run_index, int)
                or isinstance(run_index, bool)
                or (
                    score is not None
                    and (
                        not isinstance(score, int)
                        or isinstance(score, bool)
                        or score not in (0, 1)
                    )
                )
                or not isinstance(phase1, bool)
                or not isinstance(exit_code, int)
                or isinstance(exit_code, bool)
                or not isinstance(container_name, str)
                or not container_name
            ):
                return []
            runs.append(
                EvalRun(
                    task_id=task_id,
                    run_index=run_index,
                    score=score,
                    phase1_passed=phase1,
                    exit_code=exit_code,
                    container_name=container_name,
                )
            )
        if len({run.run_index for run in runs}) != len(runs):
            return []
        results.append(
            TaskGoldResult(
                task_id=task_id,
                task_dir=Path(task_id),
                image=image,
                runs=runs,
            )
        )
    return results


# --------------------------------------------------------------------------- #
# Count reconciliation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CountReconciliation:
    """jsonl == parquet == tasks/*/ count reconciliation (VAL-EXPORT-019)."""

    tasks: int
    jsonl: int
    parquet: int

    @property
    def reconciled(self) -> bool:
        return self.tasks == self.jsonl == self.parquet

    def to_dict(self) -> dict[str, object]:
        return {
            "tasks": self.tasks,
            "jsonl": self.jsonl,
            "parquet": self.parquet,
            "reconciled": self.reconciled,
        }


def count_jsonl_records(path: Path | str) -> int:
    """Count the non-empty lines (records) of a dataset jsonl, 0 if absent."""
    p = Path(path)
    if not p.is_file():
        return 0
    with p.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def count_parquet_rows(path: Path | str) -> int:
    """Count the rows of a dataset parquet, 0 if absent."""
    p = Path(path)
    if not p.is_file():
        return 0
    import pyarrow.parquet as pq

    return int(pq.read_metadata(str(p)).num_rows)


# --------------------------------------------------------------------------- #
# The benchmark report
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkReport:
    """The Stage 5 benchmark report: the two headlines + provenance audit + counts."""

    tasks_dir: Path
    provenances: list[TaskProvenance]
    gold: GoldSummary
    per_model: list[ModelAggregate]
    tier_solve_rates: dict[str, float]
    irt_difficulty: dict[str, float]
    irt_discrimination: dict[str, float]
    generator_breakdown: dict[str, int]
    language_breakdown: dict[str, int]
    frontier_threshold: float
    frontier_solve_rate: float
    counts: CountReconciliation
    completeness: ProvenanceCompletenessResult
    consistency: ProvenanceConsistencyResult
    # The pilot passes its full eligibility/admission funnel and current
    # SourceRegistry snapshots here. Standalone report builds intentionally leave
    # these empty, because they only have published artifacts to inspect.
    funnel: dict[str, int] = field(default_factory=dict)
    source_capacity: list[dict[str, object]] = field(default_factory=list)

    @property
    def shipped_count(self) -> int:
        return len(self.provenances)

    @property
    def tier_ordering_ok(self) -> bool:
        """weak <= mid <= frontier on the pooled per-tier rates (VAL-EXPORT-017)."""
        return _tier_ordering_ok(self.tier_solve_rates)

    @property
    def gold_ge_frontier(self) -> bool:
        """gold(100%) >= frontier: the gold rate dominates the frontier rate."""
        return self.gold.gold_rate + _TIER_ORDER_TOLERANCE >= self.frontier_solve_rate

    @property
    def breakdown_reconciles(self) -> bool:
        """Generator and language breakdown counts each sum to the shipped total."""
        return (
            sum(self.generator_breakdown.values()) == self.shipped_count
            and sum(self.language_breakdown.values()) == self.shipped_count
        )

    @property
    def funnel_reconciles(self) -> bool:
        """Validate calibrated keeps separately from cap-admitted exports."""
        if not self.funnel:
            return True
        required = (
            "sourced",
            "env_built",
            "synthesized",
            "oracle_pass",
            "calibration_keep",
            "cap_admitted",
            "exported",
            "export_refused",
        )
        if any(key not in self.funnel for key in required):
            return False
        values = [self.funnel[key] for key in required]
        if any(value < 0 for value in values):
            return False
        return (
            values[0] >= values[1] >= values[2] >= values[3] >= values[4] >= values[5]
            and values[5] == values[6] == self.shipped_count
            and values[7] == 0
        )

    @property
    def capacity_reconciles(self) -> bool:
        """Used source capacity equals the cap-admitted published task set."""
        if not self.source_capacity:
            return True
        used = 0
        for snapshot in self.source_capacity:
            cap = snapshot.get("cap")
            current = snapshot.get("used")
            remaining = snapshot.get("remaining")
            if not (
                isinstance(cap, int)
                and isinstance(current, int)
                and isinstance(remaining, int)
                and 0 <= current <= cap
                and remaining == cap - current
            ):
                return False
            used += current
        expected = self.funnel.get("cap_admitted", self.shipped_count)
        return used == expected

    @property
    def frontier_below_threshold(self) -> bool:
        return self.frontier_solve_rate < self.frontier_threshold

    @property
    def frontier_nonzero(self) -> bool:
        return self.frontier_solve_rate > 0.0

    @property
    def headline_a_pass(self) -> bool:
        """gold == 100% across the shipped set (VAL-EXPORT-016)."""
        return (
            self.gold.strict_proof
            and {result.task_id for result in self.gold.results}
            == {prov.task_id for prov in self.provenances}
            and self.shipped_count > 0
        )

    @property
    def headline_b_pass(self) -> bool:
        """frontier solve-rate strictly below the stated threshold yet > 0 (VAL-EXPORT-018)."""
        return self.frontier_below_threshold and self.frontier_nonzero

    @property
    def passed(self) -> bool:
        """Every headline + audit + reconciliation check holds."""
        return (
            self.shipped_count > 0
            and self.headline_a_pass
            and self.headline_b_pass
            and self.tier_ordering_ok
            and self.gold_ge_frontier
            and self.breakdown_reconciles
            and self.counts.reconciled
            and self.funnel_reconciles
            and self.capacity_reconciles
            and self.completeness.passed
            and self.consistency.passed
        )

    def _task_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for prov in self.provenances:
            rows.append(
                {
                    "task_id": prov.task_id,
                    "language": prov.language,
                    "generator": prov.generator,
                    "oracle_verdict": prov.oracle_verdict,
                    "band_verdict": prov.band_verdict,
                    "irt_difficulty": prov.difficulty,
                    "irt_discrimination": prov.discrimination,
                    "frontier_pass_at_k": prov.frontier_rate,
                    "mutants_total": prov.mutants_total,
                    "mutants_killed": prov.mutants_killed,
                }
            )
        return rows

    def to_dict(self) -> dict[str, object]:
        """The machine-parseable report; every VAL-EXPORT-019 required key present."""
        return {
            "shipped_count": self.shipped_count,
            "passed": self.passed,
            "counts": self.counts.to_dict(),
            "gold": self.gold.to_dict(),
            "gold_rate": self.gold.gold_rate,
            "headline_a": {
                "gold_rate": self.gold.gold_rate,
                "gold_count": self.gold.gold_count,
                "shipped_count": self.shipped_count,
                "strict_proof": self.gold.strict_proof,
                "passed": self.headline_a_pass,
            },
            "headline_b": {
                "frontier_threshold": self.frontier_threshold,
                "frontier_solve_rate": self.frontier_solve_rate,
                "below_threshold": self.frontier_below_threshold,
                "nonzero": self.frontier_nonzero,
                "passed": self.headline_b_pass,
            },
            "frontier_threshold": self.frontier_threshold,
            "frontier_solve_rate": self.frontier_solve_rate,
            "per_model": [agg.to_dict() for agg in self.per_model],
            "tier_solve_rates": dict(self.tier_solve_rates),
            "tier_ordering_ok": self.tier_ordering_ok,
            "gold_ge_frontier": self.gold_ge_frontier,
            "irt": {
                "difficulty": dict(self.irt_difficulty),
                "discrimination": dict(self.irt_discrimination),
            },
            "generator_breakdown": dict(self.generator_breakdown),
            "language_breakdown": dict(self.language_breakdown),
            "breakdown_reconciles": self.breakdown_reconciles,
            "funnel": dict(self.funnel),
            "funnel_reconciles": self.funnel_reconciles,
            "source_capacity": [dict(snapshot) for snapshot in self.source_capacity],
            "capacity_reconciles": self.capacity_reconciles,
            "provenance_completeness": self.completeness.to_dict(),
            "provenance_consistency": self.consistency.to_dict(),
            "tasks": self._task_rows(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def to_markdown(self) -> str:
        """A human-readable Markdown rendering of the same report."""
        gold = self.gold
        gold_pct = f"{gold.gold_rate * 100:.0f}%"
        lines: list[str] = []
        lines.append("# SWE-Forge Benchmark Report")
        lines.append("")
        lines.append(f"- Shipped tasks: **{self.shipped_count}**")
        lines.append(f"- Overall: **{'PASS' if self.passed else 'FAIL'}**")
        lines.append("")

        lines.append("## Headline A - gold solvability")
        if gold.measured:
            lines.append(
                f"- gold = **{gold_pct}** ({gold.gold_count}/{gold.shipped_count}) "
                f"-- {'PASS' if self.headline_a_pass else 'FAIL'}"
            )
            lines.append(f"- deterministic across reruns: {gold.deterministic}")
        else:
            lines.append("- gold: **not measured** (run gold-eval to populate)")
        lines.append("")

        lines.append("## Headline B - frontier solve-rate")
        lines.append(f"- stated threshold: **{self.frontier_threshold:.4f}**")
        lines.append(
            f"- measured frontier solve-rate: **{self.frontier_solve_rate:.4f}** "
            f"({'< threshold and > 0' if self.headline_b_pass else 'OUT OF BAND'})"
        )
        lines.append("")

        lines.append("## Per-model panel solve-rates")
        lines.append("")
        lines.append("| model | tier | tasks | solves/k | solve-rate |")
        lines.append("| --- | --- | --- | --- | --- |")
        for agg in self.per_model:
            lines.append(
                f"| {agg.model} | {agg.tier} | {agg.tasks} | "
                f"{agg.solves_total}/{agg.k_total} | {agg.solve_rate:.4f} |"
            )
        lines.append("")
        tier_summary = ", ".join(
            f"{tier}={self.tier_solve_rates[tier]:.4f}"
            for tier in PANEL_TIERS
            if tier in self.tier_solve_rates
        )
        lines.append(
            f"- per-tier (pooled): {tier_summary or 'n/a'} "
            f"(weak <= mid <= frontier: {self.tier_ordering_ok}; "
            f"gold {gold_pct} >= frontier: {self.gold_ge_frontier})"
        )
        lines.append("")

        lines.append("## IRT difficulty / discrimination")
        lines.append(
            f"- difficulty: mean={self.irt_difficulty['mean']:.4f}, "
            f"min={self.irt_difficulty['min']:.4f}, "
            f"max={self.irt_difficulty['max']:.4f}"
        )
        lines.append(
            f"- discrimination: mean={self.irt_discrimination['mean']:.4f}, "
            f"min={self.irt_discrimination['min']:.4f}, "
            f"max={self.irt_discrimination['max']:.4f}"
        )
        lines.append("")

        lines.append("## Breakdown")
        gen_summary = ", ".join(
            f"{name}={count}"
            for name, count in sorted(self.generator_breakdown.items())
        )
        lang_summary = ", ".join(
            f"{name}={count}" for name, count in sorted(self.language_breakdown.items())
        )
        lines.append(f"- by generator: {gen_summary or 'n/a'}")
        lines.append(f"- by language: {lang_summary or 'n/a'}")
        lines.append(f"- breakdown sums to shipped total: {self.breakdown_reconciles}")
        lines.append("")

        if self.funnel:
            lines.append("## Pilot capacity funnel")
            lines.append(
                "- sourced={sourced}, env={env_built}, synth={synthesized}, "
                "oracle_pass={oracle_pass}, calibrated_keep={calibration_keep}, "
                "cap_admitted={cap_admitted}, exported={exported}, "
                "export_refused={export_refused} "
                "(reconciled: {reconciled})".format(
                    **self.funnel,
                    reconciled=self.funnel_reconciles,
                )
            )
            lines.append(f"- source capacity reconciled: {self.capacity_reconciles}")
            lines.append("")

        lines.append("## Counts reconciliation")
        lines.append(
            f"- tasks/*/ = {self.counts.tasks}, jsonl = {self.counts.jsonl}, "
            f"parquet = {self.counts.parquet} "
            f"(reconciled: {self.counts.reconciled})"
        )
        lines.append("")

        lines.append("## Provenance audit")
        lines.append(
            f"- completeness: {self.completeness.complete}/{self.completeness.checked} "
            f"complete ({'PASS' if self.completeness.passed else 'FAIL'})"
        )
        for finding in self.completeness.findings:
            lines.append(f"  - {finding.task_id}: missing {finding.missing}")
        lines.append(
            f"- consistency: {self.consistency.consistent}/{self.consistency.checked} "
            f"consistent ({'PASS' if self.consistency.passed else 'FAIL'})"
        )
        for cfinding in self.consistency.findings:
            lines.append(f"  - {cfinding.task_id}: {cfinding.issues}")
        lines.append("")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def load_task_provenances(tasks_root: Path | str) -> list[TaskProvenance]:
    """Load provenance from every structurally valid immediate task workspace."""
    root = Path(tasks_root)
    try:
        dirs = discover_task_dirs(root)
    except GoldEvalError as exc:
        raise ReportError(str(exc)) from exc
    return [TaskProvenance.load(task_dir) for task_dir in dirs]


def build_benchmark_report(
    out_dir: Path | str,
    *,
    gold: GoldSummary | GoldEvalReport | dict[str, object] | None = None,
    jsonl_path: Path | str | None = None,
    parquet_path: Path | str | None = None,
    frontier_threshold: float = DEFAULT_FRONTIER_THRESHOLD,
    band_config: BandFilterConfig | None = None,
    kill_threshold: float = DEFAULT_KILL_THRESHOLD,
    funnel: Mapping[str, object] | None = None,
    source_capacity: Sequence[Mapping[str, object]] | None = None,
) -> BenchmarkReport:
    """Assemble the benchmark report from an exported pilot ``out_dir``.

    Resolves the ``tasks/<id>/`` workspaces (``out_dir`` itself or its ``tasks/``
    subdir) and the ``dataset.jsonl`` / ``dataset.parquet`` datasets, loads each
    task's provenance, runs the completeness + gate-consistency audits, pools the
    panel matrices into per-model/per-tier solve-rates, summarizes IRT and the
    generator/language breakdown, and reconciles the counts. ``gold`` injects the
    Headline A result (a :class:`GoldEvalReport`, its dict, or a prebuilt
    :class:`GoldSummary`); when omitted the gold section is marked unmeasured.
    """
    out_path = Path(out_dir)
    try:
        tasks_root = resolve_tasks_root(out_path)
    except GoldEvalError as exc:
        raise ReportError(str(exc)) from exc
    # Datasets live alongside the tasks/ dir (i.e. in the export out_dir).
    dataset_dir = tasks_root.parent if tasks_root.name == "tasks" else out_path
    jsonl = Path(jsonl_path) if jsonl_path else dataset_dir / DEFAULT_JSONL_NAME
    parquet = Path(parquet_path) if parquet_path else dataset_dir / DEFAULT_PARQUET_NAME

    provenances = load_task_provenances(tasks_root)
    cfg = band_config or BandFilterConfig()

    if isinstance(gold, GoldSummary):
        gold_summary = gold
    elif gold is None:
        gold_summary = GoldSummary.unmeasured()
    else:
        gold_summary = GoldSummary.from_gold_eval(gold)

    per_model, tier_rates = aggregate_panel(provenances)
    frontier_rate = tier_rates.get("frontier", 0.0)

    difficulties = [
        prov.difficulty for prov in provenances if prov.difficulty is not None
    ]
    discriminations = [
        prov.discrimination for prov in provenances if prov.discrimination is not None
    ]

    generator_breakdown: dict[str, int] = {}
    language_breakdown: dict[str, int] = {}
    for prov in provenances:
        generator_breakdown[prov.generator] = (
            generator_breakdown.get(prov.generator, 0) + 1
        )
        language_breakdown[prov.language] = language_breakdown.get(prov.language, 0) + 1

    counts = CountReconciliation(
        tasks=len(provenances),
        jsonl=count_jsonl_records(jsonl),
        parquet=count_parquet_rows(parquet),
    )

    parsed_funnel = {
        str(key): int(value)
        for key, value in (funnel or {}).items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    parsed_capacity = [dict(snapshot) for snapshot in (source_capacity or ())]

    return BenchmarkReport(
        tasks_dir=tasks_root,
        provenances=provenances,
        gold=gold_summary,
        per_model=per_model,
        tier_solve_rates=tier_rates,
        irt_difficulty=_stats(difficulties),
        irt_discrimination=_stats(discriminations),
        generator_breakdown=generator_breakdown,
        language_breakdown=language_breakdown,
        frontier_threshold=frontier_threshold,
        frontier_solve_rate=frontier_rate,
        counts=counts,
        completeness=check_provenance_completeness(
            provenances, kill_threshold=kill_threshold
        ),
        consistency=check_provenance_consistency(provenances, config=cfg),
        funnel=parsed_funnel,
        source_capacity=parsed_capacity,
    )


def write_report(report: BenchmarkReport, out_dir: Path | str) -> tuple[Path, Path]:
    """Write the report to ``report.md`` + ``report.json`` under ``out_dir``."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    md_path = out_path / "report.md"
    json_path = out_path / "report.json"
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    json_path.write_text(report.to_json() + "\n", encoding="utf-8")
    return md_path, json_path


__all__ = [
    "DEFAULT_FRONTIER_THRESHOLD",
    "BenchmarkReport",
    "CompletenessFinding",
    "ConsistencyFinding",
    "CountReconciliation",
    "GoldSummary",
    "ModelAggregate",
    "ProvenanceCompletenessResult",
    "ProvenanceConsistencyResult",
    "ReportError",
    "TaskProvenance",
    "aggregate_panel",
    "build_benchmark_report",
    "check_provenance_completeness",
    "check_provenance_consistency",
    "count_jsonl_records",
    "count_parquet_rows",
    "load_task_provenances",
    "write_report",
]
