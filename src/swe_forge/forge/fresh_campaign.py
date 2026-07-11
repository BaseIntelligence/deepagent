"""Offline-safe orchestration primitives for the authorized fresh $50 campaign.

The live campaign is intentionally a thin consumer of this module.  All
provider-facing requests are still made by ``forge.teacher``/``forge.panel``;
this module supplies the durable admission, accounting, evidence, and
transaction boundaries around those existing stages.
"""

from __future__ import annotations

import hashlib
import asyncio
import json
import os
import subprocess
import uuid
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Protocol

from swe_forge.execution.sandbox import docker_name_prefix
from swe_forge.forge.calibrate.filter import DEFAULT_BAND_FILTER, BandFilterConfig
from swe_forge.forge.export import (
    ExportRequest,
    TaskExportResult,
    _write_staged_workspace,
    assemble_forge_task,
    export_dataset,
)
from swe_forge.forge.models import ForgeTask
from swe_forge.forge.pilot import (
    CandidatePlan,
    CandidateProcessor,
    _keep_export_request,
)
from swe_forge.forge.publication import (
    PublicationEntry,
    PublicationError,
    load_published_generation,
    publish_generation,
)
from swe_forge.forge.recovery_accounting import (
    RecoveryAccountingError,
    RecoveryBudgetLedger,
    campaign_call_context,
    reconcile_recovery_reports,
)

CAP_USD = Decimal("50.00")
CAMPAIGN_ID = "fresh-harvest-2026-07-11"
HISTORICAL_HARVEST_LEDGER_NAME = "harvest_progress.json"
PROTECTED_DOCKER_RESOURCES = frozenset({"mission-test-pg", "acproxy"})


class FreshCampaignError(RuntimeError):
    """Raised when a fresh campaign cannot prove a fail-closed invariant."""


def candidate_identity(plan: CandidatePlan) -> str:
    """Return a stable identity for one candidate plan, independent of run id."""
    payload = {
        "repo_id": plan.repo.repo_id,
        "commit": plan.repo.commit,
        "generator": plan.generator,
        "seed": plan.seed,
        "file": plan.file,
        "symbol": plan.symbol,
        "op": plan.op,
        "params": dict(sorted(plan.params.items())),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return f"{plan.repo.repo_id}::{plan.generator}::{plan.seed}::{digest}"


def candidate_replay_key(plan: CandidatePlan) -> str:
    """Return a coarse key for historical records lacking pinned commit data."""
    return _candidate_replay_key_from_fields(
        plan.repo.repo_id, plan.generator, plan.seed, dict(plan.params)
    )


def _candidate_replay_key_from_fields(
    repo_id: str, generator: str, seed: int, params: dict[str, object]
) -> str:
    payload = {
        "repo_id": repo_id,
        "generator": generator,
        "seed": seed,
        "params": dict(sorted(params.items())),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return f"{repo_id}::{generator}::{seed}::{digest}"


@dataclass(frozen=True)
class FreshIdentityClaim:
    """Durable admission evidence for one never-before-attempted candidate."""

    identity: str
    reason: str
    plan: dict[str, object]


class FreshCandidateAuthority:
    """Crash-durable global no-replay authority for candidate identities."""

    def __init__(
        self,
        path: Path | str,
        *,
        historical_identities: Iterable[str] = (),
        terminal_identities: Iterable[str] = (),
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._historical = {str(value) for value in historical_identities if value}
        self._historical.update(str(value) for value in terminal_identities if value)
        self._claimed: set[str] = set()
        self._terminal: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FreshCampaignError(
                        "fresh candidate authority contains malformed JSON"
                    ) from exc
                if not isinstance(event, dict) or event.get("event") not in {
                    "claim",
                    "terminal",
                }:
                    raise FreshCampaignError(
                        "fresh candidate authority contains an invalid event"
                    )
                identity = event.get("identity")
                if not isinstance(identity, str) or not identity:
                    raise FreshCampaignError(
                        "fresh candidate authority contains an invalid identity"
                    )
                self._claimed.add(identity)
                if event.get("event") == "terminal":
                    self._terminal.add(identity)
        if self._terminal and not self._terminal.issubset(self._claimed):
            raise FreshCampaignError(
                "fresh candidate authority contains an orphan terminal event"
            )

    @property
    def claimed_identities(self) -> frozenset[str]:
        return frozenset(self._claimed)

    @property
    def historical_identities(self) -> frozenset[str]:
        return frozenset(self._historical)

    def claim(self, plan: CandidatePlan, *, reason: str) -> FreshIdentityClaim:
        identity = candidate_identity(plan)
        replay_key = candidate_replay_key(plan)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            flock(lock_handle.fileno(), LOCK_EX)
            try:
                self._reload()
                if identity in self._historical or replay_key in self._historical:
                    raise FreshCampaignError(
                        f"candidate identity {identity!r} is historical or terminal replay"
                    )
                if identity in self._claimed:
                    raise FreshCampaignError(
                        f"candidate identity {identity!r} was already attempted"
                    )
                event = {
                    "schema_version": 1,
                    "event": "claim",
                    "identity": identity,
                    "reason": str(reason).strip() or "fresh unprocessed candidate",
                    "plan": plan.to_dict(),
                }
                self._append_event(event)
                self._claimed.add(identity)
                return FreshIdentityClaim(
                    identity, str(event["reason"]), plan.to_dict()
                )
            finally:
                flock(lock_handle.fileno(), LOCK_UN)

    def _append_event(self, event: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _reload(self) -> None:
        if not self.path.exists():
            self._claimed = set()
            self._terminal = set()
            return
        claimed: set[str] = set()
        terminal: set[str] = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if (
                not isinstance(event, dict)
                or event.get("event") not in {"claim", "terminal"}
                or not isinstance(event.get("identity"), str)
            ):
                raise FreshCampaignError(
                    "fresh candidate authority contains an invalid event"
                )
            identity = event["identity"]
            claimed.add(identity)
            if event["event"] == "terminal":
                terminal.add(identity)
        if not terminal.issubset(claimed):
            raise FreshCampaignError(
                "fresh candidate authority contains an orphan terminal event"
            )
        self._claimed = claimed
        self._terminal = terminal

    def terminalize(self, identity: str, *, status: str, reason: str) -> None:
        """Persist a terminal disposition so restart cannot replay the identity."""
        if identity not in self._claimed:
            raise FreshCampaignError(
                f"cannot terminalize unclaimed candidate identity {identity!r}"
            )
        if not str(status).strip() or not str(reason).strip():
            raise FreshCampaignError("terminal candidate evidence is incomplete")
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            flock(lock_handle.fileno(), LOCK_EX)
            try:
                self._reload()
                if identity not in self._terminal:
                    self._append_event(
                        {
                            "schema_version": 1,
                            "event": "terminal",
                            "identity": identity,
                            "status": str(status),
                            "reason": str(reason),
                        }
                    )
                    self._terminal.add(identity)
            finally:
                flock(lock_handle.fileno(), LOCK_UN)

    def recover_unfinished(self) -> frozenset[str]:
        """Return claimed identities with no terminal event for fail-closed restart."""
        return frozenset(self._claimed - self._terminal)


def historical_candidate_identities(
    *,
    dispositions_path: Path | str | None = None,
    historical_task_ids: Iterable[str] = (),
) -> frozenset[str]:
    """Derive identity keys from immutable historical disposition/task records."""
    identities = {str(value) for value in historical_task_ids if value}
    identities.add("mahmoud/boltons::bug_combination::0::terminal-alternate-recovery")
    identities.add("mahmoud-boltons__bug_combination__7bb4e61cc98c")
    if dispositions_path is not None:
        path = Path(dispositions_path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise FreshCampaignError(
                        "historical dispositions contain a non-object record"
                    )
                repo_id = payload.get("repo_id")
                generator = payload.get("generator")
                seed = payload.get("seed")
                params = payload.get("params", {})
                if (
                    isinstance(repo_id, str)
                    and isinstance(generator, str)
                    and isinstance(seed, int)
                    and isinstance(params, dict)
                ):
                    digest = hashlib.sha256(
                        json.dumps(
                            {
                                "repo_id": repo_id,
                                "generator": generator,
                                "seed": seed,
                                "params": dict(sorted(params.items())),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest()[:24]
                    identities.add(f"{repo_id}::{generator}::{seed}::{digest}")
                    identities.add(
                        _candidate_replay_key_from_fields(
                            repo_id, generator, seed, params
                        )
                    )
    return frozenset(identities)


class FreshCampaignLedger:
    """A distinct, durable $50 ledger with request-level reservation."""

    def __init__(
        self,
        path: Path | str,
        *,
        run_id: str,
        worst_case_cost_usd: Decimal | str | float,
        cap_usd: Decimal | str | float = CAP_USD,
        historical_ledger: Path | str | None = None,
    ) -> None:
        self.path = Path(path)
        self.run_id = str(run_id).strip()
        self.cap_usd = Decimal(str(cap_usd))
        if self.cap_usd != CAP_USD:
            raise FreshCampaignError("fresh campaign cap_usd must be exactly 50.00")
        if (
            historical_ledger is not None
            and self.path.resolve() == Path(historical_ledger).resolve()
        ):
            raise FreshCampaignError(
                "fresh campaign ledger must not reuse historical harvest state"
            )
        self.historical_ledger = (
            str(Path(historical_ledger).resolve()) if historical_ledger else ""
        )
        self._ledger = RecoveryBudgetLedger(
            self.path,
            run_id=self.run_id,
            cap_usd=self.cap_usd,
            worst_case_cost_usd=worst_case_cost_usd,
        )
        self.metadata_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        metadata = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "run_id": self.run_id,
            "cap_usd": format(self.cap_usd, "f"),
            "historical_ledger": self.historical_ledger,
        }
        if self.metadata_path.exists():
            try:
                existing = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise FreshCampaignError(
                    "fresh campaign ledger metadata is malformed"
                ) from exc
            if existing != metadata:
                raise FreshCampaignError("fresh campaign ledger metadata mismatch")
        else:
            temp = self.metadata_path.with_name(
                f".{self.metadata_path.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temp.open("w", encoding="utf-8") as handle:
                    handle.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, self.metadata_path)
                descriptor = os.open(self.metadata_path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                temp.unlink(missing_ok=True)

    @property
    def total_exact_cost(self) -> Decimal:
        return self._ledger.total_exact_cost

    @property
    def total_active_reservations(self) -> Decimal:
        return self._ledger.total_active_reservations

    @property
    def remaining_cap(self) -> Decimal:
        return self.cap_usd - self.total_exact_cost - self.total_active_reservations

    @property
    def unresolved(self) -> bool:
        return bool(
            self._ledger.unsettled_call_ids() or self._ledger.unknown_billing_call_ids()
        )

    def reserve(
        self,
        *,
        candidate_identity: str,
        stage: str,
        logical_call_id: str,
        model: str,
        retry: int = 0,
        worst_case_cost_usd: Decimal | str | float | None = None,
    ) -> str:
        if not candidate_identity.strip():
            raise FreshCampaignError("every reservation requires a candidate identity")
        if not stage.strip():
            raise FreshCampaignError("every reservation requires a stage")
        return self._ledger.reserve(
            logical_call_id=logical_call_id,
            stage=stage,
            model=model,
            retry=retry,
            worst_case_cost_usd=worst_case_cost_usd,
            candidate_identity=candidate_identity,
        )

    def settle(self, physical_call_id: str, **kwargs: object) -> None:
        self._ledger.settle(physical_call_id, **kwargs)  # type: ignore[arg-type]

    @property
    def accounting_ledger(self) -> RecoveryBudgetLedger:
        return self._ledger

    def candidate_events(self, identity: str) -> list[dict[str, object]]:
        """Return only durable accounting events linked to one candidate."""
        return [
            event
            for event in self.events()
            if event.get("candidate_identity") == identity
        ]

    def mark_unknown_billing(self, physical_call_id: str, *, error_type: str) -> None:
        self._ledger.mark_unknown_billing(physical_call_id, error_type=error_type)

    def events(self) -> list[dict[str, object]]:
        return self._ledger.events()

    def settled_calls(self) -> list[dict[str, object]]:
        return self._ledger.settled_calls()

    def reconcile(self) -> dict[str, object]:
        if self.unresolved:
            raise FreshCampaignError(
                "fresh campaign has unresolved or unknown-billing reservations"
            )
        if self.total_exact_cost > self.cap_usd:
            raise FreshCampaignError("fresh campaign exact spend exceeds cap")
        return {
            "run_id": self.run_id,
            "cap_usd": format(self.cap_usd, "f"),
            "exact_cost_usd": format(self.total_exact_cost, "f"),
            "physical_calls": len(self.settled_calls()),
            "status": "reconciled",
        }

    @contextmanager
    def call_context(self, *, candidate_identity: str, stage: str):
        with campaign_call_context(
            self._ledger, candidate_identity=candidate_identity, stage=stage
        ):
            yield


@dataclass
class StageEvidence:
    """Append-only Stage 0 through Stage 5 invocation evidence."""

    path: Path
    markers: list[str] = field(default_factory=list)
    sink: Callable[[str], None] | None = None

    def mark(self, stage: int, status: str = "started") -> None:
        if stage != len(self.markers):
            raise FreshCampaignError(
                f"stage markers out of order: expected {len(self.markers)}, got {stage}"
            )
        marker = f"Stage {stage}: {status}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(marker + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.markers.append(marker)
        if self.sink is not None:
            self.sink(marker)

    def complete(self, exit_code: int) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"exit_status: {int(exit_code)}\n")
            handle.flush()
            os.fsync(handle.fileno())
        if self.sink is not None:
            self.sink(f"exit_status: {int(exit_code)}")


@dataclass(frozen=True)
class DockerResource:
    name: str
    identity: str
    status: str
    started_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "identity": self.identity,
            "status": self.status,
            "started_at": self.started_at,
        }


@dataclass(frozen=True)
class DockerSnapshot:
    protected: tuple[DockerResource, ...] = ()
    mission_owned: tuple[DockerResource, ...] = ()
    dangling_images: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "protected": [resource.to_dict() for resource in self.protected],
            "mission_owned": [resource.to_dict() for resource in self.mission_owned],
            "dangling_images": list(self.dangling_images),
        }


def _parse_docker_rows(raw: str) -> tuple[DockerResource, ...]:
    rows: list[DockerResource] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FreshCampaignError("docker snapshot returned malformed JSON") from exc
        if not isinstance(value, dict):
            raise FreshCampaignError("docker snapshot row is not an object")
        name = str(value.get("Names", "")).lstrip("/")
        if name:
            rows.append(
                DockerResource(
                    name=name,
                    identity=str(value.get("ID", "")),
                    status=str(value.get("Status", "")),
                    started_at=str(value.get("CreatedAt", "")),
                )
            )
    return tuple(sorted(rows, key=lambda item: item.name))


class DockerEvidenceCollector:
    """Read-only paired Docker evidence collector with protected equality."""

    def __init__(
        self,
        run_id: str,
        *,
        command_runner: Callable[[Sequence[str]], str] | None = None,
    ) -> None:
        self.run_id = run_id
        self._run = command_runner or self._command

    @staticmethod
    def _command(command: Sequence[str]) -> str:
        result = subprocess.run(
            list(command), capture_output=True, text=True, check=False, timeout=30
        )
        if result.returncode != 0:
            raise FreshCampaignError("docker snapshot command failed")
        return result.stdout

    def snapshot(self) -> DockerSnapshot:
        resources = _parse_docker_rows(
            self._run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--format",
                    "{{json .}}",
                ]
            )
        )
        prefix = f"swe-forge-fresh-{self.run_id}"
        protected = tuple(
            resource
            for resource in resources
            if (
                resource.name in PROTECTED_DOCKER_RESOURCES
                or resource.name.startswith("challenge-prism-")
            )
        )
        owned = tuple(
            resource
            for resource in resources
            if resource.name == prefix or resource.name.startswith(f"{prefix}-")
        )
        images = tuple(
            line.strip()
            for line in self._run(
                ["docker", "images", "-q", "--filter", "dangling=true"]
            ).splitlines()
            if line.strip()
        )
        return DockerSnapshot(protected, owned, images)

    def induced_failure_teardown(self, command: Sequence[str]) -> dict[str, object]:
        """Run one intentionally failing throwaway command and prove cleanup."""
        before = self.snapshot()
        name = f"swe-forge-fresh-{self.run_id}-failure"
        result = subprocess.run(
            ["docker", "run", "--rm", "--name", name, "alpine:3.20", *command],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        after = self.snapshot()
        comparison = self.compare(before, after)
        if result.returncode == 0:
            raise FreshCampaignError("induced Docker failure unexpectedly passed")
        return {
            "exit_code": result.returncode,
            "container_name": name,
            "teardown": comparison["mission_owned_teardown"],
        }

    @staticmethod
    def compare(before: DockerSnapshot, after: DockerSnapshot) -> dict[str, object]:
        if before.protected != after.protected:
            raise FreshCampaignError(
                "protected Docker resources changed during fresh campaign"
            )
        before_owned = {item.name for item in before.mission_owned}
        after_owned = {item.name for item in after.mission_owned}
        if after_owned:
            raise FreshCampaignError(
                "fresh campaign left mission-owned Docker resources"
            )
        return {
            "protected_equal": True,
            "mission_owned_before": sorted(before_owned),
            "mission_owned_after": sorted(after_owned),
            "mission_owned_teardown": not after_owned,
            "new_dangling_images": sorted(
                set(after.dangling_images) - set(before.dangling_images)
            ),
        }


@dataclass
class FreshCampaignConfig:
    plans: list[CandidatePlan]
    out_dir: Path
    ledger_path: Path
    authority_path: Path
    evidence_path: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    worst_case_cost_usd: Decimal = Decimal("1.00")
    cap_usd: Decimal = CAP_USD
    band_config: BandFilterConfig = DEFAULT_BAND_FILTER
    frontier_threshold: float = 0.5
    historical_ledger: Path | None = None
    historical_dispositions: Path | None = None
    historical_identities: tuple[str, ...] = ()
    terminal_identities: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.cap_usd != CAP_USD:
            raise FreshCampaignError("fresh campaign cap must remain exactly $50")
        if self.band_config.band_high != 0.5:
            raise FreshCampaignError("fresh campaign band_high must remain 0.5")
        if self.band_config.discrimination_threshold < 1.0:
            raise FreshCampaignError(
                "fresh campaign discrimination threshold must be at least 1.0"
            )
        if self.frontier_threshold != 0.5:
            raise FreshCampaignError(
                "fresh campaign frontier threshold must remain 0.5"
            )
        if (
            self.historical_dispositions is not None
            and not self.historical_dispositions.exists()
        ):
            raise FreshCampaignError("historical dispositions file is unavailable")


@dataclass
class FreshCampaignResult:
    status: str
    reason: str
    run_id: str
    kept_task_id: str = ""
    dispositions: list[dict[str, object]] = field(default_factory=list)
    stage_markers: tuple[str, ...] = ()
    ledger: dict[str, object] = field(default_factory=dict)
    publication_generation: str = ""
    docker_evidence: dict[str, object] = field(default_factory=dict)
    publication_expected_generation: str = ""
    exit_status: int | None = None

    @property
    def published(self) -> bool:
        return self.status == "kept"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "run_id": self.run_id,
            "kept_task_id": self.kept_task_id,
            "dispositions": self.dispositions,
            "stage_markers": list(self.stage_markers),
            "ledger": self.ledger,
            "publication_generation": self.publication_generation,
            "docker_evidence": self.docker_evidence,
            "publication_expected_generation": self.publication_expected_generation,
            "exit_status": self.exit_status,
        }


class CampaignPublisher(Protocol):
    def __call__(
        self, request: ExportRequest | None, *, result: FreshCampaignResult
    ) -> str: ...


class GoldProver(Protocol):
    def __call__(self, request: ExportRequest) -> bool: ...


def fresh_boltons_plans(
    plans: Iterable[CandidatePlan],
    authority: FreshCandidateAuthority,
) -> list[CandidatePlan]:
    """Order only fresh boltons hard-rung amplifier candidates first."""
    selected: list[CandidatePlan] = []
    for plan in plans:
        if plan.repo.repo_id != "mahmoud/boltons":
            continue
        if plan.generator not in {"bug_combination", "multi_file"}:
            continue
        minimum = plan.params.get("min_symbol_lines", 0)
        if not isinstance(minimum, int) or minimum < 20:
            continue
        if (
            candidate_identity(plan) in authority.historical_identities
            or candidate_replay_key(plan) in authority.historical_identities
        ):
            continue
        selected.append(plan)
    return selected


def _default_publisher(out_dir: Path, *, expected_generation: str | None = None):
    def publish(request: ExportRequest | None, *, result: FreshCampaignResult) -> str:
        task: ForgeTask | None = None
        if request is not None:
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
            )
        entries = [PublicationEntry(index=0, task=task)] if task is not None else []

        def workspace_writer(current: ForgeTask, tasks_dir: Path) -> TaskExportResult:
            if request is None:
                raise PublicationError("tombstone cannot write a task")
            return _write_staged_workspace(
                current,
                tasks_dir / current.task_id,
                broken_tree=request.broken_tree,
            )

        def metadata_writer(stage: Path, accepted: Sequence[PublicationEntry]) -> None:
            certification = {
                "state": "keep" if task is not None else "tombstone",
                "passed": task is not None,
                "task_ids": [entry.task.task_id for entry in accepted],
                "run_id": result.run_id,
                "reason": result.reason,
                "campaign_id": CAMPAIGN_ID,
            }
            (stage / "certification.json").write_text(
                json.dumps(certification, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (stage / "fresh-campaign-result.json").write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        generation, _ = publish_generation(
            out_dir,
            entries,
            workspace_writer=workspace_writer,
            dataset_writer=export_dataset,
            overwrite=True,
            expected_current_generation_id=expected_generation,
            metadata_writer=metadata_writer,
            extra_facade_artifacts=("certification.json", "fresh-campaign-result.json"),
        )
        return generation.generation_id

    return publish


async def run_fresh_campaign(
    config: FreshCampaignConfig,
    *,
    processor: CandidateProcessor,
    gold_prover: GoldProver,
    publisher: CampaignPublisher | None = None,
    docker_evidence: DockerEvidenceCollector | None = None,
) -> FreshCampaignResult:
    """Run a sequential, reserve-safe campaign and stop on the first gold keep."""
    config.validate()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.evidence_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = FreshCampaignLedger(
        config.ledger_path,
        run_id=config.run_id,
        worst_case_cost_usd=config.worst_case_cost_usd,
        cap_usd=config.cap_usd,
        historical_ledger=config.historical_ledger,
    )
    authority = FreshCandidateAuthority(
        config.authority_path,
        historical_identities=(
            *config.historical_identities,
            *historical_candidate_identities(
                dispositions_path=config.historical_dispositions
            ),
        ),
        terminal_identities=config.terminal_identities,
    )
    unfinished = authority.recover_unfinished()
    if unfinished:
        raise FreshCampaignError(
            "fresh campaign has unfinished claimed identities: "
            + ", ".join(sorted(unfinished))
        )
    recorder = StageEvidence(config.evidence_path)
    before = docker_evidence.snapshot() if docker_evidence else None
    if before is not None and before.mission_owned:
        raise FreshCampaignError(
            "fresh campaign run prefix already owns Docker resources"
        )
    result = FreshCampaignResult("running", "", config.run_id)
    if before is not None:
        result.docker_evidence["before"] = before.to_dict()
    current = load_published_generation(config.out_dir)
    expected_generation = current.generation_id if current is not None else ""
    result.publication_expected_generation = expected_generation
    active_publisher = publisher or _default_publisher(
        config.out_dir, expected_generation=expected_generation
    )
    pending_request: ExportRequest | None = None
    pending_identity = ""
    try:
        if docker_evidence is not None:
            result.docker_evidence["induced_failure"] = (
                docker_evidence.induced_failure_teardown(("sh", "-c", "exit 97"))
            )
        with docker_name_prefix(f"swe-forge-fresh-{config.run_id}"):
            recorder.mark(0)
            recorder.mark(1)
            recorder.mark(2)
            recorder.mark(3)
            recorder.mark(4)
            for plan in fresh_boltons_plans(config.plans, authority):
                if ledger.remaining_cap < config.worst_case_cost_usd:
                    result.status = "cap_exhausted"
                    result.reason = "no next request fits within the remaining $50 cap"
                    break
                claim = authority.claim(
                    plan,
                    reason="fresh unprocessed boltons hard-rung candidate",
                )
                identity = claim.identity
                with ledger.call_context(
                    candidate_identity=identity, stage="fresh-campaign.stage-1-4"
                ):
                    artifacts = await processor.process(
                        plan, config.out_dir / f".{identity}"
                    )
                if ledger.unresolved:
                    raise FreshCampaignError(
                        "provider billing is unresolved; publication is forbidden"
                    )
                request = _keep_export_request(artifacts)
                disposition: dict[str, object] = {
                    "identity": identity,
                    "fresh_reason": claim.reason,
                    "plan": plan.to_dict(),
                    "stage": "processed",
                }
                if request is None:
                    disposition["stage"] = "dropped"
                    disposition["reason"] = (
                        artifacts.failure_reason or "not oracle keep"
                    )
                    result.dispositions.append(disposition)
                    authority.terminalize(
                        identity, status="dropped", reason=str(disposition["reason"])
                    )
                    continue
                if not gold_prover(request):
                    disposition["stage"] = "gold_failed"
                    disposition["reason"] = "gold proof did not pass"
                    result.dispositions.append(disposition)
                    authority.terminalize(
                        identity,
                        status="gold_failed",
                        reason="gold proof did not pass",
                    )
                    continue
                calibration = artifacts.calibration_report
                if calibration is None:
                    raise FreshCampaignError(
                        "candidate has no calibration report before publication"
                    )
                band_decision = calibration.details.get("band_filter")
                band_high = (
                    band_decision.get("band_high")
                    if isinstance(band_decision, dict)
                    else None
                )
                if band_high != 0.5:
                    raise FreshCampaignError("candidate changed band_high from 0.5")
                if calibration.irt_discrimination < 1.0:
                    raise FreshCampaignError(
                        "candidate discrimination is below the unchanged threshold"
                    )
                try:
                    if artifacts.oracle_report is None:
                        raise FreshCampaignError("candidate has no oracle report")
                    reconciliation = reconcile_recovery_reports(
                        ledger.accounting_ledger,
                        artifacts.oracle_report,
                        artifacts.calibration_report,
                        require_complete=False,
                        candidate_identity=identity,
                    )
                except (RecoveryAccountingError, AttributeError) as exc:
                    raise FreshCampaignError(
                        "candidate recovery evidence did not reconcile before publication"
                    ) from exc
                result.ledger = reconciliation
                result.status = "kept"
                result.reason = "first newly certified oracle/calibration/gold keep"
                pending_request = request
                pending_identity = identity
                result.dispositions.append({**disposition, "stage": "kept"})
                break
            else:
                result.status = "cap_exhausted"
                result.reason = "candidate supply exhausted before a certified keep"
        result.ledger = ledger.reconcile()
        if docker_evidence and before is not None:
            after = docker_evidence.snapshot()
            result.docker_evidence["after"] = after.to_dict()
            result.docker_evidence["comparison"] = DockerEvidenceCollector.compare(
                before, after
            )
        recorder.mark(5)
        result.exit_status = 0
        result.stage_markers = tuple(recorder.markers)
        if pending_request is not None:
            generation = active_publisher(pending_request, result=result)
            result.kept_task_id = (
                pending_request.task_id or pending_request._fallback_id()
            )
            result.publication_generation = generation
            authority.terminalize(
                pending_identity,
                status="kept",
                reason="first newly certified oracle/calibration/gold keep",
            )
        else:
            result.publication_generation = active_publisher(None, result=result)
        recorder.complete(0)
        return result
    except BaseException as exc:
        result.status = "blocked"
        result.reason = "fresh campaign stopped fail-closed: " + type(exc).__name__
        if not isinstance(exc, asyncio.CancelledError):
            try:
                active_publisher(None, result=result)
            except Exception:
                pass
        try:
            ledger.reconcile()
        except RecoveryAccountingError:
            pass
        result.exit_status = 1
        recorder.complete(1)
        if docker_evidence and before is not None:
            after = docker_evidence.snapshot()
            result.docker_evidence["after"] = after.to_dict()
            result.docker_evidence["comparison"] = DockerEvidenceCollector.compare(
                before, after
            )
        raise


__all__ = [
    "CAMPAIGN_ID",
    "CAP_USD",
    "candidate_replay_key",
    "DockerEvidenceCollector",
    "DockerResource",
    "DockerSnapshot",
    "FreshCampaignConfig",
    "FreshCampaignError",
    "FreshCampaignLedger",
    "FreshCampaignResult",
    "FreshCandidateAuthority",
    "FreshIdentityClaim",
    "StageEvidence",
    "candidate_identity",
    "fresh_boltons_plans",
    "historical_candidate_identities",
    "run_fresh_campaign",
]
