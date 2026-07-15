"""Live micro-keep pipeline (VAL-CROSS-002).

Ordered stages with retained evidence:

  sources → envbuild → produce → oracle → panel → export

Prefer synthetic_grounded multi-file seeds with pinned SHAs. Preserve all gate
thresholds. Fail closed if a keep export lacks panel hardness fields.

Micro spend guard: escalate with funnel numbers if spend exceeds the micro
cap (default $80) without a certified keep.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from swe_factory.accounting import AccountingError, BudgetLedger, default_ledger_path
from swe_factory.config import FactorySettings, load_settings
from swe_factory.export.workspace import ExportError, write_export_bundle
from swe_factory.panel.band import hardness_dict_from_decision
from swe_factory.panel.runner import (
    DEFAULT_PANEL_K,
    DEFAULT_ROLLOUT_RESERVE_USD,
    REQUIRED_PANEL_MODELS,
    PanelRunResult,
    run_panel,
)
from swe_factory.panel.score_solver import (
    local_pytest_soft_solver,
    oracle_runner_soft_solver,
)
from swe_factory.producers.synth import (
    MUTATION_FUNCTION_REMOVAL,
    MUTATION_MULTI_FAULT,
    MutationKind,
    SynthCandidate,
    SynthError,
    SynthProducer,
)
from swe_factory.schema import EnvironmentMeta, PanelHardness, SourceTrack, TaskRecord
from swe_factory.sources.allowlist import SeedRepo, get_seed
from swe_factory.sources.clone import (
    CloneError,
    PinnedCheckout,
    ensure_pinned_checkout,
    is_immutable_sha,
)

StageName = Literal[
    "sources",
    "envbuild",
    "produce",
    "oracle",
    "panel",
    "export",
    "escalate",
    "done",
]

DEFAULT_MICRO_CAP_USD = Decimal("80")
DEFAULT_PANEL_RESERVE_USD = DEFAULT_ROLLOUT_RESERVE_USD  # 1.50


class MicroKeepError(RuntimeError):
    """Unrecoverable micro-keep infrastructure error."""


@dataclass
class StageEvent:
    stage: StageName
    status: str  # ok | skip | fail | drop | keep
    detail: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "detail": dict(self.detail),
            "at": self.at,
        }


@dataclass
class MicroKeepResult:
    """Terminal result of one micro-keep attempt (keep or honest escalate)."""

    ok: bool
    is_keep: bool
    escalated: bool
    task: TaskRecord | None
    instance_id: str | None
    source_track: str | None
    export_dir: Path | None
    stages: list[StageEvent]
    funnel: dict[str, Any]
    spend_exact_usd: Decimal
    spend_reserved_usd: Decimal
    micro_cap_usd: Decimal
    panel: dict[str, Any] | None
    reason: str
    stage_log_path: Path | None = None
    ledger_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "is_keep": self.is_keep,
            "escalated": self.escalated,
            "instance_id": self.instance_id,
            "source_track": self.source_track,
            "export_dir": str(self.export_dir) if self.export_dir else None,
            "stages": [s.to_dict() for s in self.stages],
            "funnel": dict(self.funnel),
            "spend_exact_usd": format(self.spend_exact_usd, "f"),
            "spend_reserved_usd": format(self.spend_reserved_usd, "f"),
            "micro_cap_usd": format(self.micro_cap_usd, "f"),
            "panel": self.panel,
            "reason": self.reason,
            "stage_log_path": str(self.stage_log_path) if self.stage_log_path else None,
            "ledger_path": str(self.ledger_path) if self.ledger_path else None,
            "task_source_track": (
                self.task.source_track.value
                if self.task is not None and hasattr(self.task.source_track, "value")
                else str(self.task.source_track)
                if self.task is not None
                else None
            ),
            "has_panel_hardness": bool(
                self.task is not None
                and self.task.panel is not None
                and self.task.panel.pass_at_k is not None
            ),
        }


def require_panel_hardness(task: TaskRecord) -> None:
    """Fail closed: certified keep export must include panel hardness fields."""
    if task.panel is None:
        raise ExportError(
            f"keep export fail-closed: task {task.instance_id!r} lacks panel hardness fields"
        )
    p = task.panel
    if p.pass_at_k is None:
        raise ExportError(
            f"keep export fail-closed: task {task.instance_id!r} panel.pass_at_k missing"
        )
    if p.discrimination is None:
        raise ExportError(
            f"keep export fail-closed: task {task.instance_id!r} panel.discrimination missing"
        )
    # At least one model rate must be present for required panel set.
    if p.grok_4_5 is None and p.kimi_k2_6 is None and p.opus_4_8 is None:
        raise ExportError(
            f"keep export fail-closed: task {task.instance_id!r} missing panel model rates"
        )
    # Force-label source_track
    track = task.source_track
    value = track.value if hasattr(track, "value") else str(track)
    if value not in {SourceTrack.REAL_PR.value, SourceTrack.SYNTHETIC_GROUNDED.value}:
        raise ExportError(
            f"keep export fail-closed: invalid source_track {value!r} on {task.instance_id!r}"
        )


def _write_stage_log(path: Path, stages: Sequence[StageEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(s.to_dict(), sort_keys=True) for s in stages) + "\n",
        encoding="utf-8",
    )


def _ledger_snapshot(ledger: BudgetLedger) -> tuple[Decimal, Decimal]:
    summary = ledger.summary()
    return summary.settled_exact_usd, summary.open_reserved_usd


def _estimate_panel_budget(k: int, reserve: Decimal) -> Decimal:
    return reserve * Decimal(len(REQUIRED_PANEL_MODELS) * k)


def _build_env_digest(
    seed: SeedRepo,
    checkout: PinnedCheckout,
    *,
    dual: bool = False,
    skip_docker: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Envbuild stage: produce image digest or honest offline digest.

    When skip_docker, return a deterministic pending digest so fixture path
    still records env stage evidence (oracle may still use Docker later).
    """
    if skip_docker:
        dig = f"sha256:micro_offline_{seed.seed_id}_{checkout.base_commit[:12]}"
        return dig, {
            "mode": "offline_skip_docker",
            "image_digest": dig,
            "base_commit": checkout.base_commit,
            "repo": seed.repo,
        }

    from swe_factory.envbuild.builder import (
        DockerCLI,
        EnvBuilder,
        dual_build,
        remove_leftover_sdf_containers,
    )
    from swe_factory.envbuild.models import EnvRecipe

    recipe = EnvRecipe(
        repo_id=seed.repo.replace("/", "__"),
        base_commit=checkout.base_commit,
        language=seed.language,
        base_image=seed.base_image or "python:3.12-slim",
        install_commands=list(seed.install_commands)
        or (["pip install -q pytest"] if seed.language == "python" else []),
        baseline_test_command=seed.baseline_test_command
        or ("python -m pytest -q" if seed.language == "python" else "true"),
        local_path=str(checkout.path),
    )
    docker = DockerCLI()
    try:
        if dual:
            first, second, verified = dual_build(recipe, docker=docker)
            if not (first.success and second.success and first.env_image is not None):
                reason = first.reason or second.reason or "dual_build failed"
                raise MicroKeepError(f"envbuild dual failed: {reason}")
            if not verified:
                raise MicroKeepError("envbuild dual failed: recipe not usable / digests diverge")
            result = first
        else:
            result = EnvBuilder(docker=docker).build(recipe)
    finally:
        remove_leftover_sdf_containers(docker)

    if not result.success or result.env_image is None:
        raise MicroKeepError(
            f"envbuild failed: {result.failure_kind or 'unknown'} — {result.reason}"
        )
    digest = result.env_image.image_digest
    meta = {
        "mode": "docker_dual" if dual else "docker",
        "image_digest": digest,
        "image_tag": result.env_image.image_tag,
        "base_commit": checkout.base_commit,
        "repo": seed.repo,
        "recipe_usable": True,
        "dual_build_verified": bool(getattr(result.env_image, "dual_build_verified", dual)),
    }
    return digest, meta


def _remap_f2p_for_modules(
    green_root: Path,
    *,
    gold_files: Sequence[str],
    language: str,
    fallback_f2p: Sequence[str],
    fallback_p2p: Sequence[str],
    broken_root: Path | None = None,
    install_commands: Sequence[str] | None = None,
) -> tuple[list[str], list[str]] | None:
    """Map mutated module files to colocated unit test files when present.

    When ``broken_root`` is provided, only keep test modules that actually fail
    on the broken tree (G1 requires every F2P to fail). Prefer combining all
    failing module tests into one multi-select pytest command so partial suite
    failures still fail closed as a single motor.

    Python: host pytest on package-relative test paths.
    JS: npm/node test scripts when seed has npm-based baselines.
    Go: `go test` package scopes closed around mutated files.
    """
    import subprocess

    lang = language.strip().lower()
    if lang in {"js", "ts", "typescript"}:
        lang = "javascript"

    # ---------------- python ----------------
    if lang == "python":
        test_paths: list[str] = []
        for rel in gold_files:
            stem = Path(rel).stem
            # boltons style: boltons/foo.py -> tests/test_foo.py
            candidates = [
                Path("tests") / f"test_{stem}.py",
                Path("test") / f"test_{stem}.py",
                Path("tests") / f"test_{stem.replace('_', '')}.py",
                Path(rel).with_name(f"test_{stem}.py"),
            ]
            # cachetools: src/cachetools/lru.py → tests/test_lru.py
            if "/" in rel:
                candidates.append(Path("tests") / f"test_{stem}.py")
            for cand in candidates:
                if (green_root / cand).is_file():
                    test_paths.append(cand.as_posix())
                    break
        if not test_paths:
            return None

        failing: list[str] = []
        probe_root = Path(broken_root) if broken_root is not None else green_root
        # Ensure package installed for host probe when needed
        if install_commands and probe_root.is_dir():
            for cmd in install_commands:
                subprocess.run(
                    cmd,
                    cwd=str(probe_root),
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )
        if broken_root is not None and Path(broken_root).is_dir():
            for tpath in sorted(set(test_paths)):
                proc = subprocess.run(
                    ["python", "-m", "pytest", tpath, "-q"],
                    cwd=str(broken_root),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if proc.returncode != 0:
                    failing.append(tpath)
        else:
            failing = sorted(set(test_paths))

        if not failing:
            # Host remount / import path can miss isolate-module fails. Fall back
            # to combined module test command so Docker oracle can still G1/G2.
            if test_paths:
                f2p = [f"pytest {' '.join(sorted(set(test_paths)))} -q"]
                p2p = list(fallback_p2p) if fallback_p2p else ["pytest -q"]
                p2p = [c for c in p2p if c not in f2p]
                return f2p, p2p
            return None

        f2p = [f"pytest {' '.join(failing)} -q"]
        p2p = list(fallback_p2p) if fallback_p2p else ["pytest -q"]
        p2p = [c for c in p2p if c not in f2p]
        return f2p, p2p

    # ---------------- go ----------------
    if lang == "go":
        # Map mutated .go files → package dirs with tests.
        pkgs: list[str] = []
        for rel in gold_files:
            p = Path(rel)
            if p.suffix != ".go":
                continue
            pkg_dir = "." if str(p.parent) in {".", ""} else "./" + p.parent.as_posix()
            # Prefer directory that has *_test.go nearby.
            check_dir = green_root if pkg_dir == "." else green_root / p.parent
            has_tests = any(check_dir.glob("*_test.go")) if check_dir.is_dir() else False
            if has_tests and pkg_dir not in pkgs:
                pkgs.append(pkg_dir)
        if not pkgs:
            # fall back: entire module
            if any(green_root.rglob("*_test.go")):
                pkgs = ["./..."]
            else:
                return None
        f2p_cmds = [f"go test {pkg}" for pkg in pkgs]
        # Probe on broken tree: only keep motors that fail
        if broken_root is not None and Path(broken_root).is_dir():
            failing_cmds: list[str] = []
            for cmd in f2p_cmds:
                proc = subprocess.run(
                    cmd,
                    cwd=str(broken_root),
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )
                if proc.returncode != 0:
                    failing_cmds.append(cmd)
            if not failing_cmds:
                return None
            f2p_cmds = failing_cmds
        p2p = list(fallback_p2p) if fallback_p2p else ["go test ./..."]
        p2p = [c for c in p2p if c not in f2p_cmds]
        return f2p_cmds, p2p

    # ---------------- javascript ----------------
    if lang == "javascript":
        # Prefer package-level npm test as F2P when mutated sources make it fail;
        # otherwise skip (no motor).
        baseline = "npm test --silent"
        if (green_root / "package.json").is_file():
            try:
                pkg = json.loads((green_root / "package.json").read_text(encoding="utf-8"))
                scripts = pkg.get("scripts") or {}
                if "tests-only" in scripts:
                    baseline = "npm run tests-only --silent"
                elif "test" in scripts:
                    baseline = "npm test --silent"
            except Exception:  # noqa: BLE001
                pass
        f2p_cmds = [baseline]
        if broken_root is not None and Path(broken_root).is_dir():
            # Ensure node_modules somewhat available; best-effort.
            if install_commands:
                for cmd in install_commands:
                    subprocess.run(
                        cmd,
                        cwd=str(broken_root),
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        check=False,
                    )
            elif not (Path(broken_root) / "node_modules").is_dir():
                subprocess.run(
                    "npm install --no-audit --no-fund --legacy-peer-deps",
                    cwd=str(broken_root),
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
            proc = subprocess.run(
                baseline,
                cwd=str(broken_root),
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if proc.returncode == 0:
                return None
        p2p = list(fallback_p2p) if fallback_p2p else []
        return f2p_cmds, p2p

    return None


def _seed_with_pin(seed: SeedRepo, pin: str) -> SeedRepo:
    """Return a SeedRepo copy with immutable SHA pin."""
    return SeedRepo(
        seed_id=seed.seed_id,
        language=seed.language,
        repo=seed.repo,
        base_commit=pin,
        license=seed.license,
        description=seed.description,
        local_fixture=seed.local_fixture,
        source_globs=seed.source_globs,
        f2p_commands=seed.f2p_commands,
        p2p_commands=seed.p2p_commands,
        install_commands=seed.install_commands,
        baseline_test_command=seed.baseline_test_command,
        base_image=seed.base_image,
        modular=seed.modular,
        notes=seed.notes,
    )


def run_micro_keep(
    *,
    out_dir: Path | str,
    seed_id: str = "fixture_tiny_green",
    mutation: MutationKind | str = MUTATION_MULTI_FAULT,
    settings: FactorySettings | None = None,
    ledger: BudgetLedger | None = None,
    micro_cap_usd: Decimal = DEFAULT_MICRO_CAP_USD,
    panel_k: int = DEFAULT_PANEL_K,
    panel_reserve_usd: Decimal = DEFAULT_PANEL_RESERVE_USD,
    use_docker_oracle: bool = True,
    use_docker_envbuild: bool = False,
    dual_envbuild: bool = False,
    soft_backend: Literal["local", "oracle", "never"] = "local",
    live_panel: bool = True,
    client: Any | None = None,
    soft_solver: Callable[..., bool] | None = None,
    require_immutable_sha: bool = True,
    stage_callback: Callable[[StageEvent], None] | None = None,
    diversification_index: int = 0,
    prefer_stems: Sequence[str] | None = None,
    exclude_stems: Sequence[str] | None = None,
) -> MicroKeepResult:
    """Run sources→env→produce→oracle→panel→export for one candidate.

    Offline tests inject ``live_panel=False`` + scripted client/soft_solver, and
    can use FakeOracle via ``use_docker_oracle=False``.
    """
    settings = settings or load_settings()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    stage_log = out / "stage_log.jsonl"
    stages: list[StageEvent] = []
    funnel: dict[str, Any] = {
        "candidates_started": 1,
        "sources_ok": 0,
        "envbuild_ok": 0,
        "produce_ok": 0,
        "oracle_pass": 0,
        "oracle_fail": 0,
        "panel_run": 0,
        "panel_keep": 0,
        "panel_drop": 0,
        "export_ok": 0,
        "escalated": 0,
    }

    def emit(stage: StageName, status: str, **detail: Any) -> StageEvent:
        ev = StageEvent(stage=stage, status=status, detail=detail)
        stages.append(ev)
        _write_stage_log(stage_log, stages)
        if stage_callback is not None:
            stage_callback(ev)
        return ev

    # Durable mission ledger by default so spend accumulates across runs.
    if ledger is None:
        ledger_path = default_ledger_path(Path.cwd())
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = BudgetLedger(
            ledger_path,
            cap_usd=settings.budget_usd,
            worst_case_cost_usd=panel_reserve_usd,
            run_id="micro-keep",
        )
    start_exact, _ = _ledger_snapshot(ledger)
    spend_before = start_exact

    def spent_this_run() -> Decimal:
        exact, reserved = _ledger_snapshot(ledger)
        # Count only the delta settled since start + open reserved for honesty.
        return max(Decimal("0"), exact - spend_before) + reserved

    def terminal(
        *,
        ok: bool,
        is_keep: bool,
        escalated: bool,
        task: TaskRecord | None,
        export_dir: Path | None,
        reason: str,
        panel_dict: dict[str, Any] | None = None,
    ) -> MicroKeepResult:
        exact, reserved = _ledger_snapshot(ledger)
        with contextlib.suppress(AccountingError):
            ledger.write_summary_json()
        instance_id = task.instance_id if task is not None else None
        track = None
        if task is not None:
            t = task.source_track
            track = t.value if hasattr(t, "value") else str(t)
        return MicroKeepResult(
            ok=ok,
            is_keep=is_keep,
            escalated=escalated,
            task=task,
            instance_id=instance_id,
            source_track=track,
            export_dir=export_dir,
            stages=list(stages),
            funnel=dict(funnel),
            spend_exact_usd=max(Decimal("0"), exact - spend_before),
            spend_reserved_usd=reserved,
            micro_cap_usd=micro_cap_usd,
            panel=panel_dict,
            reason=reason,
            stage_log_path=stage_log,
            ledger_path=ledger.path,
        )

    # ------------------------------------------------------------------ sources
    try:
        seed = get_seed(seed_id)
    except KeyError as exc:
        emit("sources", "fail", error=str(exc))
        return terminal(
            ok=False,
            is_keep=False,
            escalated=False,
            task=None,
            export_dir=None,
            reason=f"sources: unknown seed_id {seed_id!r}",
        )

    try:
        checkout = ensure_pinned_checkout(seed, dest_root=work / "checkout", prefer_local=True)
    except CloneError as exc:
        emit("sources", "fail", error=str(exc), seed_id=seed_id)
        return terminal(
            ok=False,
            is_keep=False,
            escalated=False,
            task=None,
            export_dir=None,
            reason=f"sources: clone/pin failed: {exc}",
        )

    if require_immutable_sha and not is_immutable_sha(checkout.base_commit):
        emit(
            "sources",
            "fail",
            error="non-immutable base_commit",
            base_commit=checkout.base_commit,
        )
        return terminal(
            ok=False,
            is_keep=False,
            escalated=False,
            task=None,
            export_dir=None,
            reason=f"sources: non-immutable base_commit {checkout.base_commit!r}",
        )

    pinned_seed = _seed_with_pin(seed, checkout.base_commit)
    funnel["sources_ok"] = 1
    emit(
        "sources",
        "ok",
        seed_id=seed.seed_id,
        repo=seed.repo,
        base_commit=checkout.base_commit,
        path=str(checkout.path),
        source=checkout.source,
        language=seed.language,
    )

    # ------------------------------------------------------------------ envbuild
    try:
        image_digest, env_meta = _build_env_digest(
            pinned_seed,
            checkout,
            dual=dual_envbuild,
            skip_docker=not use_docker_envbuild,
        )
    except (MicroKeepError, Exception) as exc:  # noqa: BLE001
        emit("envbuild", "fail", error=str(exc))
        return terminal(
            ok=False,
            is_keep=False,
            escalated=False,
            task=None,
            export_dir=None,
            reason=f"envbuild: {exc}",
        )
    funnel["envbuild_ok"] = 1
    emit("envbuild", "ok", **env_meta)

    # ------------------------------------------------------------------ produce
    kind_raw = str(mutation).strip().lower().replace("-", "_")
    if kind_raw in {"multi_fault", "multifault", "multi"}:
        mutation_kind: MutationKind = MUTATION_MULTI_FAULT
    elif kind_raw in {"function_removal", "functionremoval", "removal", "remove"}:
        mutation_kind = MUTATION_FUNCTION_REMOVAL
    else:
        emit("produce", "fail", error=f"unknown mutation {mutation!r}")
        return terminal(
            ok=False,
            is_keep=False,
            escalated=False,
            task=None,
            export_dir=None,
            reason=f"produce: unknown mutation {mutation!r}",
        )

    producer = SynthProducer(
        work_root=work / "produce",
        image_digest=image_digest,
        min_files=2,
        keep_workspaces=True,
    )
    try:
        # stage produce only (oracle separate for ordered evidence)
        candidate = producer.produce(
            pinned_seed,
            mutation_kind=mutation_kind,
            base_path=checkout.path,
            run_stub_oracle=True,
            diversification_index=int(diversification_index),
            prefer_stems=prefer_stems,
            exclude_stems=exclude_stems,
        )
        # Remap F2P/P2P for real-repo clones to module-level suites when available.
        f2p = list(candidate.task.fail_to_pass)
        p2p = list(candidate.task.pass_to_pass)
        if checkout.source == "clone" and candidate.gold_files:
            remapped = _remap_f2p_for_modules(
                checkout.path,
                gold_files=list(candidate.gold_files),
                language=pinned_seed.language,
                fallback_f2p=f2p,
                fallback_p2p=p2p,
                broken_root=candidate.broken_workspace,
                install_commands=list(pinned_seed.install_commands),
            )
            if remapped is None:
                raise SynthError(
                    "produce: no motorized F2P for mutated modules "
                    f"{list(candidate.gold_files)}; re-seed mutation"
                )
            f2p, p2p = remapped
        # Force pin onto task (produce already uses seed.base_commit)
        task = candidate.task.model_copy(
            update={
                "base_commit": checkout.base_commit,
                "environment": EnvironmentMeta(image_digest=image_digest),
                "fail_to_pass": f2p,
                "pass_to_pass": p2p,
            }
        )
        candidate = SynthCandidate(
            task=task,
            broken_workspace=candidate.broken_workspace,
            green_workspace=candidate.green_workspace,
            mutation_kind=candidate.mutation_kind,
            targets=candidate.targets,
            gold_files=candidate.gold_files,
            inverse_meta=candidate.inverse_meta,
            gates=candidate.gates,
            provider_calls=candidate.provider_calls,
        )
    except SynthError as exc:
        emit("produce", "fail", error=str(exc))
        return terminal(
            ok=False,
            is_keep=False,
            escalated=False,
            task=None,
            export_dir=None,
            reason=f"produce: {exc}",
        )

    track_val = (
        candidate.task.source_track.value
        if hasattr(candidate.task.source_track, "value")
        else str(candidate.task.source_track)
    )
    if track_val != SourceTrack.SYNTHETIC_GROUNDED.value:
        emit("produce", "fail", error=f"expected synthetic_grounded, got {track_val}")
        return terminal(
            ok=False,
            is_keep=False,
            escalated=False,
            task=candidate.task,
            export_dir=None,
            reason=f"produce: wrong source_track {track_val}",
        )

    funnel["produce_ok"] = 1
    emit(
        "produce",
        "ok",
        instance_id=candidate.task.instance_id,
        source_track=track_val,
        mutation_kind=candidate.mutation_kind,
        gold_files=list(candidate.gold_files),
        multi_file=len(candidate.gold_files) >= 2,
        base_commit=candidate.task.base_commit,
    )

    # ------------------------------------------------------------------ oracle
    try:
        from swe_factory.oracle.docker_run import (
            FakeOracleRunner,
            OracleDockerRunner,
            ScriptedSuite,
        )
        from swe_factory.oracle.gates import append_gate_audit, run_certified_gates_for_task

        cleanup_containers: Callable[[], Any]

        if use_docker_oracle:
            from swe_factory.envbuild.builder import DockerCLI, remove_leftover_sdf_containers

            runner: Any = OracleDockerRunner(
                docker=DockerCLI(),
                base_image=pinned_seed.base_image or "python:3.12-slim",
                install_commands=list(pinned_seed.install_commands) or ["pip install -q pytest"],
                command_timeout=180.0,
            )

            def cleanup_containers() -> None:
                remove_leftover_sdf_containers()

        else:
            runner = FakeOracleRunner(
                broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
                gold_runs=[
                    ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
                    ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
                ],
                null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
            )

            def cleanup_containers() -> None:
                return None

        try:
            gate_result = run_certified_gates_for_task(
                candidate.task,
                workspace=candidate.broken_workspace,
                runner=runner,
                agent_workspace=None,
                require_multi_file=True,
                dual_runs=2,
                check_null_patch=True,
                check_leak=True,
            )
        finally:
            cleanup_containers()

        audit_path = out / "gate_audit.jsonl"
        append_gate_audit(
            audit_path,
            gate_result,
            candidate.task.instance_id,
            extra={
                "source_track": track_val,
                "stage": "oracle",
                "pipeline": "micro-keep",
            },
        )
        task = candidate.task.model_copy(
            update={
                "gate_proof": {
                    **(candidate.task.gate_proof or {}),
                    **gate_result.to_gate_proof(),
                    "inverse_meta": candidate.inverse_meta,
                }
            }
        )
        candidate = SynthCandidate(
            task=task,
            broken_workspace=candidate.broken_workspace,
            green_workspace=candidate.green_workspace,
            mutation_kind=candidate.mutation_kind,
            targets=candidate.targets,
            gold_files=candidate.gold_files,
            inverse_meta=candidate.inverse_meta,
            gates=gate_result,
            provider_calls=0,
        )
    except Exception as exc:  # noqa: BLE001
        emit("oracle", "fail", error=str(exc))
        return terminal(
            ok=False,
            is_keep=False,
            escalated=False,
            task=candidate.task,
            export_dir=None,
            reason=f"oracle: {exc}",
        )

    if not gate_result.passed:
        funnel["oracle_fail"] = 1
        emit(
            "oracle",
            "fail",
            passed=False,
            reason_codes=list(gate_result.reason_codes),
        )
        return terminal(
            ok=False,
            is_keep=False,
            escalated=False,
            task=candidate.task,
            export_dir=None,
            reason=f"oracle rejected: {gate_result.reason_codes}",
        )

    funnel["oracle_pass"] = 1
    emit(
        "oracle",
        "ok",
        passed=True,
        reason_codes=list(gate_result.reason_codes),
        instance_id=candidate.task.instance_id,
    )

    # micro cap guard before live panel reservations
    need = _estimate_panel_budget(panel_k, panel_reserve_usd) if live_panel else Decimal("0")
    if live_panel and spent_this_run() + need > micro_cap_usd:
        funnel["escalated"] = 1
        emit(
            "escalate",
            "fail",
            reason="micro_cap_before_panel",
            spent=format(spent_this_run(), "f"),
            need=format(need, "f"),
            micro_cap=format(micro_cap_usd, "f"),
        )
        return terminal(
            ok=True,  # honest escalate is a valid micro outcome
            is_keep=False,
            escalated=True,
            task=candidate.task,
            export_dir=None,
            reason=(
                f"escalate: remaining micro budget insufficient for panel "
                f"(spent={spent_this_run()}, need={need}, cap={micro_cap_usd})"
            ),
        )

    if live_panel and ledger.remaining_usd() < need:
        funnel["escalated"] = 1
        emit(
            "escalate",
            "fail",
            reason="global_cap_before_panel",
            remaining=format(ledger.remaining_usd(), "f"),
            need=format(need, "f"),
        )
        return terminal(
            ok=True,
            is_keep=False,
            escalated=True,
            task=candidate.task,
            export_dir=None,
            reason=(
                f"escalate: global ledger remaining {ledger.remaining_usd()} < panel need {need}"
            ),
        )

    # ------------------------------------------------------------------ panel
    sol = soft_solver
    panel_runner_backend: Any | None = None
    if sol is None:
        if soft_backend == "local":
            sol = local_pytest_soft_solver(
                broken_workspace=candidate.broken_workspace,
                fail_to_pass=list(candidate.task.fail_to_pass),
                pass_to_pass=list(candidate.task.pass_to_pass),
                install_commands=list(pinned_seed.install_commands),
            )
        elif soft_backend == "oracle":
            # Separate Fake/Docker runner dedicated to soft scoring.
            if use_docker_oracle:
                from swe_factory.envbuild.builder import DockerCLI
                from swe_factory.oracle.docker_run import OracleDockerRunner

                panel_runner_backend = OracleDockerRunner(
                    docker=DockerCLI(),
                    base_image=pinned_seed.base_image or "python:3.12-slim",
                    install_commands=list(pinned_seed.install_commands)
                    or ["pip install -q pytest"],
                    command_timeout=180.0,
                )
            else:
                from swe_factory.oracle.docker_run import FakeOracleRunner, ScriptedSuite

                # Default fake never resolves model patches → pass@k=0 solve-none.
                # Offline tests should inject soft_solver instead.
                panel_runner_backend = FakeOracleRunner(
                    broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
                    gold_runs=[ScriptedSuite(f2p_exits=[0], p2p_exits=[0])],
                    null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
                )
            sol = oracle_runner_soft_solver(
                task=candidate.task,
                broken_workspace=candidate.broken_workspace,
                runner=panel_runner_backend,
            )
        else:
            from swe_factory.panel.score_solver import never_solve_soft_solver

            sol = never_solve_soft_solver()

    chat_client = client
    if live_panel and chat_client is None:
        if not settings.has_api_key():
            emit("panel", "fail", error="OPENROUTER_API_KEY missing")
            return terminal(
                ok=False,
                is_keep=False,
                escalated=False,
                task=candidate.task,
                export_dir=None,
                reason="panel: OPENROUTER_API_KEY missing for live panel",
            )
        from swe_factory.openrouter import OpenRouterClient

        chat_client = OpenRouterClient.from_settings(settings)

    try:
        if live_panel and hasattr(chat_client, "__enter__"):
            enter = chat_client.__enter__()  # type: ignore[union-attr]
            chat_client = enter if enter is not None else chat_client

        panel_result: PanelRunResult = run_panel(
            task_id=candidate.task.instance_id,
            problem_statement=candidate.task.problem_statement,
            ledger=ledger,
            client=chat_client,
            models=list(REQUIRED_PANEL_MODELS),
            k=panel_k,
            stage="hardness-panel",
            soft_solver=sol,
            reserve_usd=panel_reserve_usd,
            max_tokens=4096 if live_panel else 64,
            allow_missing_cost_as_zero=not live_panel,
            # Slight nonzero temperature for k>1 diversity on frontier pair.
            temperature=0.2 if live_panel and panel_k > 1 else 0.0,
        )
    except Exception as exc:  # noqa: BLE001
        emit("panel", "fail", error=str(exc), spent=format(spent_this_run(), "f"))
        return terminal(
            ok=False,
            is_keep=False,
            escalated=spent_this_run() >= micro_cap_usd,
            task=candidate.task,
            export_dir=None,
            reason=f"panel: {exc}",
        )
    finally:
        if live_panel and client is None and chat_client is not None:
            # Close client we created.
            close = getattr(chat_client, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
            exit_fn = getattr(chat_client, "__exit__", None)
            if callable(exit_fn):
                with contextlib.suppress(Exception):
                    exit_fn(None, None, None)
        if panel_runner_backend is not None:
            with contextlib.suppress(Exception):
                panel_runner_backend.cleanup()
            if use_docker_oracle:
                from swe_factory.envbuild.builder import remove_leftover_sdf_containers

                remove_leftover_sdf_containers()

    funnel["panel_run"] = 1
    panel_dict = panel_result.to_dict()
    (out / "panel_report.json").write_text(
        json.dumps(panel_dict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not panel_result.is_keep:
        funnel["panel_drop"] = 1
        emit(
            "panel",
            "drop",
            verdict=panel_result.decision.verdict,
            rule=panel_result.decision.rule,
            pass_at_k=panel_result.decision.frontier_pass_at_k,
            discrimination=panel_result.decision.discrimination,
            cost_usd=format(panel_result.total_cost_usd, "f"),
            spent_run=format(spent_this_run(), "f"),
        )
        # Escalate with numbers when over micro cap or simply return drop funnel.
        escalated = spent_this_run() >= micro_cap_usd
        if escalated:
            funnel["escalated"] = 1
            emit(
                "escalate",
                "fail",
                reason="micro_cap_without_keep",
                spent=format(spent_this_run(), "f"),
                micro_cap=format(micro_cap_usd, "f"),
                funnel=dict(funnel),
            )
        return terminal(
            ok=True,  # honest non-keep outcome
            is_keep=False,
            escalated=escalated,
            task=candidate.task,
            export_dir=None,
            reason=(
                f"panel drop: rule={panel_result.decision.rule} "
                f"pass@k={panel_result.decision.frontier_pass_at_k:.4f} "
                f"disc={panel_result.decision.discrimination:.4f}"
                + ("; micro_cap exceeded without keep" if escalated else "")
            ),
            panel_dict=panel_dict,
        )

    # keep path: attach hardness
    hardness = panel_result.panel_hardness()
    task = candidate.task.model_copy(
        update={
            "panel": PanelHardness(
                grok_4_5=hardness.grok_4_5,
                opus_4_8=hardness.opus_4_8,
                pass_at_k=hardness.pass_at_k,
                discrimination=hardness.discrimination,
            ),
            "gate_proof": {
                **(candidate.task.gate_proof or {}),
                "panel": hardness_dict_from_decision(panel_result.decision),
                "panel_scaffold": panel_result.scaffold,
                "panel_models": list(panel_result.reserved_models),
            },
        }
    )
    funnel["panel_keep"] = 1
    emit(
        "panel",
        "keep",
        verdict=panel_result.decision.verdict,
        rule=panel_result.decision.rule,
        pass_at_k=panel_result.decision.frontier_pass_at_k,
        discrimination=panel_result.decision.discrimination,
        hardness=hardness_dict_from_decision(panel_result.decision),
        cost_usd=format(panel_result.total_cost_usd, "f"),
        models=list(panel_result.reserved_models),
    )

    # ------------------------------------------------------------------ export
    try:
        require_panel_hardness(task)
        # Count export success before writing the report so funnel counters
        # in report.md / micro_report.json match grounded stage outcomes.
        funnel["export_ok"] = 1
        export_root = out / "export"
        if export_root.exists():
            shutil.rmtree(export_root)
        bundle = write_export_bundle(
            tasks=[task],
            out_dir=export_root,
            broken_repos={task.instance_id: candidate.broken_workspace},
            require_clean_leak_scan=True,
            require_panel=True,
        )
        # Stage-ordered funnel report (post-export; counters fully updated)
        report = {
            "pipeline": "micro-keep",
            "instance_id": task.instance_id,
            "source_track": track_val,
            "base_commit": task.base_commit,
            "funnel": dict(funnel),
            "spend_exact_usd": format(spent_this_run(), "f"),
            "panel": hardness_dict_from_decision(panel_result.decision),
            "stages": [s.to_dict() for s in stages],
            "models": list(REQUIRED_PANEL_MODELS),
            "export": {
                "out_dir": str(bundle.out_dir),
                "tasks_jsonl": str(bundle.tasks_jsonl),
                "leak_clean": bundle.leak_scan.clean,
            },
        }
        (export_root / "report.md").write_text(
            _render_report(report),
            encoding="utf-8",
        )
        (export_root / "micro_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # also mirror stage log into export
        shutil.copy2(stage_log, export_root / "stage_log.jsonl")
    except (ExportError, Exception) as exc:  # noqa: BLE001
        funnel["export_ok"] = 0
        emit("export", "fail", error=str(exc))
        return terminal(
            ok=False,
            is_keep=False,
            escalated=False,
            task=task,
            export_dir=None,
            reason=f"export fail-closed: {exc}",
            panel_dict=panel_dict,
        )

    emit(
        "export",
        "ok",
        out_dir=str(bundle.out_dir),
        tasks_jsonl=str(bundle.tasks_jsonl),
        instance_id=task.instance_id,
        leak_clean=bundle.leak_scan.clean,
        has_panel=True,
        source_track=track_val,
    )
    emit("done", "keep", instance_id=task.instance_id)

    return terminal(
        ok=True,
        is_keep=True,
        escalated=False,
        task=task,
        export_dir=bundle.out_dir,
        reason="certified keep exported with panel hardness + source_track",
        panel_dict=panel_dict,
    )


def _render_report(report: dict[str, Any]) -> str:
    funnel = report.get("funnel") or {}
    panel = report.get("panel") or {}
    lines = [
        "# Micro-keep pipeline report",
        "",
        f"- instance_id: `{report.get('instance_id')}`",
        f"- source_track: `{report.get('source_track')}`",
        f"- base_commit: `{report.get('base_commit')}`",
        f"- spend_exact_usd (run delta): {report.get('spend_exact_usd')}",
        "",
        "## Funnel",
        "",
    ]
    for key, val in funnel.items():
        lines.append(f"- {key}: {val}")
    lines.extend(
        [
            "",
            "## Panel hardness",
            "",
            f"- pass_at_k: {panel.get('pass_at_k')}",
            f"- discrimination: {panel.get('discrimination')}",
            f"- grok_4_5: {panel.get('grok_4_5')}",
            f"- kimi_k2_6: {panel.get('kimi_k2_6')}",
            f"- opus_4_8: {panel.get('opus_4_8')}",
            f"- band_verdict: {panel.get('band_verdict')}",
            f"- band_rule: {panel.get('band_rule')}",
            "",
            "## Models",
            "",
        ]
    )
    for m in report.get("models") or []:
        lines.append(f"- {m}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MICRO_CAP_USD",
    "MicroKeepError",
    "MicroKeepResult",
    "StageEvent",
    "require_panel_hardness",
    "run_micro_keep",
]
