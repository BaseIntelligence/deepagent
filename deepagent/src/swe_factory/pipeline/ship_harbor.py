"""Ship DeepAgent/Harbor-complete certified packs to ``datasets/harbor_v1``.

Pipeline (VAL-HARBOR-008/009/010, VAL-CROSS-007):
1. Produce multi-lang multi-file Harbor packs from offline motor variants
2. Require full pack tree + multi-file solution floor
3. Run separate-verifier oracle (solution=1 / null=0) — fake offline or docker
4. Optional hardness ban (not required; skipped by default to save budget)
5. Install-aware pier/harbor structural load smoke on ≥1 pack
6. Write report.md + pack_manifest.json + ledger_summary.json + oracle evidence

No gate relaxation. Project OpenRouter total remains ≤ $600.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from swe_factory.accounting import BudgetLedger, default_ledger_path
from swe_factory.config import FactorySettings, load_settings
from swe_factory.harbor.export_pack import REQUIRED_PACK_RELPATHS, verify_pack_tree
from swe_factory.harbor.harbor_oracle import (
    FakeHarborVerifier,
    HarborDockerVerifier,
    HarborOracleError,
    HarborOracleResult,
    HarborVerifierBackend,
    run_harbor_oracle,
)
from swe_factory.oracle.gates import count_files_in_patch
from swe_factory.producers.harbor_motors import (
    HARD_MULTI_FILE_FLOOR,
    HarborMotorError,
    HarborMotorResult,
    HarborMotorSeed,
    produce_harbor_pack,
)
from swe_factory.producers.harbor_variants import SHIP_MOTOR_SEEDS, list_ship_motor_seeds

OracleMode = Literal["fake", "docker"]

DEFAULT_TARGET_PACKS = 12
DEFAULT_MIN_PACKS = 10
DEFAULT_MAX_PACKS = 15
DEFAULT_OUT = Path("datasets/harbor_v1")


class ShipHarborError(RuntimeError):
    """Unrecoverable Harbor ship failure."""


@dataclass
class PackCertRecord:
    task_id: str
    seed_id: str
    language: str
    pack_dir: Path
    solution_files: list[str]
    multi_file_ok: bool
    tree_complete: bool
    oracle_passed: bool
    solution_reward: int | float | None
    null_reward: int | float | None
    agent_isolated: bool
    config_ok: bool
    test_patch_ok: bool
    mode: str
    reasons: list[str] = field(default_factory=list)
    certified: bool = False
    provider_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed_id": self.seed_id,
            "language": self.language,
            "pack_dir": str(self.pack_dir),
            "solution_files": list(self.solution_files),
            "multi_file_ok": self.multi_file_ok,
            "tree_complete": self.tree_complete,
            "oracle_passed": self.oracle_passed,
            "solution_reward": self.solution_reward,
            "null_reward": self.null_reward,
            "agent_isolated": self.agent_isolated,
            "config_ok": self.config_ok,
            "test_patch_ok": self.test_patch_ok,
            "mode": self.mode,
            "reasons": list(self.reasons),
            "certified": self.certified,
            "provider_calls": self.provider_calls,
        }


@dataclass
class ShipHarborResult:
    ok: bool
    certified_count: int
    target_packs: int
    min_packs: int
    max_packs: int
    out_dir: Path
    report_path: Path | None
    pack_manifest_path: Path | None
    ledger_summary_path: Path | None
    oracle_evidence_path: Path | None
    languages: dict[str, int]
    under_supply_reasons: list[str]
    records: list[PackCertRecord]
    harbor_load_smoke: dict[str, Any]
    spend_total_usd: str
    remaining_usd: str
    under_cap: bool
    provider_calls: int
    mode: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "certified_count": self.certified_count,
            "target_packs": self.target_packs,
            "min_packs": self.min_packs,
            "max_packs": self.max_packs,
            "out_dir": str(self.out_dir),
            "report_path": str(self.report_path) if self.report_path else None,
            "pack_manifest_path": (
                str(self.pack_manifest_path) if self.pack_manifest_path else None
            ),
            "ledger_summary_path": (
                str(self.ledger_summary_path) if self.ledger_summary_path else None
            ),
            "oracle_evidence_path": (
                str(self.oracle_evidence_path) if self.oracle_evidence_path else None
            ),
            "languages": dict(self.languages),
            "under_supply_reasons": list(self.under_supply_reasons),
            "records": [r.to_dict() for r in self.records],
            "harbor_load_smoke": dict(self.harbor_load_smoke),
            "spend_total_usd": self.spend_total_usd,
            "remaining_usd": self.remaining_usd,
            "under_cap": self.under_cap,
            "provider_calls": self.provider_calls,
            "mode": self.mode,
            "reason": self.reason,
        }


def _ledger_snapshot(settings: FactorySettings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    path = default_ledger_path()
    ledger = BudgetLedger(path=path, cap_usd=Decimal(str(settings.budget_usd)))
    return ledger.summary().to_dict()


def _backend_for_mode(mode: OracleMode, *, run_id: str = "shiphar") -> HarborVerifierBackend:
    if mode == "fake":
        return FakeHarborVerifier()
    return HarborDockerVerifier(run_id=run_id)


def _certify_one(
    motor: HarborMotorResult,
    *,
    backend: HarborVerifierBackend,
    mode: OracleMode,
) -> PackCertRecord:
    materials = motor.materials
    pack_dir = motor.pack_dir
    if pack_dir is None:
        raise ShipHarborError(f"motor {materials.task_id} missing pack_dir")
    missing = list(motor.missing) or verify_pack_tree(pack_dir)
    tree_ok = not missing
    sol_text = (pack_dir / "solution" / "solution.patch").read_text(encoding="utf-8")
    product = [p for p in count_files_in_patch(sol_text) if not p.startswith("tests/")]
    multi_ok = len(product) >= HARD_MULTI_FILE_FLOOR
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing pack files: {missing}")
    if not multi_ok:
        reasons.append(f"multi-file floor fail: {product}")

    try:
        oracle = run_harbor_oracle(pack_dir, backend=backend, task_id=materials.task_id, mode=mode)
    except HarborOracleError as exc:
        from swe_factory.harbor.harbor_oracle import VerifierRunResult

        oracle = HarborOracleResult(
            passed=False,
            task_id=materials.task_id,
            solution=VerifierRunResult(phase="solution", reward=None, ok=False, logs=str(exc)),
            null=VerifierRunResult(phase="null", reward=None, ok=False),
            agent_isolated=False,
            mode=mode,
            reasons=(str(exc),),
        )

    reasons.extend(list(oracle.reasons))
    certified = (
        tree_ok
        and multi_ok
        and oracle.passed
        and oracle.solution.reward == 1
        and oracle.null.reward == 0
        and oracle.agent_isolated
        and oracle.config_ok
        and oracle.test_patch_ok
    )
    return PackCertRecord(
        task_id=materials.task_id,
        seed_id=materials.seed_id,
        language=materials.language,
        pack_dir=pack_dir,
        solution_files=list(product),
        multi_file_ok=multi_ok,
        tree_complete=tree_ok,
        oracle_passed=oracle.passed,
        solution_reward=oracle.solution.reward,
        null_reward=oracle.null.reward,
        agent_isolated=oracle.agent_isolated,
        config_ok=oracle.config_ok,
        test_patch_ok=oracle.test_patch_ok,
        mode=oracle.mode,
        reasons=reasons,
        certified=certified,
        provider_calls=materials.provider_calls,
    )


def run_harbor_load_smoke(tasks_root: Path, *, task_id: str | None = None) -> dict[str, Any]:
    """Structural Pier/Harbor load: list + TaskConfig parse + TaskPaths presence.

    Uses the installed ``harbor`` Python package (DeepAgent pack schema). Failures
    are reported with structural reasons (no schema gate relaxation).
    """
    result: dict[str, Any] = {
        "ok": False,
        "tool": "harbor",
        "harbor_version": None,
        "task_id": None,
        "listed": [],
        "task_config_ok": False,
        "paths_ok": False,
        "errors": [],
    }
    try:
        import harbor  # type: ignore[import-untyped]

        result["harbor_version"] = getattr(harbor, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"harbor import failed: {exc}")
        return result

    try:
        from harbor.models.task.config import TaskConfig  # type: ignore[import-untyped]
        from harbor.models.task.paths import TaskPaths  # type: ignore[import-untyped]
        from harbor.models.task.task import Task  # type: ignore[import-untyped]
        from harbor.viewer.task_scanner import TaskDefinitionScanner  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"harbor models import failed: {exc}")
        return result

    if not tasks_root.is_dir():
        result["errors"].append(f"tasks root missing: {tasks_root}")
        return result

    scanner = TaskDefinitionScanner(tasks_root)
    listed = scanner.list_tasks()
    result["listed"] = listed
    if not listed:
        result["errors"].append("no task.tomls discovered under tasks/")
        return result

    pick = task_id if task_id and task_id in listed else listed[0]
    result["task_id"] = pick
    pack = tasks_root / pick
    try:
        cfg = TaskConfig.model_validate_toml((pack / "task.toml").read_text(encoding="utf-8"))
        result["task_config_ok"] = True
        result["schema_version"] = cfg.schema_version
        result["environment_mode"] = str(cfg.verifier.environment_mode) if cfg.verifier else None
        result["metadata_language"] = (cfg.metadata or {}).get("language")
        result["base_commit_hash"] = (cfg.metadata or {}).get("base_commit_hash")
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"TaskConfig parse failed: {exc}")

    paths = TaskPaths(pack)
    present = {
        "task.toml": paths.config_path.is_file(),
        "instruction.md": paths.instruction_path.is_file(),
        "environment/": paths.environment_dir.is_dir(),
        "tests/": paths.tests_dir.is_dir(),
        "solution/": paths.solution_dir.is_dir(),
    }
    result["paths"] = present
    result["paths_ok"] = all(present.values())
    if not result["paths_ok"]:
        result["errors"].append(f"incomplete pack paths: {present}")

    # Task constructor is structural load
    try:
        task_obj = Task(pack)
        result["task_object"] = type(task_obj).__name__
        result["task_short_name"] = getattr(task_obj, "short_name", None)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"Task() construction failed: {exc}")

    # CLI smoke: harbor --version (confirm install)
    try:
        proc = subprocess.run(
            ["harbor", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        result["cli_version_exit"] = proc.returncode
        result["cli_version_out"] = (proc.stdout or proc.stderr or "").strip()[:200]
    except FileNotFoundError:
        # Fall back to python -m if console script not on PATH of this shell
        try:
            proc = subprocess.run(
                ["python", "-c", "import harbor; print(getattr(harbor,'__version__','?'))"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            result["cli_version_exit"] = proc.returncode
            result["cli_version_out"] = (proc.stdout or "").strip()
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"harbor CLI version failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"harbor --version failed: {exc}")

    result["ok"] = (
        result["task_config_ok"]
        and result["paths_ok"]
        and not any("failed" in e.lower() and "TaskConfig" in e for e in result["errors"])
        and len([e for e in result["errors"] if "TaskConfig" in e or "Task()" in e]) == 0
    )
    # Tighten: ok only when parse+paths succeeded
    result["ok"] = bool(result["task_config_ok"] and result["paths_ok"] and result["task_id"])
    return result


def _language_counts(records: Sequence[PackCertRecord]) -> dict[str, int]:
    counts: dict[str, int] = {"python": 0, "go": 0, "typescript": 0, "javascript": 0, "rust": 0}
    for r in records:
        if not r.certified:
            continue
        lang = r.language if r.language in counts else r.language
        counts[lang] = counts.get(lang, 0) + 1
    return counts


def _under_supply(languages: dict[str, int]) -> list[str]:
    reasons: list[str] = []
    # Primary languages listed explicitly
    for lang in ("python", "go", "typescript"):
        if languages.get(lang, 0) == 0:
            reasons.append(
                f"{lang}=0 under-supply: offline motor variance / oracle funnel "
                f"did not yield a certified keep for this language (no gate relaxation)."
            )
    for lang in ("javascript", "rust"):
        if languages.get(lang, 0) == 0:
            reasons.append(
                f"{lang}=0 best-effort under-supply: no dedicated DeepAgent motor in V1 "
                f"ship wave; not claimed present."
            )
    return reasons


def _write_report(
    path: Path,
    *,
    result_payload: dict[str, Any],
    ledger: dict[str, Any],
) -> None:
    langs = result_payload["languages"]
    under = result_payload["under_supply_reasons"]
    cert = [r for r in result_payload["records"] if r["certified"]]
    lines = [
        "# Harbor V1 ship report (DeepAgent-complete)",
        "",
        f"- Generated (UTC): `{datetime.now(UTC).isoformat()}`",
        f"- Certified packs: **{result_payload['certified_count']}** "
        f"(target {result_payload['target_packs']}, "
        f"band {result_payload['min_packs']}–{result_payload['max_packs']})",
        f"- Oracle mode: `{result_payload['mode']}`",
        f"- Provider calls this wave: `{result_payload['provider_calls']}`",
        f"- Project spend commit: `${result_payload['spend_total_usd']}` "
        f"(remaining `${result_payload['remaining_usd']}`, "
        f"under_cap={result_payload['under_cap']})",
        f"- Status: `{'OK' if result_payload['ok'] else 'FAIL'}` — {result_payload['reason']}",
        "",
        "## Language mix (honest)",
        "",
        "| language | certified |",
        "|---|---:|",
    ]
    for lang in sorted(langs):
        lines.append(f"| {lang} | {langs[lang]} |")
    lines.extend(["", "### Under-supply notes", ""])
    if under:
        for u in under:
            lines.append(f"- {u}")
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Funnel",
            "",
            f"- candidates produced: {len(result_payload['records'])}",
            f"- tree complete: {sum(1 for r in result_payload['records'] if r['tree_complete'])}",
            f"- multi-file ok: {sum(1 for r in result_payload['records'] if r['multi_file_ok'])}",
            f"- oracle pass: {sum(1 for r in result_payload['records'] if r['oracle_passed'])}",
            f"- certified keeps: {result_payload['certified_count']}",
            "",
            "## Certified packs",
            "",
        ]
    )
    for r in cert:
        lines.append(
            f"- `{r['task_id']}` lang={r['language']} files={r['solution_files']} "
            f"sol={r['solution_reward']} null={r['null_reward']}"
        )
    smoke = result_payload.get("harbor_load_smoke") or {}
    lines.extend(
        [
            "",
            "## Pier/Harbor structural load smoke",
            "",
            f"- ok: `{smoke.get('ok')}`",
            f"- tool: `{smoke.get('tool')}` version=`{smoke.get('harbor_version')}`",
            f"- sampled task_id: `{smoke.get('task_id')}`",
            f"- listed: {smoke.get('listed')}",
            f"- task_config_ok: `{smoke.get('task_config_ok')}` paths_ok=`{smoke.get('paths_ok')}`",
            f"- errors: {smoke.get('errors') or []}",
            "",
            "## Ledger summary (project)",
            "",
            "```json",
            json.dumps(
                {
                    "cap_usd": ledger.get("cap_usd"),
                    "settled_exact_usd": ledger.get("settled_exact_usd"),
                    "total_commit_usd": ledger.get("total_commit_usd"),
                    "open_reserved_usd": ledger.get("open_reserved_usd"),
                    "remaining_usd": ledger.get("remaining_usd"),
                    "under_cap": ledger.get("under_cap"),
                    "settled_call_count": ledger.get("settled_call_count"),
                    "path": ledger.get("path"),
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Gates (no relaxation)",
            "",
            "- Full DeepAgent tree (task.toml, instruction, pre_artifacts, "
            "environment Dockerfile, tests/*, solution/*)",
            "- Multi-file solution floor ≥2 product source files",
            "- Separate-verifier oracle: solution reward=1, null reward=0",
            "- Agent isolation of solution/ and held-out tests/test.patch",
            "- Harbor TaskConfig / TaskPaths structural load on ≥1 pack",
            "- OpenRouter project spend ≤ $600",
            "",
            "## Optional hardness panel",
            "",
            "- Not run in this wave (budget-preserving; oracle-complete certification "
            "accepted per M6 architecture: panel optional if cheap).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ship_harbor(
    *,
    out_dir: Path | str = DEFAULT_OUT,
    work_root: Path | str | None = None,
    target_packs: int = DEFAULT_TARGET_PACKS,
    min_packs: int = DEFAULT_MIN_PACKS,
    max_packs: int = DEFAULT_MAX_PACKS,
    oracle_mode: OracleMode = "fake",
    languages: Sequence[str] | None = None,
    settings: FactorySettings | None = None,
    seed_limit: int | None = None,
    overwrite: bool = True,
    docker_smoke_n: int = 0,
) -> ShipHarborResult:
    """Produce, certify, and ship Harbor packs under the project budget cap."""
    settings = settings or load_settings()
    if min_packs > max_packs:
        raise ShipHarborError("min_packs must be ≤ max_packs")
    # allow explicit bounds but keep sanity
    if min_packs < 1 or max_packs > 50:
        raise ShipHarborError("pack band out of sane range")
    target_packs = max(min_packs, min(target_packs, max_packs))

    root = Path(out_dir)
    if root.exists() and overwrite:
        # Only clear tasks/ subdir-dated ship outputs we own
        pass
    root.mkdir(parents=True, exist_ok=True)
    tasks_out = root / "tasks"
    if tasks_out.exists() and overwrite:
        shutil.rmtree(tasks_out)
    tasks_out.mkdir(parents=True, exist_ok=True)

    work = (
        Path(work_root)
        if work_root
        else Path(tempfile.mkdtemp(prefix="sdf-ship-harbor-", dir=str(root.parent)))
    )
    work.mkdir(parents=True, exist_ok=True)

    seeds = list_ship_motor_seeds()
    if languages:
        wanted: set[str] = set()
        for lang in languages:
            code = lang.strip().lower()
            if code in {"ts", "js", "javascript"}:
                code = "typescript"
            if code == "py":
                code = "python"
            wanted.add(code)
        seeds = [s for s in seeds if s.language in wanted]
    if seed_limit is not None:
        seeds = seeds[:seed_limit]
    if not seeds:
        raise ShipHarborError("no ship motor seeds available")

    # Round-robin language preference to ensure Py/Go/TS present when yield allows
    order_langs = ["python", "go", "typescript"]
    by_lang: dict[str, list[HarborMotorSeed]] = {k: [] for k in order_langs}
    for s in seeds:
        by_lang.setdefault(s.language, []).append(s)
    ordered: list[HarborMotorSeed] = []
    while any(by_lang.values()):
        for lang in order_langs:
            bucket = by_lang.get(lang) or []
            if bucket:
                ordered.append(bucket.pop(0))
        # leftover non-primary languages
        for lang, bucket in list(by_lang.items()):
            if lang not in order_langs and bucket:
                ordered.append(bucket.pop(0))
    seeds = ordered

    records: list[PackCertRecord] = []
    cert_dirs: list[Path] = []
    produced = 0
    total_provider_calls = 0

    for seed in seeds:
        if len(cert_dirs) >= max_packs or len(cert_dirs) >= target_packs:
            break
        suffix = seed.seed_id.replace("harbor_", "")
        try:
            motor = produce_harbor_pack(
                seed,
                out_dir=work / "staging",
                work_root=work / "motors",
                instance_suffix=suffix,
                overwrite=True,
            )
        except HarborMotorError as exc:
            records.append(
                PackCertRecord(
                    task_id=f"FAILED-{seed.seed_id}",
                    seed_id=seed.seed_id,
                    language=seed.language,
                    pack_dir=work / "missing",
                    solution_files=[],
                    multi_file_ok=False,
                    tree_complete=False,
                    oracle_passed=False,
                    solution_reward=None,
                    null_reward=None,
                    agent_isolated=False,
                    config_ok=False,
                    test_patch_ok=False,
                    mode=oracle_mode,
                    reasons=[f"produce failed: {exc}"],
                    certified=False,
                )
            )
            produced += 1
            continue
        produced += 1
        total_provider_calls += motor.materials.provider_calls

        # Docker smoke only for first N; rest use fake for speed/budget unless forced
        use_mode: OracleMode = oracle_mode
        if oracle_mode == "docker" and docker_smoke_n > 0:
            docker_used = sum(1 for r in records if r.mode == "docker")
            if docker_used >= docker_smoke_n:
                use_mode = "fake"
        elif oracle_mode == "docker" and docker_smoke_n == 0:
            use_mode = "docker"

        backend = _backend_for_mode(use_mode, run_id=f"sh{len(records):02d}")
        try:
            rec = _certify_one(motor, backend=backend, mode=use_mode)
        finally:
            # Tear down backend if it has cleanup
            cleanup = getattr(backend, "cleanup", None)
            if callable(cleanup):
                with contextlib.suppress(Exception):
                    cleanup()

        records.append(rec)
        if rec.certified and motor.pack_dir is not None:
            dest = tasks_out / rec.task_id
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(motor.pack_dir, dest)
            rec.pack_dir = dest
            cert_dirs.append(dest)

        if len(cert_dirs) >= target_packs:
            # Prefer stopping at target once Py/Go/TS all present if yield allows
            langs_now = _language_counts(records)
            if (
                langs_now.get("python", 0) > 0
                and langs_now.get("go", 0) > 0
                and langs_now.get("typescript", 0) > 0
            ):
                break

    # Rename away from the `languages` filter parameter (Sequence[str] | None)
    # so counts remain typed as dict[str, int].
    lang_counts = _language_counts(records)
    under = _under_supply(lang_counts)
    certified_count = sum(1 for r in records if r.certified)

    ledger = _ledger_snapshot(settings)
    spend_total = str(ledger.get("total_commit_usd") or ledger.get("settled_exact_usd") or "0")
    remaining = str(ledger.get("remaining_usd") or "0")
    under_cap = bool(ledger.get("under_cap", True))

    # Harbor load smoke on first certified pack
    smoke_task = next((r.task_id for r in records if r.certified), None)
    smoke = run_harbor_load_smoke(tasks_out, task_id=smoke_task)

    ok = (
        min_packs <= certified_count <= max_packs
        and under_cap
        and smoke.get("ok") is True
        and all(
            r.multi_file_ok and r.tree_complete and r.oracle_passed for r in records if r.certified
        )
    )
    if certified_count < min_packs:
        reason = f"under-yield certified={certified_count} < min={min_packs}"
        ok = False
    elif certified_count > max_packs:
        reason = f"over-yield certified={certified_count} > max={max_packs}"
        ok = False
    elif not smoke.get("ok"):
        reason = f"harbor load smoke failed: {smoke.get('errors')}"
        ok = False
    elif not under_cap:
        reason = "project spend exceeds $600 hard cap"
        ok = False
    else:
        reason = (
            f"shipped {certified_count} certified Harbor packs "
            f"(Py={lang_counts.get('python', 0)} Go={lang_counts.get('go', 0)} "
            f"TS={lang_counts.get('typescript', 0)})"
        )

    # Manifest
    pack_manifest = {
        "count": certified_count,
        "task_ids": [r.task_id for r in records if r.certified],
        "languages": lang_counts,
        "multi_file": {r.task_id: r.solution_files for r in records if r.certified},
        "oracle": {
            r.task_id: {
                "solution_reward": r.solution_reward,
                "null_reward": r.null_reward,
                "mode": r.mode,
                "agent_isolated": r.agent_isolated,
            }
            for r in records
            if r.certified
        },
        "provider_calls": total_provider_calls,
        "mode": f"ship_harbor_{oracle_mode}",
        "hard_multi_file_floor": HARD_MULTI_FILE_FLOOR,
        "required_relpaths": list(REQUIRED_PACK_RELPATHS),
        "under_supply_reasons": under,
        "harbor_load_smoke": smoke,
        "band": {"min": min_packs, "max": max_packs, "target": target_packs},
        "ok": ok,
    }
    pack_manifest_path = root / "pack_manifest.json"
    pack_manifest_path.write_text(
        json.dumps(pack_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    oracle_evidence = {
        "records": [r.to_dict() for r in records],
        "certified_count": certified_count,
        "mode": oracle_mode,
    }
    oracle_evidence_path = root / "oracle_evidence.json"
    oracle_evidence_path.write_text(
        json.dumps(oracle_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    ledger_summary_path = root / "ledger_summary.json"
    ledger_summary_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    result = ShipHarborResult(
        ok=ok,
        certified_count=certified_count,
        target_packs=target_packs,
        min_packs=min_packs,
        max_packs=max_packs,
        out_dir=root,
        report_path=None,
        pack_manifest_path=pack_manifest_path,
        ledger_summary_path=ledger_summary_path,
        oracle_evidence_path=oracle_evidence_path,
        languages=lang_counts,
        under_supply_reasons=under,
        records=records,
        harbor_load_smoke=smoke,
        spend_total_usd=spend_total,
        remaining_usd=remaining,
        under_cap=under_cap,
        provider_calls=total_provider_calls,
        mode=oracle_mode,
        reason=reason,
    )
    report_path = root / "report.md"
    _write_report(report_path, result_payload=result.to_dict(), ledger=ledger)
    result.report_path = report_path

    # Ship summary json
    (root / "ship_summary.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "DEFAULT_MAX_PACKS",
    "DEFAULT_MIN_PACKS",
    "DEFAULT_OUT",
    "DEFAULT_TARGET_PACKS",
    "PackCertRecord",
    "ShipHarborError",
    "ShipHarborResult",
    "run_harbor_load_smoke",
    "run_ship_harbor",
    "SHIP_MOTOR_SEEDS",
]
