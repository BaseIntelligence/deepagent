"""DeepAgent Docker oracle certification path (VAL-ORCD-001..007).

Certified keeps for ``datasets/deepagent_v1`` require:

- Docker backend only (fake refused on cert CLI/API)
- solution reward == 1 and null reward == 0
- agent isolation scan clean (no solution/ / gold / held-out test.patch)
- audit evidence: sol/null rewards, isolation status, repository_url, base SHA
- dual-run flake disagreement never certifies (G2_FLAKE / FLAKE_REJECT)
- pier-ready structural load hooks for exported Harbor trees

Historical motor fixtures may still use ``FakeHarborVerifier`` via
``harbor-oracle --backend fake`` without ``--certified``.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from swe_factory.harbor.export_pack import verify_pack_tree
from swe_factory.harbor.harbor_docker import (
    HarborDockerError,
    scan_agent_context_forbidden,
    stage_agent_context,
    summarize_agent_context,
)
from swe_factory.harbor.harbor_oracle import (
    FakeHarborVerifier,
    HarborDockerVerifier,
    HarborOracleError,
    HarborOracleResult,
    HarborVerifierBackend,
    VerifierRunResult,
    run_harbor_oracle,
)
from swe_factory.oracle import codes as C
from swe_factory.producers.harbor_labeling import (
    SuiteOutcome,
    detect_dual_run_flake,
)

CertDisposition = Literal["accept", "reject"]
CertBackend = Literal["docker"]  # certified path only ever allows docker

_SHA40_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_HTTP_REPO_RE = re.compile(r"^https?://", re.IGNORECASE)
_FAKE_BACKENDS = frozenset({"fake", "stub", "mock", "offline"})
_DEEPAGENT_SHIP_TOKENS = ("deepagent_v1", "datasets/deepagent", "deepagent-v1")


class DeepAgentCertError(RuntimeError):
    """Unrecoverable DeepAgent docker-oracle certification failure."""


class FakeBackendRejected(DeepAgentCertError):
    """Raised when cert path is asked to use a fake/mock oracle backend."""


@dataclass(frozen=True, slots=True)
class PackMetaEvidence:
    """Real-repo identity fields required on DeepAgent cert audit rows."""

    task_id: str
    repository_url: str
    base_commit_hash: str
    language: str = ""
    schema_version: str = ""
    source_track: str | None = None
    license: str | None = None
    real_url_ok: bool = False
    real_sha_ok: bool = False
    reason_codes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "repository_url": self.repository_url,
            "base_commit_hash": self.base_commit_hash,
            "language": self.language,
            "schema_version": self.schema_version,
            "source_track": self.source_track,
            "license": self.license,
            "real_url_ok": self.real_url_ok,
            "real_sha_ok": self.real_sha_ok,
            "reason_codes": list(self.reason_codes),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class IsolationEvidence:
    """Agent context isolation scan result (VAL-ORCD-003)."""

    clean: bool
    hits: tuple[str, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "isolation": "clean" if self.clean else "leak",
            "hits": list(self.hits),
            "summary": dict(self.summary),
        }


@dataclass(frozen=True, slots=True)
class FlakeEvidence:
    """Dual-run flake gate outcome (VAL-ORCD-007)."""

    is_flake: bool
    phase: str
    reason_codes: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_flake": self.is_flake,
            "phase": self.phase,
            "reason_codes": list(self.reason_codes),
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class PierReadyHooks:
    """Pier-loadability structural hooks for enrolled packs (VAL-ORCD-005)."""

    structural_ok: bool
    pack_dir: str
    required_relpaths_ok: bool
    missing_relpaths: tuple[str, ...] = ()
    pier_job_prefix: str = "/tmp/harbor-deepagent-jobs"
    load_smoke: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "structural_ok": self.structural_ok,
            "pack_dir": self.pack_dir,
            "required_relpaths_ok": self.required_relpaths_ok,
            "missing_relpaths": list(self.missing_relpaths),
            "pier_job_prefix": self.pier_job_prefix,
            "load_smoke": dict(self.load_smoke),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class DeepAgentCertResult:
    """Aggregate DeepAgent docker-oracle certification outcome."""

    certified: bool
    disposition: CertDisposition
    task_id: str
    pack_dir: str
    backend: str
    solution_reward: int | float | None
    null_reward: int | float | None
    isolation: IsolationEvidence
    pack_meta: PackMetaEvidence
    flake: FlakeEvidence
    pier_ready: PierReadyHooks
    reason_codes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    oracle: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "certified": self.certified,
            "disposition": self.disposition,
            "task_id": self.task_id,
            "pack_dir": self.pack_dir,
            "backend": self.backend,
            "mode": self.backend,
            "solution_reward": self.solution_reward,
            "null_reward": self.null_reward,
            "isolation": self.isolation.to_dict(),
            "isolation_status": "clean" if self.isolation.clean else "leak",
            "agent_isolated": self.isolation.clean,
            "pack_meta": self.pack_meta.to_dict(),
            "repository_url": self.pack_meta.repository_url,
            "base_commit_hash": self.pack_meta.base_commit_hash,
            "flake": self.flake.to_dict(),
            "pier_ready": self.pier_ready.to_dict(),
            "reason_codes": list(self.reason_codes),
            "reasons": list(self.reasons),
            "oracle": dict(self.oracle),
            "audit": dict(self.audit),
        }

    def to_audit_row(self) -> dict[str, Any]:
        """Compact gate/audit row for oracle_evidence / gate_audit.jsonl."""
        return {
            "instance_id": self.task_id,
            "task_id": self.task_id,
            "disposition": self.disposition,
            "certified": self.certified,
            "backend": self.backend,
            "solution_reward": self.solution_reward,
            "null_reward": self.null_reward,
            "sol": self.solution_reward,
            "null": self.null_reward,
            "isolation": "clean" if self.isolation.clean else "leak",
            "agent_isolated": self.isolation.clean,
            "repository_url": self.pack_meta.repository_url,
            "base_commit_hash": self.pack_meta.base_commit_hash,
            "reason_codes": list(self.reason_codes),
            "flake": self.flake.is_flake,
            "pier_structural_ok": self.pier_ready.structural_ok,
        }


def refuse_fake_backend(
    backend: str | HarborVerifierBackend | None,
    *,
    certified: bool = True,
    dest: Path | str | None = None,
) -> None:
    """Fail closed when DeepAgent cert path is asked for a fake oracle.

    VAL-ORCD-004: CLI paths that certify into deepagent_v1 / --certified must
    refuse fake backends with an explicit error.

    Product dests (true ``deepagent_v1`` ship paths) additionally refuse any
    injectable that is not the real :class:`HarborDockerVerifier` class —
    *ScriptedDockerVerifier* / *FakeHarborVerifier* masquerade is not docker.
    Offline unit injectables remain allowed only on non-product dests.
    """
    dest_is_product = dest is not None and _is_product_deepagent_dest(dest)
    if not certified and not dest_is_product:
        return

    mode = _backend_mode_name(backend)

    if dest_is_product:
        _refuse_non_real_docker_on_product(backend, mode=mode)
        return

    if mode in _FAKE_BACKENDS or isinstance(backend, FakeHarborVerifier):
        raise FakeBackendRejected(
            "deepagent cert path refuses fake/mock oracle backend "
            f"(got backend={mode!r}); use --backend docker for certified keeps "
            "(VAL-ORCD-004 / VAL-PIER-004)"
        )
    # Unit scripted injectables may self-identify as docker via type name
    # (non-product dest only).
    if (
        mode not in {"docker", ""}
        and backend is not None
        and not isinstance(backend, HarborDockerVerifier)
    ):
        raise FakeBackendRejected(f"deepagent cert path requires backend=docker (got {mode!r})")


def _refuse_non_real_docker_on_product(
    backend: str | HarborVerifierBackend | None,
    *,
    mode: str,
) -> None:
    """Product deepagent_v1: only HarborDockerVerifier / backend='docker' allowed."""
    if isinstance(backend, FakeHarborVerifier):
        raise FakeBackendRejected(
            "product deepagent_v1 cert refuses FakeHarborVerifier; "
            "use HarborDockerVerifier only (VAL-RORC-004 / product honesty)"
        )
    if mode in _FAKE_BACKENDS:
        raise FakeBackendRejected(
            "product deepagent_v1 cert refuses fake/mock oracle backend "
            f"(got backend={mode!r}); use HarborDockerVerifier "
            "(VAL-ORCD-004 / VAL-RORC-004)"
        )
    if backend is None:
        return  # resolve to real HarborDockerVerifier downstream
    if isinstance(backend, str):
        if mode not in {"docker", ""}:
            raise FakeBackendRejected(
                f"product deepagent_v1 cert requires backend=docker (got {mode!r})"
            )
        return
    if not isinstance(backend, HarborDockerVerifier):
        cls_name = type(backend).__name__
        raise FakeBackendRejected(
            "product deepagent_v1 cert requires real HarborDockerVerifier "
            f"(got {cls_name}); refuse Scripted*/Fake* docker masquerade "
            "(product honesty / VAL-RORC-004)"
        )
    # Real HarborDockerVerifier instance is accepted.
    lower = type(backend).__name__.lower()
    if any(tok in lower for tok in ("scripted", "fake", "stub", "mock")):
        raise FakeBackendRejected(
            f"product deepagent_v1 refused masquerading verifier class {type(backend).__name__}"
        )


def _backend_mode_name(backend: str | HarborVerifierBackend | None) -> str:
    if backend is None:
        return "docker"
    if isinstance(backend, str):
        return backend.strip().lower()
    if isinstance(backend, FakeHarborVerifier):
        return "fake"
    if isinstance(backend, HarborDockerVerifier):
        return "docker"
    # Protocol backends used in unit tests report via type name heuristic
    name = type(backend).__name__.lower()
    if "fake" in name or "stub" in name or "mock" in name:
        return "fake"
    if "docker" in name:
        return "docker"
    return name or "unknown"


def _is_deepagent_ship_dest(dest: Path | str) -> bool:
    """Historical helper: token match (includes offline tmp deepagent_v1 paths)."""
    text = str(dest).replace("\\", "/").lower()
    return any(token in text for token in _DEEPAGENT_SHIP_TOKENS)


def _is_product_deepagent_dest(dest: Path | str) -> bool:
    """True only for *product* promote dests — not offline_only / fixture sandboxes.

    Only path segments containing ``deepagent`` are inspected so pytest tmp
    parents like ``test_offline_only_*`` never disable product refuse, while
    ``datasets/deepagent_v1/tasks/x`` remains product.

    Bare package-root segment ``deepagent`` (the monorepo product folder) is
    ignored so ``deepagent/datasets/deepagent_v1`` still counts as product.
    """
    text = str(dest).replace("\\", "/").lower().rstrip("/")
    parts = [p for p in text.split("/") if p]
    deepagent_parts = [p for p in parts if "deepagent" in p and p != "deepagent"]
    if not deepagent_parts:
        return False
    for part in deepagent_parts:
        if part == "deepagent_v1":
            continue
        if any(m in part for m in ("offline", "_ut_", "fixture", "sandbox", "unit", "hybrid")):
            return False
        if part != "deepagent_v1":
            return False
    return "deepagent_v1" in parts


def is_real_repository_url(url: str) -> bool:
    """True when repository_url looks like a real HTTPS public remote."""
    cleaned = (url or "").strip()
    if not cleaned:
        return False
    lower = cleaned.lower()
    if lower.startswith("file://"):
        return False
    if "example.com" in lower or "localhost" in lower:
        return False
    if "placeholder" in lower or "fake" in lower:
        return False
    return bool(_HTTP_REPO_RE.match(cleaned))


def is_real_base_sha(sha: str) -> bool:
    """True when base commit is a full 40-char hex SHA."""
    cleaned = (sha or "").strip()
    if not _SHA40_RE.match(cleaned):
        return False
    # Reject synthetic placeholders like a1000… used historically in fixtures.
    return not (cleaned.lower().startswith("a1000") and set(cleaned[5:].lower()) <= {"0", "a", "1"})


def read_pack_meta(pack_dir: Path | str) -> PackMetaEvidence:
    """Parse task.toml for cert audit identity fields."""
    root = Path(pack_dir)
    task_id = root.name
    codes: list[str] = []
    reasons: list[str] = []
    repository_url = ""
    base_commit = ""
    language = ""
    schema_version = ""
    source_track: str | None = None
    license_name: str | None = None

    toml_path = root / "task.toml"
    if not toml_path.is_file():
        codes.append("META_MISSING_TASK_TOML")
        reasons.append("pack missing task.toml")
        return PackMetaEvidence(
            task_id=task_id,
            repository_url="",
            base_commit_hash="",
            reason_codes=tuple(codes),
            reasons=tuple(reasons),
        )

    try:
        data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        codes.append("META_TASK_TOML_PARSE")
        reasons.append(f"task.toml parse failed: {exc}")
        return PackMetaEvidence(
            task_id=task_id,
            repository_url="",
            base_commit_hash="",
            reason_codes=tuple(codes),
            reasons=tuple(reasons),
        )

    schema_version = str(data.get("schema_version") or "")
    meta = data.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    repository_url = str(meta.get("repository_url") or "").strip()
    base_commit = str(meta.get("base_commit_hash") or "").strip()
    language = str(meta.get("language") or "").strip()
    source_track = meta.get("source_track")
    if source_track is not None:
        source_track = str(source_track)
    license_name = meta.get("license")
    if license_name is not None:
        license_name = str(license_name)
    if meta.get("task_id"):
        task_id = str(meta["task_id"]).strip() or task_id

    real_url = is_real_repository_url(repository_url)
    real_sha = is_real_base_sha(base_commit)
    if not real_url:
        codes.append("META_BAD_REPOSITORY_URL")
        reasons.append(f"repository_url not real public git remote: {repository_url!r}")
    if not real_sha:
        codes.append("META_BAD_BASE_COMMIT")
        reasons.append(f"base_commit_hash not real 40-char SHA: {base_commit!r}")

    return PackMetaEvidence(
        task_id=task_id,
        repository_url=repository_url,
        base_commit_hash=base_commit,
        language=language,
        schema_version=schema_version,
        source_track=source_track,
        license=license_name,
        real_url_ok=real_url,
        real_sha_ok=real_sha,
        reason_codes=tuple(codes),
        reasons=tuple(reasons),
    )


def scan_pack_agent_isolation(
    pack_dir: Path | str, *, work_dir: Path | None = None
) -> IsolationEvidence:
    """Stage agent context and scan for solution/gold/held-out leaks (VAL-ORCD-003)."""
    import shutil
    import tempfile

    root = Path(pack_dir)
    tmp_parent = work_dir or Path(tempfile.mkdtemp(prefix="sdf-deepagent-isol-"))
    own = work_dir is None
    agent_dest = Path(tmp_parent) / "agent_context"
    hits: list[str] = []
    summary: dict[str, Any] = {}
    try:
        staged = stage_agent_context(root, agent_dest, overwrite=True)
        hits = list(staged.forbidden_hits) or scan_agent_context_forbidden(staged.context_dir)
        summary = summarize_agent_context(staged.context_dir)
        # Defense: also reject if environment itself embeds gold bodies at pack root copy misses
        env = root / "environment"
        if env.is_dir():
            for name in ("solution.patch", "gold.patch", "test.patch", "solve.sh"):
                if (env / name).is_file():
                    hits.append(f"forbidden agent file at environment/{name}")
            if (env / "solution").exists():
                hits.append("forbidden agent path: environment/solution/")
    except HarborDockerError as exc:
        hits.append(str(exc))
        summary = {"error": str(exc)}
    finally:
        if own:
            shutil.rmtree(tmp_parent, ignore_errors=True)

    clean = len(hits) == 0
    if clean and summary.get("isolated") is False:
        clean = False
        hits.append("summarize_agent_context reports not isolated")
    return IsolationEvidence(clean=clean, hits=tuple(hits), summary=summary)


def evaluate_flake_gate(
    *,
    gold_runs: Sequence[SuiteOutcome] | None = None,
    null_runs: Sequence[SuiteOutcome] | None = None,
    forced_flake: bool = False,
    phase: str = "gold",
) -> FlakeEvidence:
    """VAL-ORCD-007: dual gold/null signature disagreement rejects cert."""
    if forced_flake:
        return FlakeEvidence(
            is_flake=True,
            phase=phase,
            reason_codes=(C.G2_FLAKE, C.FLAKE_REJECT),
            details={"forced": True},
        )

    runs = list(gold_runs or [])
    if len(runs) >= 2:
        is_flake, codes, details = detect_dual_run_flake(runs, phase="gold")
        if is_flake:
            return FlakeEvidence(
                is_flake=True,
                phase="gold",
                reason_codes=tuple(codes) or (C.G2_FLAKE, C.FLAKE_REJECT),
                details=details,
            )

    nulls = list(null_runs or [])
    if len(nulls) >= 2:
        is_flake, codes, details = detect_dual_run_flake(nulls, phase="null")
        if is_flake:
            return FlakeEvidence(
                is_flake=True,
                phase="null",
                reason_codes=tuple(codes) or (C.G2_FLAKE, C.FLAKE_REJECT),
                details=details,
            )

    return FlakeEvidence(is_flake=False, phase=phase, reason_codes=(), details={})


def build_pier_ready_hooks(
    pack_dir: Path | str,
    *,
    run_load_smoke: bool = True,
    pier_job_prefix: str = "/tmp/harbor-deepagent-jobs",
) -> PierReadyHooks:
    """Produce pier-ready structural hooks for a Harbor pack (VAL-ORCD-005).

    Full Pier oracle agent execution is deferred to the pier-cert adapter;
    this hook guarantees tree completeness + optional Harbor TaskConfig load.
    """
    root = Path(pack_dir).resolve()
    missing = tuple(verify_pack_tree(root))
    relpaths_ok = not missing
    smoke: dict[str, Any] = {}
    notes: list[str] = []
    if run_load_smoke and relpaths_ok:
        try:
            from swe_factory.pipeline.ship_harbor import run_harbor_load_smoke

            # When pack is .../tasks/<id>, root for scanner is parent; when pack
            # is standalone, create a temporary listing via parent name.
            parent = root.parent
            smoke = run_harbor_load_smoke(parent, task_id=root.name)
            if not smoke.get("ok"):
                notes.append(f"pier structural load smoke not ok: {smoke.get('errors')}")
        except Exception as exc:  # noqa: BLE001
            smoke = {"ok": False, "error": str(exc)}
            notes.append(f"pier load smoke unavailable: {exc}")
    elif not relpaths_ok:
        notes.append(f"missing required pack relpaths: {list(missing)}")

    structural_ok = relpaths_ok and (smoke.get("ok", True) if smoke else relpaths_ok)
    return PierReadyHooks(
        structural_ok=structural_ok,
        pack_dir=str(root),
        required_relpaths_ok=relpaths_ok,
        missing_relpaths=missing,
        pier_job_prefix=pier_job_prefix,
        load_smoke=smoke,
        notes=tuple(notes),
    )


def _resolve_cert_backend(
    backend: str | HarborVerifierBackend | None,
    *,
    run_id: str = "deepagent",
) -> tuple[HarborVerifierBackend, str]:
    refuse_fake_backend(backend, certified=True)
    if backend is None:
        return HarborDockerVerifier(run_id=run_id), "docker"
    if isinstance(backend, str):
        mode = backend.strip().lower()
        refuse_fake_backend(mode, certified=True)
        if mode != "docker":
            raise FakeBackendRejected(f"deepagent cert path requires backend=docker (got {mode!r})")
        return HarborDockerVerifier(run_id=run_id), "docker"
    if isinstance(backend, FakeHarborVerifier):
        raise FakeBackendRejected("deepagent cert path refuses FakeHarborVerifier (VAL-ORCD-004)")
    mode_name = _backend_mode_name(backend)
    if mode_name in _FAKE_BACKENDS:
        raise FakeBackendRejected(
            f"deepagent cert path refuses fake backend class {type(backend).__name__}"
        )
    # Injectable backends (unit tests) must identify as docker
    return backend, "docker"


def collect_cert_audit_fields(result: DeepAgentCertResult) -> dict[str, Any]:
    """Checklist of required audit fields for VAL-ORCD-006 ship inclusion."""
    required = {
        "solution_reward": result.solution_reward,
        "null_reward": result.null_reward,
        "isolation": "clean" if result.isolation.clean else "leak",
        "repository_url": result.pack_meta.repository_url,
        "base_commit_hash": result.pack_meta.base_commit_hash,
        "backend": result.backend,
        "task_id": result.task_id,
    }
    missing = [
        key
        for key, val in required.items()
        if val is None or val == "" or (key == "backend" and val != "docker")
    ]
    sol_ok = result.solution_reward == 1
    null_ok = result.null_reward == 0
    isol_ok = result.isolation.clean
    complete = (
        not missing
        and sol_ok
        and null_ok
        and isol_ok
        and result.pack_meta.real_url_ok
        and result.pack_meta.real_sha_ok
        and result.backend == "docker"
    )
    return {
        "fields": required,
        "missing": missing,
        "sol_ok": sol_ok,
        "null_ok": null_ok,
        "isolation_ok": isol_ok,
        "complete": complete,
        "blocks_ship": not complete,
    }


def write_oracle_evidence(
    path: Path | str,
    result: DeepAgentCertResult,
) -> Path:
    """Persist full cert evidence JSON (VAL-ORCD-006 surface)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def append_cert_audit(
    path: Path | str,
    result: DeepAgentCertResult,
) -> Path:
    """Append one audit row for the keep funnel (gate_audit style)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_audit_row(), sort_keys=True) + "\n")
    return out


def certify_deepagent_pack(
    pack_dir: Path | str,
    *,
    backend: str | HarborVerifierBackend | None = "docker",
    task_id: str | None = None,
    evidence_out: Path | str | None = None,
    audit_out: Path | str | None = None,
    gold_runs: Sequence[SuiteOutcome] | None = None,
    null_runs: Sequence[SuiteOutcome] | None = None,
    forced_flake: bool = False,
    run_pier_hooks: bool = True,
    dest_hint: Path | str | None = None,
    cleanup: bool = True,
    run_id: str = "deepagent",
) -> DeepAgentCertResult:
    """Docker-oracle certify one Harbor pack for DeepAgent ship eligibility.

    Always refuses fake backends (VAL-ORCD-004). Requires sol=1 / null=0
    (VAL-ORCD-001/002), isolation clean (VAL-ORCD-003), complete audit fields
    (VAL-ORCD-006), and no dual-run flake (VAL-ORCD-007).
    """
    refuse_fake_backend(backend, certified=True, dest=dest_hint)
    root = Path(pack_dir)
    if not root.is_dir():
        raise DeepAgentCertError(f"pack dir not found: {root}")

    pack_meta = read_pack_meta(root)
    tid = task_id or pack_meta.task_id or root.name
    isolation = scan_pack_agent_isolation(root)
    flake = evaluate_flake_gate(
        gold_runs=gold_runs,
        null_runs=null_runs,
        forced_flake=forced_flake,
    )

    codes: list[str] = []
    reasons: list[str] = []
    codes.extend(pack_meta.reason_codes)
    reasons.extend(pack_meta.reasons)

    if not isolation.clean:
        codes.append(C.G5_LEAK)
        reasons.append(f"agent isolation leak: {list(isolation.hits)}")

    if flake.is_flake:
        codes.extend(flake.reason_codes or (C.G2_FLAKE, C.FLAKE_REJECT))
        reasons.append(f"flake dual-run reject phase={flake.phase}")

    oracle_dict: dict[str, Any] = {}
    solution_reward: int | float | None = None
    null_reward: int | float | None = None
    try:
        runner, mode = _resolve_cert_backend(backend, run_id=run_id)
    except FakeBackendRejected:
        raise
    except DeepAgentCertError:
        raise

    try:
        oracle = run_harbor_oracle(
            root,
            backend=runner,
            task_id=tid,
            mode=mode,
            cleanup=cleanup,
        )
    except HarborOracleError as exc:
        oracle = HarborOracleResult(
            passed=False,
            task_id=tid,
            solution=VerifierRunResult(phase="solution", reward=None, ok=False, logs=str(exc)),
            null=VerifierRunResult(phase="null", reward=None, ok=False),
            agent_isolated=isolation.clean,
            mode=mode,
            reasons=(str(exc),),
        )
        codes.append(C.ORACLE_REJECT)
        reasons.append(f"oracle error: {exc}")

    if oracle.mode in _FAKE_BACKENDS:
        # Hard refuse any path that ended up in fake mode
        raise FakeBackendRejected(f"deepagent cert path cannot accept oracle mode={oracle.mode!r}")

    solution_reward = oracle.solution.reward
    null_reward = oracle.null.reward
    oracle_dict = oracle.to_dict()
    reasons.extend(list(oracle.reasons))

    if solution_reward != 1:
        codes.append(C.G2_GOLD_FAIL if solution_reward is not None else C.ORACLE_REJECT)
        reasons.append(f"solution reward expected 1, got {solution_reward!r}")
    else:
        codes.append("ORCD_SOL_1")

    if null_reward != 0:
        if null_reward == 1:
            codes.append(C.G3_NULL_RESOLVES)
        else:
            codes.append(C.G3_NULL_EVAL_ERROR)
        reasons.append(f"null reward expected 0, got {null_reward!r}")
    else:
        codes.append(C.G3_NULL_NOT_RESOLVE)

    if isolation.clean:
        codes.append(C.G5_LEAK_CLEAN)

    pier = (
        build_pier_ready_hooks(root)
        if run_pier_hooks
        else PierReadyHooks(
            structural_ok=not verify_pack_tree(root),
            pack_dir=str(root.resolve()),
            required_relpaths_ok=not verify_pack_tree(root),
            missing_relpaths=tuple(verify_pack_tree(root)),
        )
    )
    if not pier.required_relpaths_ok:
        codes.append("PACK_TREE_INCOMPLETE")
        reasons.append(f"incomplete pack tree: {list(pier.missing_relpaths)}")

    certified = (
        solution_reward == 1
        and null_reward == 0
        and isolation.clean
        and not flake.is_flake
        and pack_meta.real_url_ok
        and pack_meta.real_sha_ok
        and pier.required_relpaths_ok
        and mode == "docker"
        and not any(c in C.HARD_REJECT_CODES for c in codes)
    )
    if certified:
        codes.append(C.ORACLE_PASS)
    else:
        codes.append(C.ORACLE_REJECT)

    # Deduplicate reason codes while preserving order
    seen: set[str] = set()
    uniq_codes: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            uniq_codes.append(code)

    # Isolation already folded; ensure agent_isolated from oracle does not override scan
    agent_isolated = isolation.clean

    result = DeepAgentCertResult(
        certified=certified,
        disposition="accept" if certified else "reject",
        task_id=tid,
        pack_dir=str(root.resolve()),
        backend=mode,
        solution_reward=solution_reward,
        null_reward=null_reward,
        isolation=isolation,
        pack_meta=pack_meta,
        flake=flake,
        pier_ready=pier,
        reason_codes=tuple(uniq_codes),
        reasons=tuple(dict.fromkeys(reasons)),
        oracle={**oracle_dict, "agent_isolated": agent_isolated},
        audit={},
    )
    audit = collect_cert_audit_fields(result)
    if audit["blocks_ship"] and certified:
        # Audit incomplete => never ship (VAL-ORCD-006)
        result = DeepAgentCertResult(
            certified=False,
            disposition="reject",
            task_id=result.task_id,
            pack_dir=result.pack_dir,
            backend=result.backend,
            solution_reward=result.solution_reward,
            null_reward=result.null_reward,
            isolation=result.isolation,
            pack_meta=result.pack_meta,
            flake=result.flake,
            pier_ready=result.pier_ready,
            reason_codes=tuple([*result.reason_codes, "AUDIT_INCOMPLETE", C.ORACLE_REJECT]),
            reasons=tuple([*result.reasons, "cert audit fields incomplete for ship"]),
            oracle=result.oracle,
            audit=audit,
        )
    else:
        result = DeepAgentCertResult(
            certified=result.certified,
            disposition=result.disposition,
            task_id=result.task_id,
            pack_dir=result.pack_dir,
            backend=result.backend,
            solution_reward=result.solution_reward,
            null_reward=result.null_reward,
            isolation=result.isolation,
            pack_meta=result.pack_meta,
            flake=result.flake,
            pier_ready=result.pier_ready,
            reason_codes=result.reason_codes,
            reasons=result.reasons,
            oracle=result.oracle,
            audit=audit,
        )

    if evidence_out is not None:
        write_oracle_evidence(evidence_out, result)
    if audit_out is not None:
        append_cert_audit(audit_out, result)
    return result


def assert_cert_audit_complete(audit: Mapping[str, Any]) -> None:
    """Raise if audit checklist would block ship (VAL-ORCD-006)."""
    if not audit.get("complete"):
        missing = audit.get("missing") or []
        raise DeepAgentCertError(f"cert audit incomplete (blocks ship); missing={missing}")


__all__ = [
    "CertBackend",
    "CertDisposition",
    "DeepAgentCertError",
    "DeepAgentCertResult",
    "FakeBackendRejected",
    "FlakeEvidence",
    "IsolationEvidence",
    "PackMetaEvidence",
    "PierReadyHooks",
    "append_cert_audit",
    "assert_cert_audit_complete",
    "build_pier_ready_hooks",
    "certify_deepagent_pack",
    "collect_cert_audit_fields",
    "evaluate_flake_gate",
    "is_real_base_sha",
    "is_real_repository_url",
    "read_pack_meta",
    "refuse_fake_backend",
    "scan_pack_agent_isolation",
    "write_oracle_evidence",
]
