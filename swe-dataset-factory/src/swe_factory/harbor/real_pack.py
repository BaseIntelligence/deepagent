"""Real-repo Harbor pack export + structural gates (VAL-HPACK-* + VAL-RPACK-*).

Synthetic motors (file://, short/synthetic SHA) remain valid for offline
fixture export via :func:`export_harbor_pack`. Certified DeepSWE **product**
packs under ``datasets/deepswe_v1`` must pass the real-PR product gate:

- source_track=real_pr
- real HTTPS repository_url + 40-char base_commit_hash
- agent tree = clone@SHA (no hybrid_bind / motor COPY product packaging)
- multi-file gold + held-out test.patch + dual-run node ids

Historical hybrid motors (``hybrid_curated``) may still use the HPACK
structural inventory for archive-era ship, but product promote/export to
``datasets/deepswe_v1`` refuses hybrid via :func:`export_real_harbor_pack`
and :func:`assert_product_real_pr_export`.

Gates:
- HPACK 001..009 structural (shared inventory)
- RPACK product: real_pr track + real_url/sha + refuse hybrid_bind
- RCLN: clone@SHA Dockerfile when track is real_pr / product mode
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swe_factory.envbuild.agent_recipe import (
    RealPrDockerfileError,
    assert_real_pr_agent_dockerfile,
    dockerfile_has_clone_at_sha,
    looks_motor_fixture_copy,
)
from swe_factory.harbor.deepswe_cert import (
    is_real_base_sha,
    is_real_repository_url,
    scan_pack_agent_isolation,
)
from swe_factory.harbor.export_pack import (
    REQUIRED_PACK_RELPATHS,
    HarborExportError,
    HarborPackResult,
    export_harbor_pack,
    verify_pack_tree,
)
from swe_factory.harbor.harbor_docker import scan_agent_context_forbidden
from swe_factory.harbor.schema import HarborPackSpec, validate_pack_spec
from swe_factory.oracle.gates import MULTI_FILE_FLOOR, count_files_in_patch

MULTI_PRODUCT_FLOOR = max(2, MULTI_FILE_FLOOR)

# Product ship surface tokens (VAL-RPACK / VAL-RSHIP). Archive hybrid is exempt.
_PRODUCT_DEEPSWE_TOKENS = ("deepswe_v1", "datasets/deepswe", "deepswe-v1")
_HYBRID_ARCHIVE_MARKERS = (
    "hybrid_archive",
    "deepswe_v1_hybrid_archive",
    "hybrid-archive",
)
REAL_PR_SOURCE_TRACK = "real_pr"
HYBRID_SOURCE_TRACKS = frozenset(
    {
        "hybrid_curated",
        "hybrid",
        "hybrid_bind",
        "motor",
        "motor_hybrid",
        "synthetic_grounded",
        "synthetic",
    }
)

_TEST_PATH_HINTS = (
    "tests/",
    "test/",
    "spec/",
    "__tests__/",
    ".test.",
    "_test.",
    "_test.go",
    ".spec.",
)


class RealPackError(RuntimeError):
    """Raised when a pack fails real DeepSWE/Harbor export gates."""


@dataclass(frozen=True, slots=True)
class PreArtifactsCaptureResult:
    """Outcome of running pre_artifacts capture against a workspace."""

    ok: bool
    byte_size: int
    patch_path: Path | None
    patch_text: str = ""
    base_ref: str = ""
    logs: str = ""
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "byte_size": self.byte_size,
            "patch_path": str(self.patch_path) if self.patch_path else None,
            "base_ref": self.base_ref,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class RealPackValidationResult:
    """Structured HPACK gate result for one pack directory."""

    ok: bool
    task_id: str
    pack_dir: str
    tree_complete: bool
    missing_relpaths: tuple[str, ...] = ()
    real_url_ok: bool = False
    real_sha_ok: bool = False
    repository_url: str = ""
    base_commit_hash: str = ""
    schema_version: str = ""
    separate_verifier: bool = False
    base_object_ok: bool | None = None
    test_patch_ok: bool = False
    config_ok: bool = False
    f2p_count: int = 0
    p2p_count: int = 0
    multi_file_ok: bool = False
    solution_files: tuple[str, ...] = ()
    isolation_clean: bool = False
    isolation_hits: tuple[str, ...] = ()
    instruction_ok: bool = False
    instruction_leak_hits: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "task_id": self.task_id,
            "pack_dir": self.pack_dir,
            "tree_complete": self.tree_complete,
            "missing_relpaths": list(self.missing_relpaths),
            "real_url_ok": self.real_url_ok,
            "real_sha_ok": self.real_sha_ok,
            "repository_url": self.repository_url,
            "base_commit_hash": self.base_commit_hash,
            "schema_version": self.schema_version,
            "separate_verifier": self.separate_verifier,
            "base_object_ok": self.base_object_ok,
            "test_patch_ok": self.test_patch_ok,
            "config_ok": self.config_ok,
            "f2p_count": self.f2p_count,
            "p2p_count": self.p2p_count,
            "multi_file_ok": self.multi_file_ok,
            "solution_files": list(self.solution_files),
            "isolation_clean": self.isolation_clean,
            "isolation_hits": list(self.isolation_hits),
            "instruction_ok": self.instruction_ok,
            "instruction_leak_hits": list(self.instruction_leak_hits),
            "reason_codes": list(self.reason_codes),
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class RealHarborPackResult:
    """Export + validation for one real DeepSWE pack."""

    pack: HarborPackResult
    validation: RealPackValidationResult

    @property
    def pack_dir(self) -> Path:
        return self.pack.pack_dir

    @property
    def task_id(self) -> str:
        return self.pack.task_id


def is_test_path(path: str) -> bool:
    """Heuristic: path is test-suite, not product source."""
    norm = path.replace("\\", "/").lstrip("./")
    lower = norm.lower()
    name = Path(norm).name.lower()
    if any(h in lower for h in ("tests/", "test/", "spec/", "__tests__/")):
        return True
    if lower.endswith(
        ("_test.go", "_test.py", "_test.rs", ".test.ts", ".test.js", ".spec.ts", ".spec.js")
    ):
        return True
    if name.startswith("test_") and name.endswith((".py", ".js", ".ts")):
        # root-level test_*.py still test-only for multi-file floor
        parent = Path(norm).parent.as_posix()
        if parent in {".", ""}:
            return True
    return bool(".test." in name or ".spec." in name)


def product_source_files(solution_patch: str) -> list[str]:
    """Unique product/source files touched by solution.patch (VAL-HPACK-007)."""
    return [p for p in count_files_in_patch(solution_patch) if not is_test_path(p)]


def scan_instruction_gold_leak(
    instruction_md: str,
    solution_patch: str,
    *,
    min_marker_len: int = 24,
) -> list[str]:
    """Return leak findings if instruction embeds gold patch bodies (VAL-HPACK-009)."""
    hits: list[str] = []
    text = instruction_md or ""
    if not text.strip():
        hits.append("instruction.md is empty")
        return hits

    # Full unified-diff body co-located with agent prompt.
    if "diff --git" in text and ("+++ " in text or "--- " in text):
        hits.append("instruction embeds unified-diff markers (diff --git)")

    sol = solution_patch or ""
    if not sol.strip():
        return hits

    # Whole-body inclusion
    compact_sol = re.sub(r"\s+", "", sol.strip())
    compact_text = re.sub(r"\s+", "", text)
    if len(compact_sol) >= min_marker_len and compact_sol in compact_text:
        hits.append("instruction embeds full solution.patch body")
        return hits

    # Distinct gold + lines long enough to be unique answer-key content
    markers: list[str] = []
    for ln in sol.splitlines():
        if not ln.startswith("+") or ln.startswith("+++"):
            continue
        body = ln[1:].strip()
        if len(body) < min_marker_len:
            continue
        if body.startswith(("import ", "from ", "package ", "//", "#", "/*", "*")):
            continue
        markers.append(body)
        if len(markers) >= 8:
            break
    matched = 0
    for marker in markers:
        if marker in text:
            matched += 1
    if matched >= 2 or (matched >= 1 and len(markers) <= 2 and matched == len(markers)):
        hits.append(f"instruction embeds {matched} gold solution line marker(s)")
    return hits


def base_commit_is_object(repo: Path | str, sha: str) -> bool:
    """True when ``git rev-parse --verify <sha>^{commit}`` succeeds (VAL-HPACK-003)."""
    cleaned = (sha or "").strip()
    if not cleaned:
        return False
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", f"{cleaned}^{{commit}}"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def run_pre_artifacts_capture(
    workspace: Path | str,
    *,
    pre_artifacts_sh: str | None = None,
    base_commit: str | None = None,
    logs_dir: Path | str | None = None,
) -> PreArtifactsCaptureResult:
    """Execute pre_artifacts script against a local sol worktree (VAL-HPACK-004).

    Rewrites ``cd /app`` and ``/logs/artifacts`` so capture works offline without
    Docker. Returns byte size of produced model.patch.
    """
    from swe_factory.harbor.pre_artifacts import render_pre_artifacts_sh

    repo = Path(workspace)
    if not repo.is_dir():
        return PreArtifactsCaptureResult(
            ok=False,
            byte_size=0,
            patch_path=None,
            reasons=(f"workspace missing: {repo}",),
        )
    logs = Path(logs_dir) if logs_dir is not None else repo.parent / "pre_artifacts_logs"
    artifacts = logs / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    body = pre_artifacts_sh
    if body is None:
        body = render_pre_artifacts_sh(base_commit)
    # Localize paths: /app → workspace, /logs/artifacts → logs/artifacts
    localized = body.replace("cd /app || exit 0", f"cd {repo} || exit 0")
    localized = localized.replace("/logs/artifacts", str(artifacts))
    # Marker file path if present
    localized = localized.replace("/app/.harbor_base_commit", str(repo / ".harbor_base_commit"))

    script = logs / "run_pre_artifacts.sh"
    script.write_text(localized if localized.endswith("\n") else localized + "\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | 0o111)

    try:
        completed = subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
            env={**os.environ, "BASE_COMMIT": (base_commit or "").strip()},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PreArtifactsCaptureResult(
            ok=False,
            byte_size=0,
            patch_path=None,
            logs=str(exc),
            reasons=(f"pre_artifacts execution failed: {exc}",),
        )

    patch_path = artifacts / "model.patch"
    if not patch_path.is_file():
        return PreArtifactsCaptureResult(
            ok=False,
            byte_size=0,
            patch_path=None,
            logs=(completed.stdout or "") + (completed.stderr or ""),
            reasons=("model.patch missing after pre_artifacts",),
        )
    data = patch_path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    ok = len(data) > 0
    reasons: list[str] = []
    if not ok:
        reasons.append("model.patch is empty after solution capture")
    base_ref = ""
    for line in (completed.stdout or "").splitlines():
        if "base=" in line:
            # [pre_artifacts] base=<ref> captured N bytes
            m = re.search(r"base=(\S+)", line)
            if m:
                base_ref = m.group(1)
                break
    return PreArtifactsCaptureResult(
        ok=ok,
        byte_size=len(data),
        patch_path=patch_path,
        patch_text=text,
        base_ref=base_ref,
        logs=(completed.stdout or "") + (completed.stderr or ""),
        reasons=tuple(reasons),
    )


def _read_pack_identity(pack_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "task_id": pack_dir.name,
        "repository_url": "",
        "base_commit_hash": "",
        "schema_version": "",
        "environment_mode": "",
        "language": "",
        "source_track": "",
    }
    toml_path = pack_dir / "task.toml"
    if not toml_path.is_file():
        return out
    try:
        data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return out
    out["schema_version"] = str(data.get("schema_version") or "")
    meta = data.get("metadata") or {}
    if isinstance(meta, dict):
        out["repository_url"] = str(meta.get("repository_url") or "").strip()
        out["base_commit_hash"] = str(meta.get("base_commit_hash") or "").strip()
        out["language"] = str(meta.get("language") or "").strip()
        out["source_track"] = str(meta.get("source_track") or "").strip()
        if meta.get("task_id"):
            out["task_id"] = str(meta["task_id"]).strip() or out["task_id"]
    verifier = data.get("verifier") or {}
    if isinstance(verifier, dict):
        out["environment_mode"] = str(verifier.get("environment_mode") or "").strip()
    return out


def is_product_deepswe_dest(path: Path | str | None) -> bool:
    """True when *path* targets product deepswe_v1 (not hybrid archive / fixtures)."""
    if path is None:
        return False
    text = str(path).replace("\\", "/").lower()
    if any(marker in text for marker in _HYBRID_ARCHIVE_MARKERS):
        return False
    if "/datasets/harbor_v1" in text or text.endswith("datasets/harbor_v1"):
        return False
    if ("/datasets/v1" in text or text.endswith("datasets/v1")) and "deepswe" not in text:
        # boltons V1 fixture, not DeepSWE product
        return False
    return any(token in text for token in _PRODUCT_DEEPSWE_TOKENS)


def is_hybrid_or_motor_track(source_track: str | None) -> bool:
    """True when *source_track* is hybrid/motor/synthetic (not product real_pr)."""
    track = (source_track or "").strip().lower()
    if not track:
        return False
    if track == REAL_PR_SOURCE_TRACK:
        return False
    if track in HYBRID_SOURCE_TRACKS:
        return True
    return track.startswith("hybrid") or track.startswith("motor")


def assert_product_real_pr_export(
    *,
    source_track: str | None,
    repository_url: str = "",
    base_commit: str = "",
    dest: Path | str | None = None,
    copy_repo_into_environment: Path | str | None = None,
    force_product: bool = False,
    allow_hybrid: bool = False,
) -> None:
    """Fail closed when hybrid/motors try to land on product deepswe_v1 (VAL-RPACK-003).

    Raises :class:`RealPackError` with an explicit reason when:
    - dest is product ``datasets/deepswe_v1`` and track is hybrid/motor, or
    - force_product and track is not real_pr, or
    - hybrid motor bind (``copy_repo_into_environment``) is forced on product path.
    """
    track = (source_track or "").strip().lower()
    product = force_product or is_product_deepswe_dest(dest)
    if allow_hybrid and not product:
        return
    if not product and not force_product:
        # Non-product dest: only refuse explicit hybrid_bind on a forced product API.
        return

    if is_hybrid_or_motor_track(track) or track in HYBRID_SOURCE_TRACKS:
        raise RealPackError(
            "product deepswe_v1 export refuses hybrid/motor packaging "
            f"(source_track={source_track!r}); requires source_track=real_pr "
            f"(dest={dest!r})"
        )
    if track and track != REAL_PR_SOURCE_TRACK:
        raise RealPackError(
            "product deepswe_v1 export requires source_track=real_pr; "
            f"got {source_track!r} (dest={dest!r})"
        )
    if not track:
        raise RealPackError(
            "product deepswe_v1 export requires source_track=real_pr "
            f"(missing track; dest={dest!r})"
        )
    if copy_repo_into_environment is not None:
        raise RealPackError(
            "product deepswe_v1 export refuses hybrid_bind / "
            "copy_repo_into_environment motor tree; agent must be clone@SHA "
            f"(repository_url={repository_url!r}, dest={dest!r})"
        )
    if repository_url and not is_real_repository_url(repository_url):
        raise RealPackError(
            "product deepswe_v1 export requires real public HTTPS repository_url; "
            f"got {repository_url!r}"
        )
    if base_commit and not is_real_base_sha(base_commit):
        raise RealPackError(
            f"product deepswe_v1 export requires real 40-char base_commit_hash; got {base_commit!r}"
        )


def validate_real_harbor_pack(
    pack_dir: Path | str,
    *,
    workspace_repo: Path | str | None = None,
    multi_file_floor: int = MULTI_PRODUCT_FLOOR,
    require_real_pr_track: bool = False,
    require_clone_dockerfile: bool | None = None,
) -> RealPackValidationResult:
    """Run Harbor real-pack gates on an exported pack directory.

    Parameters
    ----------
    require_real_pr_track:
        When True (product DeepSWE real-PR path / VAL-RPACK-002), refuse packs
        whose ``source_track`` is not ``real_pr`` (including hybrid_curated).
    require_clone_dockerfile:
        When True, agent Dockerfile must be clone@SHA (VAL-RCLN). When None
        (default), clone gate runs automatically for ``source_track=real_pr``
        and for product force mode; hybrid_curated archive packs skip it so
        historical motor sub trees still pass HPACK url/sha/tree inventory.
    """
    root = Path(pack_dir)
    codes: list[str] = []
    reasons: list[str] = []
    identity = _read_pack_identity(root)
    task_id = str(identity["task_id"])
    source_track = str(identity.get("source_track") or "").strip()

    missing = tuple(verify_pack_tree(root))
    tree_complete = not missing
    if not tree_complete:
        codes.append("HPACK_TREE_INCOMPLETE")
        reasons.append(f"missing required relpaths: {list(missing)}")

    repository_url = str(identity["repository_url"])
    base_commit = str(identity["base_commit_hash"])
    schema_version = str(identity["schema_version"])
    env_mode = str(identity["environment_mode"])

    real_url_ok = is_real_repository_url(repository_url)
    real_sha_ok = is_real_base_sha(base_commit)
    if not real_url_ok:
        codes.append("HPACK_BAD_REPOSITORY_URL")
        reasons.append(f"repository_url not real public HTTPS remote: {repository_url!r}")
    if not real_sha_ok:
        codes.append("HPACK_BAD_BASE_COMMIT")
        reasons.append(f"base_commit_hash not real 40-char SHA: {base_commit!r}")
    if schema_version and not str(schema_version).startswith("1."):
        codes.append("HPACK_BAD_SCHEMA")
        reasons.append(f"schema_version must be 1.x, got {schema_version!r}")

    separate = env_mode == "separate"
    if not separate:
        codes.append("HPACK_SHARED_VERIFIER")
        reasons.append(f"verifier.environment_mode must be 'separate', got {env_mode!r}")

    track_ok = True
    track_norm = source_track.lower()
    if require_real_pr_track:
        if track_norm != REAL_PR_SOURCE_TRACK:
            track_ok = False
            codes.append("RPACK_BAD_SOURCE_TRACK")
            track_display = source_track if source_track else "(missing)"
            reasons.append(f"product pack requires source_track=real_pr; got {track_display!r}")
        if is_hybrid_or_motor_track(source_track):
            track_ok = False
            codes.append("RPACK_HYBRID_REFUSED")
            reasons.append(f"product export refuses hybrid/motor source_track={source_track!r}")

    base_object_ok: bool | None = None
    if workspace_repo is not None:
        base_object_ok = base_commit_is_object(workspace_repo, base_commit)
        if not base_object_ok:
            codes.append("HPACK_BASE_NOT_OBJECT")
            reasons.append(
                f"base_commit_hash {base_commit!r} is not a git object in {workspace_repo}"
            )

    # Held-out test patch + node ids (006 / RPACK-004)
    test_patch_path = root / "tests" / "test.patch"
    test_patch_text = ""
    test_patch_ok = False
    if test_patch_path.is_file():
        test_patch_text = test_patch_path.read_text(encoding="utf-8", errors="replace")
        test_patch_ok = bool(test_patch_text.strip())
    if not test_patch_ok:
        codes.append("HPACK_EMPTY_TEST_PATCH")
        reasons.append("tests/test.patch missing or empty")

    config_ok = False
    f2p_count = 0
    p2p_count = 0
    cfg_path = root / "tests" / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            f2p = [x for x in (cfg.get("f2p_node_ids") or []) if str(x).strip()]
            p2p = [x for x in (cfg.get("p2p_node_ids") or []) if str(x).strip()]
            f2p_count = len(f2p)
            p2p_count = len(p2p)
            base_cfg = str(cfg.get("base_commit") or "").strip()
            config_ok = f2p_count >= 1 and (not base_commit or base_cfg == base_commit)
            if f2p_count < 1:
                codes.append("HPACK_EMPTY_F2P")
                reasons.append("tests/config.json f2p_node_ids must be non-empty")
            if base_commit and base_cfg and base_cfg != base_commit:
                codes.append("HPACK_CONFIG_BASE_MISMATCH")
                reasons.append("tests/config.json base_commit mismatches metadata.base_commit_hash")
                config_ok = False
        except json.JSONDecodeError as exc:
            codes.append("HPACK_CONFIG_PARSE")
            reasons.append(f"tests/config.json parse failed: {exc}")
    else:
        codes.append("HPACK_MISSING_CONFIG")
        reasons.append("tests/config.json missing")

    # Multi-file solution (007 / RPACK-004)
    sol_path = root / "solution" / "solution.patch"
    sol_text = ""
    solution_files: tuple[str, ...] = ()
    multi_file_ok = False
    if sol_path.is_file():
        sol_text = sol_path.read_text(encoding="utf-8", errors="replace")
        solution_files = tuple(product_source_files(sol_text))
        multi_file_ok = len(solution_files) >= multi_file_floor
    if not multi_file_ok:
        codes.append("HPACK_MULTI_FILE")
        reasons.append(
            f"solution.patch must touch ≥{multi_file_floor} product files; "
            f"got {list(solution_files)}"
        )

    # Isolation (008)
    isol = scan_pack_agent_isolation(root)
    isolation_hits = list(isol.hits)
    env_dir = root / "environment"
    if env_dir.is_dir():
        isolation_hits.extend(scan_agent_context_forbidden(env_dir))
    # Deduplicate
    isolation_hits = list(dict.fromkeys(isolation_hits))
    isolation_clean = len(isolation_hits) == 0
    if not isolation_clean:
        codes.append("HPACK_ISOLATION")
        reasons.append(f"agent isolation hits: {isolation_hits}")

    # Real-PR clone@SHA / no-motor-COPY: product track or explicit require.
    if require_clone_dockerfile is None:
        require_clone = track_norm == REAL_PR_SOURCE_TRACK or require_real_pr_track
    else:
        require_clone = require_clone_dockerfile
    dockerfile_ok = True
    df_path = root / "environment" / "Dockerfile"
    if require_clone and df_path.is_file() and real_url_ok and real_sha_ok:
        df_text = df_path.read_text(encoding="utf-8", errors="replace")
        try:
            assert_real_pr_agent_dockerfile(
                df_text,
                repository_url=repository_url,
                base_commit=base_commit,
                source_track=REAL_PR_SOURCE_TRACK,
            )
        except RealPrDockerfileError as exc:
            dockerfile_ok = False
            codes.append("RCLN_MOTOR_OR_NO_CLONE")
            reasons.append(str(exc))
        else:
            if looks_motor_fixture_copy(df_text) and not dockerfile_has_clone_at_sha(df_text):
                dockerfile_ok = False
                codes.append("RCLN_MOTOR_COPY")
                reasons.append(
                    "product real_pr agent Dockerfile uses motor COPY fixture without clone@SHA"
                )
        # Product packs must not emboss a motor hybrid bind tree under environment/repo
        if (env_dir / "repo").is_dir() and require_real_pr_track:
            dockerfile_ok = False
            codes.append("RPACK_HYBRID_BIND_TREE")
            reasons.append(
                "product real_pr pack must not ship environment/repo motor bind tree "
                "(clone@SHA only)"
            )

    # Instruction (009)
    instr_path = root / "instruction.md"
    instr_text = (
        instr_path.read_text(encoding="utf-8", errors="replace") if instr_path.is_file() else ""
    )
    leak_hits = scan_instruction_gold_leak(instr_text, sol_text)
    instruction_ok = bool(instr_text.strip()) and not leak_hits
    if not instruction_ok:
        codes.append("HPACK_INSTRUCTION")
        reasons.append(f"instruction.md empty or gold-leak: {leak_hits or ['empty']}")

    ok = (
        tree_complete
        and real_url_ok
        and real_sha_ok
        and separate
        and test_patch_ok
        and config_ok
        and multi_file_ok
        and isolation_clean
        and instruction_ok
        and dockerfile_ok
        and track_ok
        and (base_object_ok is not False)
    )

    return RealPackValidationResult(
        ok=ok,
        task_id=task_id,
        pack_dir=str(root.resolve() if root.exists() else root),
        tree_complete=tree_complete,
        missing_relpaths=missing,
        real_url_ok=real_url_ok,
        real_sha_ok=real_sha_ok,
        repository_url=repository_url,
        base_commit_hash=base_commit,
        schema_version=schema_version,
        separate_verifier=separate,
        base_object_ok=base_object_ok,
        test_patch_ok=test_patch_ok,
        config_ok=config_ok,
        f2p_count=f2p_count,
        p2p_count=p2p_count,
        multi_file_ok=multi_file_ok,
        solution_files=solution_files,
        isolation_clean=isolation_clean,
        isolation_hits=tuple(isolation_hits),
        instruction_ok=instruction_ok,
        instruction_leak_hits=tuple(leak_hits),
        reason_codes=tuple(codes),
        reasons=tuple(reasons),
        details={
            "multi_file_floor": multi_file_floor,
            "source_track": source_track,
            "require_real_pr_track": require_real_pr_track,
            "require_clone_dockerfile": require_clone,
        },
    )


def assert_real_pack_spec(
    spec: HarborPackSpec,
    *,
    multi_file_floor: int = MULTI_PRODUCT_FLOOR,
    require_real_pr_track: bool = True,
) -> HarborPackSpec:
    """Validate a HarborPackSpec is eligible for real DeepSWE export (pre-write).

    Product default (``require_real_pr_track=True``) enforces VAL-RPACK-002:
    ``source_track=real_pr``, real url/sha, multi-file gold, held-out tests,
    clone@SHA Dockerfile. Historical hybrid inventory callers may set
    ``require_real_pr_track=False`` to skip the product track / clone force.
    """
    cleaned = validate_pack_spec(spec)
    md = cleaned.task_toml.metadata
    reasons: list[str] = []
    if not is_real_repository_url(md.repository_url):
        reasons.append(f"repository_url not real public remote: {md.repository_url!r}")
    if not is_real_base_sha(md.base_commit_hash):
        reasons.append(f"base_commit_hash not real 40-char SHA: {md.base_commit_hash!r}")
    if cleaned.task_toml.verifier.environment_mode != "separate":
        reasons.append("verifier.environment_mode must be 'separate'")
    track = str(md.source_track or "").strip()
    if require_real_pr_track:
        if track.lower() != REAL_PR_SOURCE_TRACK:
            reasons.append(
                "source_track must be 'real_pr' for product real pack export; "
                f"got {md.source_track!r}"
            )
        if is_hybrid_or_motor_track(track):
            reasons.append(f"product export refuses hybrid/motor source_track={md.source_track!r}")
    product = product_source_files(cleaned.solution_patch)
    if len(product) < multi_file_floor:
        reasons.append(
            "solution.patch multi-file product floor failed: "
            f"files={product} floor={multi_file_floor}"
        )
    if not cleaned.test_patch.strip():
        reasons.append("test_patch must be non-empty")
    if not cleaned.tests_config.f2p_node_ids:
        reasons.append("f2p_node_ids must be non-empty")
    leak = scan_instruction_gold_leak(cleaned.instruction_md, cleaned.solution_patch)
    if leak:
        reasons.append(f"instruction gold leak: {leak}")
    if not cleaned.instruction_md.strip():
        reasons.append("instruction_md must be non-empty")
    # VAL-RCLN / VAL-RPACK: product real_pr forbids motor-only COPY while claiming real URL.
    if require_real_pr_track or track.lower() == REAL_PR_SOURCE_TRACK:
        try:
            assert_real_pr_agent_dockerfile(
                cleaned.environment_dockerfile,
                repository_url=md.repository_url,
                base_commit=md.base_commit_hash,
                source_track=REAL_PR_SOURCE_TRACK,
            )
        except RealPrDockerfileError as exc:
            reasons.append(str(exc))
    if reasons:
        raise RealPackError("real pack gate failed: " + "; ".join(reasons))
    return cleaned


def export_real_harbor_pack(
    spec: HarborPackSpec,
    *,
    dest: Path | str,
    overwrite: bool = True,
    extra_environment_files: dict[str, str] | None = None,
    copy_repo_into_environment: Path | str | None = None,
    multi_file_floor: int = MULTI_PRODUCT_FLOOR,
    workspace_repo: Path | str | None = None,
    require_real_pr_track: bool = True,
    allow_hybrid: bool = False,
) -> RealHarborPackResult:
    """Export one Harbor pack and fail closed unless real/product gates pass.

    VAL-RPACK-001..005 / VAL-RCLN-002:
    - Product path requires ``source_track=real_pr`` and clone@SHA agent tree
    - ``copy_repo_into_environment`` hybrid_bind is always refused on product path
    - Multi-file gold + held-out tests + dual-run metadata required

    Offline motors keep :func:`export_harbor_pack` with file:// fixtures.
    """
    cleaned = assert_real_pack_spec(
        spec,
        multi_file_floor=multi_file_floor,
        require_real_pr_track=require_real_pr_track,
    )
    md = cleaned.task_toml.metadata
    # Explicit product refusal for hybrid motors forced onto product dest/API.
    try:
        assert_product_real_pr_export(
            source_track=md.source_track,
            repository_url=md.repository_url,
            base_commit=md.base_commit_hash,
            dest=dest,
            copy_repo_into_environment=copy_repo_into_environment,
            force_product=require_real_pr_track,
            allow_hybrid=allow_hybrid and not require_real_pr_track,
        )
    except RealPackError:
        raise
    if copy_repo_into_environment is not None:
        # Refuse hybrid_bind: motor fixture body + real remote metadata.
        raise RealPackError(
            "real_pr product export refuses copy_repo_into_environment "
            "(motor/hybrid bind); agent tree must come from git clone@SHA of "
            f"repository_url={md.repository_url!r}"
        )
    try:
        pack = export_harbor_pack(
            cleaned,
            dest=dest,
            overwrite=overwrite,
            extra_environment_files=extra_environment_files,
            copy_repo_into_environment=None,
        )
    except HarborExportError as exc:
        raise RealPackError(str(exc)) from exc

    validation = validate_real_harbor_pack(
        pack.pack_dir,
        workspace_repo=workspace_repo,
        multi_file_floor=multi_file_floor,
        require_real_pr_track=require_real_pr_track,
        require_clone_dockerfile=True if require_real_pr_track else None,
    )
    if not validation.ok:
        # Do not leave a non-certifiable pack as success for deepswe path.
        raise RealPackError(
            f"real pack validation failed for {pack.task_id}: {list(validation.reasons)}"
        )
    return RealHarborPackResult(pack=pack, validation=validation)


def write_real_harbor_export(
    specs: Sequence[HarborPackSpec],
    out_dir: Path | str,
    *,
    overwrite: bool = True,
    tasks_subdir: str = "tasks",
    repo_for: dict[str, Path | str] | None = None,
    multi_file_floor: int = MULTI_PRODUCT_FLOOR,
    require_real_pr_track: bool = True,
) -> tuple[Path, tuple[RealHarborPackResult, ...]]:
    """Write one or more real packs under ``out_dir/tasks/<id>/`` (all gated).

    Product dest ``datasets/deepswe_v1`` (or force ``require_real_pr_track``)
    refuses hybrid/motor tracks and hybrid_bind repos (VAL-RPACK-003).
    """
    base = Path(out_dir)
    if not specs:
        raise RealPackError("refusing empty real harbor export")
    product_dest = is_product_deepswe_dest(base)
    if product_dest or require_real_pr_track:
        for spec in specs:
            md = validate_pack_spec(spec).task_toml.metadata
            candidate_bind = (repo_for or {}).get(spec.task_id)
            assert_product_real_pr_export(
                source_track=md.source_track,
                repository_url=md.repository_url,
                base_commit=md.base_commit_hash,
                dest=base,
                copy_repo_into_environment=candidate_bind,
                force_product=True,
            )
    if base.exists() and overwrite:
        tasks_root = base / tasks_subdir
        if tasks_root.is_dir():
            import shutil

            shutil.rmtree(tasks_root)
    base.mkdir(parents=True, exist_ok=True)
    tasks_root = base / tasks_subdir
    tasks_root.mkdir(parents=True, exist_ok=True)
    repos = dict(repo_for or {})
    packs: list[RealHarborPackResult] = []
    force_product = bool(product_dest or require_real_pr_track)
    for spec in specs:
        repo_bind: Path | str | None = None if force_product else repos.get(spec.task_id)
        packs.append(
            export_real_harbor_pack(
                spec,
                dest=tasks_root / spec.task_id,
                overwrite=True,
                # Never pass hybrid bind on product realpath; export_real raises.
                copy_repo_into_environment=repo_bind,
                multi_file_floor=multi_file_floor,
                require_real_pr_track=force_product,
            )
        )
    manifest_path = base / "pack_manifest.json"
    payload: dict[str, Any] = {
        "count": len(packs),
        "task_ids": [p.task_id for p in packs],
        "required_relpaths": list(REQUIRED_PACK_RELPATHS),
        "schema_version_target": "1.1",
        "real_pack_gate": True,
        "require_real_pr_track": require_real_pr_track or product_dest,
        "product_surface": product_dest,
        "source_tracks": {
            p.task_id: (p.validation.details or {}).get("source_track", REAL_PR_SOURCE_TRACK)
            for p in packs
        },
        "multi_file_floor": multi_file_floor,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, tuple(packs)


__all__ = [
    "HYBRID_SOURCE_TRACKS",
    "MULTI_PRODUCT_FLOOR",
    "REAL_PR_SOURCE_TRACK",
    "PreArtifactsCaptureResult",
    "RealHarborPackResult",
    "RealPackError",
    "RealPackValidationResult",
    "assert_product_real_pr_export",
    "assert_real_pack_spec",
    "base_commit_is_object",
    "export_real_harbor_pack",
    "is_hybrid_or_motor_track",
    "is_product_deepswe_dest",
    "is_test_path",
    "product_source_files",
    "run_pre_artifacts_capture",
    "scan_instruction_gold_leak",
    "validate_real_harbor_pack",
    "write_real_harbor_export",
]
