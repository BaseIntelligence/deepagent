"""Hugging Face pack I/O for DeepAgent (M16).

Upload / pull Harbor pack corpus trees against dataset repo
``BaseIntelligence/deepagent`` using ``huggingface_hub``.

Contract (architecture / VAL-DHF-*):
- Validate pack schema before any push (task.toml, environment/, tests/, solution/).
- Live automation targets revision ``test`` only (do not force-push ``main``).
- Auth from ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` only; **never** log tokens.
- Fail closed on missing token (live upload) and on bad revision (pull).
- Corpus layout: ``pack_manifest.json`` + ``tasks/<task_id>/…``.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swe_factory.harbor.export_pack import REQUIRED_PACK_RELPATHS, verify_pack_tree

DEFAULT_HF_REPO_ID = "BaseIntelligence/deepagent"
DEFAULT_HF_REVISION = "test"
HF_PACKS_SCHEMA = "deepagent.hf_packs.v1"

# Optional corpus-level files allowed alongside tasks/ + pack_manifest.json
_OPTIONAL_CORPUS_FILES = (
    "PROVENANCE.md",
    "report.md",
    "ledger_summary.json",
    "oracle_evidence.json",
    "pier_evidence.json",
    "ship_summary.json",
    "PRODUCT_README.md",
)


class HfPacksError(RuntimeError):
    """Fail-closed HF pack I/O error (schema, auth, revision, empty corpus)."""


# ---------------------------------------------------------------------------
# Auth-safe constant messages (never interpolate raw Hub ``str(exc)``).
# User-facing CLI paths surface these constants so Hub HTTP text cannot leak
# tokens, request ids, or other provider detail into console / JSON logs.
# ---------------------------------------------------------------------------

MSG_TOKEN_MISSING = (
    "upload: HF_TOKEN / HUGGING_FACE_HUB_TOKEN missing (fail-closed; no network spam)"
)
MSG_UPLOAD_AUTH = (
    "upload: Hugging Face authentication failed "
    "(check HF_TOKEN / HUGGING_FACE_HUB_TOKEN and write access)"
)
MSG_PULL_AUTH = (
    "pull: Hugging Face authentication failed "
    "(check HF_TOKEN / HUGGING_FACE_HUB_TOKEN and read access)"
)
MSG_UPLOAD_REPO = "upload: Hugging Face dataset repository create/access failed"
MSG_UPLOAD_HUB = "upload: Hugging Face Hub push failed"
MSG_PULL_REVISION = "pull: revision not found (or inaccessible) on remote dataset"
MSG_PULL_HUB = "pull: Hugging Face Hub download failed"
MSG_HUB_DEPENDENCY = "huggingface_hub not installed; pip install 'huggingface_hub>=0.23'"


def _exc_class_name(exc: BaseException) -> str:
    """Best-effort exception class name (supports Hub error hierarchy)."""
    return type(exc).__name__


def _hub_exc_blob_lower(exc: BaseException) -> str:
    """Lowercased status+message blob used only for classification (never echoed)."""
    parts: list[str] = [_exc_class_name(exc), str(exc)]
    # HfHubHTTPError often carries .response.status_code
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status is not None:
            parts.append(str(status))
    status_attr = getattr(exc, "status_code", None)
    if status_attr is not None:
        parts.append(str(status_attr))
    return " ".join(parts).lower()


def is_auth_hub_failure(exc: BaseException) -> bool:
    """True when *exc* looks like HF auth / gate / unauthorized failure.

    Classification only — never used as a user-visible string source.
    """
    name = _exc_class_name(exc)
    if name in {
        "GatedRepoError",
        "LocalTokenNotFoundError",
        "DisabledRepoError",
    }:
        return True
    blob = _hub_exc_blob_lower(exc)
    auth_markers = (
        " 401",
        "401 ",
        "401:",
        " 403",
        "403 ",
        "403:",
        "unauthorized",
        "forbidden",
        "invalid token",
        "invalid credentials",
        "authentication",
        "not authenticated",
        "access denied",
        "permission denied",
        "gated repo",
        "gated repository",
        "token is required",
        "invalid username or password",
        "wrong credentials",
    )
    # status codes may appear bare; require common framing or known Hub HTTP types
    auth_type = name in {
        "HfHubHTTPError",
        "HTTPError",
        "RepositoryNotFoundError",  # HF often 401-masquerades as 404
        "GatedRepoError",
    }
    if ("401" in blob or "403" in blob) and (any(m in blob for m in auth_markers) or auth_type):
        return True
    return any(m.strip() in blob for m in auth_markers if m.strip())


def is_revision_not_found(exc: BaseException) -> bool:
    """True when *exc* indicates a missing/invalid HF revision/ref."""
    name = _exc_class_name(exc)
    if name in {"RevisionNotFoundError", "EntryNotFoundError"}:
        return True
    blob = _hub_exc_blob_lower(exc)
    return any(
        m in blob
        for m in (
            "revisionnotfound",
            "revision not found",
            "invalid ref",
            "does not exist on the server",
            "is not a valid git identifier",
        )
    )


def map_hub_failure(action: str, exc: BaseException, *, stage: str = "") -> str:
    """Map a Hub/network exception to a constant auth-safe message.

    Never interpolates ``str(exc)``. ``action`` is ``\"upload\"`` or ``\"pull\"``.
    """
    act = (action or "hub").strip().lower()
    if act not in {"upload", "pull"}:
        act = "upload"

    if is_auth_hub_failure(exc):
        return MSG_UPLOAD_AUTH if act == "upload" else MSG_PULL_AUTH

    if act == "pull" and is_revision_not_found(exc):
        return MSG_PULL_REVISION

    stage_l = (stage or "").strip().lower()
    if act == "upload" and stage_l in {"create_repo", "repo", "branch"}:
        # create_repo access failures that are not auth still stay constant
        return MSG_UPLOAD_REPO

    if act == "upload":
        return MSG_UPLOAD_HUB
    return MSG_PULL_HUB


def hub_error(action: str, exc: BaseException, *, stage: str = "") -> HfPacksError:
    """Build an :class:`HfPacksError` from a Hub failure without leaking Hub text."""
    return HfPacksError(map_hub_failure(action, exc, stage=stage))


@dataclass(frozen=True, slots=True)
class PackCorpusValidation:
    """Result of local Harbor pack corpus validation."""

    ok: bool
    task_ids: tuple[str, ...]
    pack_dirs: tuple[Path, ...]
    reasons: tuple[str, ...]
    root: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "task_ids": list(self.task_ids),
            "reasons": list(self.reasons),
            "root": str(self.root),
            "pack_count": len(self.task_ids),
        }


def resolve_hf_token(token: str | None = None) -> str | None:
    """Resolve HF auth token from explicit arg or env. Never prints the value."""
    if token is not None:
        cleaned = token.strip()
        return cleaned or None
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return None


def list_pack_dirs(local_root: Path | str) -> list[Path]:
    """Return sorted ``tasks/<id>`` pack directories (or a single-pack root)."""
    root = Path(local_root)
    tasks_dir = root / "tasks"
    if tasks_dir.is_dir():
        return sorted(p for p in tasks_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    # Flat single-pack layout (task.toml at root)
    if (root / "task.toml").is_file():
        return [root]
    return []


def validate_pack_corpus(local_root: Path | str) -> PackCorpusValidation:
    """Validate Harbor pack trees under a corpus root before any HF push.

    Requires each pack to have the product Harbor layout nodes:
    ``task.toml``, ``environment/``, ``tests/``, ``solution/`` plus the full
    ``REQUIRED_PACK_RELPATHS`` file set.
    """
    root = Path(local_root)
    reasons: list[str] = []
    if not root.is_dir():
        return PackCorpusValidation(
            ok=False,
            task_ids=(),
            pack_dirs=(),
            reasons=(f"source root missing: {root}",),
            root=root,
        )

    packs = list_pack_dirs(root)
    if not packs:
        if not (root / "tasks").is_dir() and not (root / "task.toml").is_file():
            reasons.append(f"no tasks/ under {root} and no task.toml at root")
        else:
            reasons.append(f"tasks/ under {root} is empty — refusing empty pack upload")
        return PackCorpusValidation(
            ok=False, task_ids=(), pack_dirs=(), reasons=tuple(reasons), root=root
        )

    task_ids: list[str] = []
    for pack in packs:
        tid = pack.name if pack != root else pack.name
        task_ids.append(tid)

        # Layout directories required by VAL-DHF / product contract
        for dname in ("environment", "tests", "solution"):
            if not (pack / dname).is_dir():
                reasons.append(f"missing {dname}/ in {tid}")

        if not (pack / "task.toml").is_file():
            reasons.append(f"missing task.toml in {tid}")

        missing = verify_pack_tree(pack)
        for rel in missing:
            reasons.append(f"missing {rel} in {tid}")

    # Dedupe reasons while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)

    return PackCorpusValidation(
        ok=not uniq,
        task_ids=tuple(task_ids),
        pack_dirs=tuple(packs),
        reasons=tuple(uniq),
        root=root,
    )


def build_pack_manifest(
    local_root: Path | str,
    pack_dirs: list[Path] | tuple[Path, ...] | None = None,
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    revision: str = DEFAULT_HF_REVISION,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build corpus ``pack_manifest`` payload keyed by discovered task_ids.

    Preserves *all* existing ship-manifest keys (band, identity, languages,
    multi_file, …) so product honesty metadata is not stripped on rewrite.
    HF-facing schema keys are layered on top / refreshed.
    """
    root = Path(local_root)
    packs = list(pack_dirs) if pack_dirs is not None else list_pack_dirs(root)
    task_ids = [p.name for p in packs]
    payload: dict[str, Any] = {}
    existing = root / "pack_manifest.json"
    if existing.is_file():
        try:
            prev = json.loads(existing.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                payload.update(prev)
        except (OSError, json.JSONDecodeError):
            pass
    # Refresh / overlay HF + integrity fields (task_id set is ground truth from tree).
    payload["schema"] = HF_PACKS_SCHEMA
    payload["ok"] = True
    payload["count"] = len(task_ids)
    payload["task_ids"] = task_ids
    payload["repo_id"] = repo_id
    payload["revision"] = revision
    payload["required_relpaths"] = list(REQUIRED_PACK_RELPATHS)
    payload["layout"] = "tasks/<task_id>/{task.toml,environment/,tests/,solution/}"
    if extra:
        payload.update(extra)
    return payload


def write_pack_manifest(
    local_root: Path | str,
    manifest: dict[str, Any] | None = None,
    **build_kwargs: Any,
) -> Path:
    """Write (or rewrite) ``pack_manifest.json`` under the corpus root."""
    root = Path(local_root)
    root.mkdir(parents=True, exist_ok=True)
    payload = manifest if manifest is not None else build_pack_manifest(root, **build_kwargs)
    path = root / "pack_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _redact_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip any accidental token/auth keys before returning to callers."""
    banned = {
        "token",
        "hf_token",
        "authorization",
        "auth",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "password",
        "api_key",
        "apikey",
    }
    return {k: v for k, v in payload.items() if k not in banned and k.lower() not in banned}


def _ensure_branch(api: Any, *, repo_id: str, revision: str) -> None:
    """Best-effort create revision/branch (ignore if already present)."""
    if revision in {"main", "master"}:
        return
    try:
        api.create_branch(repo_id=repo_id, branch=revision, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001 — branch may already exist
        msg = str(exc).lower()
        if "already exists" in msg or "409" in msg or "reference already exists" in msg:
            return
        # Other race/permission errors surfaces on upload_folder


def upload_packs(
    local_root: Path | str,
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    revision: str = DEFAULT_HF_REVISION,
    token: str | None = None,
    dry_run: bool = False,
    allow_empty: bool = False,
    api: Any | None = None,
    commit_message: str | None = None,
) -> dict[str, Any]:
    """Validate + upload a local pack corpus to an HF dataset revision.

    Parameters
    ----------
    local_root:
        Corpus root containing ``tasks/<id>/`` (and preferably ``pack_manifest.json``).
    repo_id:
        HF dataset id (default ``BaseIntelligence/deepagent``).
    revision:
        Branch / revision to push. M16 live automation uses ``test``.
    token:
        Explicit token; if None, read from env. Required unless dry_run.
    dry_run:
        Schema-only path: no network, no token required.
    allow_empty:
        When False (default), refuse empty pack trees.
    api:
        Optional prebuilt HfApi (for unit tests).
    """
    root = Path(local_root)
    validation = validate_pack_corpus(root)
    if not validation.ok:
        if allow_empty and not validation.task_ids:
            raise HfPacksError(
                "upload: pack corpus empty and schema invalid: "
                + "; ".join(validation.reasons[:12])
            )
        raise HfPacksError("upload: pack schema invalid: " + "; ".join(validation.reasons[:12]))

    result: dict[str, Any] = {
        "ok": True,
        "action": "upload",
        "src": str(root),
        "repo_id": repo_id,
        "revision": revision,
        "dry_run": dry_run,
        "schema_ok": True,
        "pushed": False,
        "task_ids": list(validation.task_ids),
        "pack_count": len(validation.task_ids),
        "pack_manifest": str(root / "pack_manifest.json"),
        "message": "schema OK",
    }

    if dry_run:
        # Validate only — never rewrite product pack_manifest.json on dry-run.
        result["message"] = "schema OK; dry-run (no HF push)"
        return _redact_result(result)

    # Live push: refresh pack_manifest (preserving ship honesty keys) before upload.
    manifest = build_pack_manifest(
        root,
        validation.pack_dirs,
        repo_id=repo_id,
        revision=revision,
    )
    write_pack_manifest(root, manifest)

    auth = resolve_hf_token(token)
    if not auth:
        raise HfPacksError(MSG_TOKEN_MISSING)

    # Lazy import so offline unit paths without the wheel still import this module
    # when ``api`` is injected — but production requires huggingface_hub.
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - env issue
            raise HfPacksError(f"upload: {MSG_HUB_DEPENDENCY}") from exc
        client = HfApi(token=auth)
    else:
        client = api

    try:
        client.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        # Repo may already exist with different visibility; still attempt upload.
        # Classification uses message for "already exists" only — never re-echoed.
        existing_blob = _hub_exc_blob_lower(exc)
        if (
            "already" not in existing_blob
            and "409" not in existing_blob
            and "exist" not in existing_blob
        ):
            raise hub_error("upload", exc, stage="create_repo") from exc

    _ensure_branch(client, repo_id=repo_id, revision=revision)

    msg = commit_message or (
        f"deepagent upload: {len(validation.task_ids)} packs → {repo_id}@{revision}"
    )
    # Mirror local corpus exactly: remove remote-only pack trees / evidence
    # left by prior waves (e.g. curated prod_hard_keep drops misalign/solve-all).
    try:
        commit = client.upload_folder(
            folder_path=str(root),
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            commit_message=msg,
            delete_patterns=["*"],
        )
    except Exception as exc:  # noqa: BLE001
        raise hub_error("upload", exc, stage="upload_folder") from exc

    result["pushed"] = True
    result["message"] = "upload_folder complete"
    result["commit"] = str(commit) if commit is not None else None
    return _redact_result(result)


def upload_pack_tree(
    *,
    src: Path | str,
    repo_id: str = DEFAULT_HF_REPO_ID,
    revision: str = DEFAULT_HF_REVISION,
    token: str | None = None,
    dry_run: bool = False,
    api: Any | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """CLI-facing alias of :func:`upload_packs` (keyword ``src``)."""
    return upload_packs(
        local_root=src,
        repo_id=repo_id,
        revision=revision,
        token=token,
        dry_run=dry_run,
        api=api,
    )


def _materialized_pack_ids(out: Path) -> list[str]:
    """Discover task_ids under a brought-local corpus (manifest preferred)."""
    manifest_path = out / "pack_manifest.json"
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                ids = data.get("task_ids")
                if isinstance(ids, list) and ids:
                    return [str(x) for x in ids]
                # identity map fallback
                identity = data.get("identity")
                if isinstance(identity, dict) and identity:
                    return sorted(str(k) for k in identity)
                count = data.get("count")
                packs = list_pack_dirs(out)
                if packs and (count is None or int(count) == len(packs)):
                    return [p.name for p in packs]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return [p.name for p in list_pack_dirs(out)]


def pull_packs(
    out_dir: Path | str,
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    revision: str = DEFAULT_HF_REVISION,
    token: str | None = None,
    dry_run: bool = False,
    snapshot_fn: Any | None = None,
    require_complete_layout: bool = True,
) -> dict[str, Any]:
    """Download a pack corpus from HF into ``out_dir``.

    Fails closed on missing / bad revision (non-empty error, no silent OK).
    After pull, verifies that at least one pack has the Harbor layout nodes.
    """
    out = Path(out_dir)
    rev = (revision or DEFAULT_HF_REVISION).strip()
    if not rev:
        raise HfPacksError("pull: revision/branch required (documented: main | test)")

    result: dict[str, Any] = {
        "ok": True,
        "action": "pull",
        "out": str(out),
        "repo_id": repo_id,
        "revision": rev,
        "dry_run": dry_run,
        "pulled": False,
        "task_ids": [],
        "pack_count": 0,
        "message": "",
    }

    if dry_run:
        result["message"] = f"would download {repo_id}@{rev} → {out}"
        return _redact_result(result)

    auth = resolve_hf_token(token)
    # Public datasets may allow anonymous pull; still use token when present.

    if snapshot_fn is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover
            raise HfPacksError(f"pull: {MSG_HUB_DEPENDENCY}") from exc
        download = snapshot_download
    else:
        download = snapshot_fn

    out.mkdir(parents=True, exist_ok=True)
    try:
        path = download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=rev,
            local_dir=str(out),
            token=auth,
        )
    except Exception as exc:  # noqa: BLE001
        # Auth first (constant message; never re-echo Hub HTTP body).
        if is_auth_hub_failure(exc):
            raise hub_error("pull", exc, stage="snapshot_download") from exc
        # VAL-DHF-004: missing/invalid revision → constant pull message.
        blob = _hub_exc_blob_lower(exc)
        if is_revision_not_found(exc) or any(
            m in blob
            for m in (
                "revision",
                "not found",
                "404",
                "does not exist",
                "invalid ref",
            )
        ):
            raise HfPacksError(MSG_PULL_REVISION) from exc
        raise hub_error("pull", exc, stage="snapshot_download") from exc

    task_ids = _materialized_pack_ids(out)
    if not task_ids:
        raise HfPacksError(
            f"pull: no packs materialized under {out} from {repo_id}@{rev} "
            "(empty or unexpected layout)"
        )

    if require_complete_layout:
        # At least one complete pack (VAL-DHF-003)
        complete = 0
        for tid in task_ids:
            pack = out / "tasks" / tid
            if pack == out / "tasks" / tid and not pack.is_dir():
                # single-pack / flat edge
                pack = out / tid if (out / tid).is_dir() else out
            if not pack.is_dir():
                continue
            has_core = (
                (pack / "task.toml").is_file()
                and (pack / "environment").is_dir()
                and (pack / "tests").is_dir()
                and (pack / "solution").is_dir()
            )
            if has_core:
                complete += 1
        if complete < 1:
            raise HfPacksError(
                f"pull: no complete Harbor packs (task.toml + environment/ + "
                f"tests/ + solution/) under {out} from {repo_id}@{rev}"
            )

    result["pulled"] = True
    result["path"] = str(path)
    result["task_ids"] = task_ids
    result["pack_count"] = len(task_ids)
    result["message"] = "snapshot_download complete"
    result["ok"] = True
    return _redact_result(result)


def pull_pack_tree(
    *,
    out: Path | str,
    repo_id: str = DEFAULT_HF_REPO_ID,
    revision: str = DEFAULT_HF_REVISION,
    token: str | None = None,
    dry_run: bool = False,
    snapshot_fn: Any | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """CLI-facing alias of :func:`pull_packs` (keyword ``out``)."""
    return pull_packs(
        out_dir=out,
        repo_id=repo_id,
        revision=revision,
        token=token,
        dry_run=dry_run,
        snapshot_fn=snapshot_fn,
    )


def staging_copy_subset(
    source_root: Path | str,
    dest_root: Path | str,
    task_ids: list[str],
) -> Path:
    """Copy a subset of packs (plus refreshed manifest) for small live smokes."""
    src = Path(source_root)
    dest = Path(dest_root)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tasks_src = src / "tasks"
    tasks_dest = dest / "tasks"
    tasks_dest.mkdir(parents=True, exist_ok=True)
    for tid in task_ids:
        module = tasks_src / tid
        if not module.is_dir():
            raise HfPacksError(f"staging: pack {tid} missing under {tasks_src}")
        shutil.copytree(module, tasks_dest / tid)
    validation = validate_pack_corpus(dest)
    if not validation.ok:
        raise HfPacksError("staging: subset schema invalid: " + "; ".join(validation.reasons[:8]))
    write_pack_manifest(
        dest,
        build_pack_manifest(dest, validation.pack_dirs, extra={"subset_of": str(src)}),
    )
    # Copy optional corpus docs when present
    for name in _OPTIONAL_CORPUS_FILES:
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, dest / name)
    return dest


__all__ = [
    "DEFAULT_HF_REPO_ID",
    "DEFAULT_HF_REVISION",
    "HF_PACKS_SCHEMA",
    "HfPacksError",
    "MSG_HUB_DEPENDENCY",
    "MSG_PULL_AUTH",
    "MSG_PULL_HUB",
    "MSG_PULL_REVISION",
    "MSG_TOKEN_MISSING",
    "MSG_UPLOAD_AUTH",
    "MSG_UPLOAD_HUB",
    "MSG_UPLOAD_REPO",
    "PackCorpusValidation",
    "build_pack_manifest",
    "hub_error",
    "is_auth_hub_failure",
    "is_revision_not_found",
    "list_pack_dirs",
    "map_hub_failure",
    "pull_pack_tree",
    "pull_packs",
    "resolve_hf_token",
    "staging_copy_subset",
    "upload_pack_tree",
    "upload_packs",
    "validate_pack_corpus",
    "write_pack_manifest",
]
