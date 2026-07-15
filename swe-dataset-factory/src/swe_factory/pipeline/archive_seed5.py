"""Archive prior real_pr product seed (M13 N=5) before M14 live overwrite.

VAL-LSHIP-001: before any live-mine promote overwrites ``datasets/deepswe_v1``,
the prior honesty product seed is archived to
``datasets/deepswe_v1_seed5_archive/`` (idempotent). Hybrid remains under
``datasets/deepswe_v1_hybrid_archive/`` and is never folded into product N.

This step does **not** clear product; clearing/overwrite is reserved for
``run_ship_deepswe_real_pr`` **after** gate_audit dual-truth passes
(VAL-LSHIP-007).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from swe_factory.pipeline.archive_hybrid import count_task_packs, has_archive_evidence

DEFAULT_SOURCE = Path("datasets/deepswe_v1")
DEFAULT_SEED5_ARCHIVE = Path("datasets/deepswe_v1_seed5_archive")

_CORPUS_ROOT_NAMES: tuple[str, ...] = (
    "pack_manifest.json",
    "PROVENANCE.md",
    "report.md",
    "ship_summary.json",
    "ledger_summary.json",
    "oracle_evidence.json",
    "pier_evidence.json",
    "e2e_drip.jsonl",
    "gate_audit.jsonl",
    "mine_inventory.json",
    "PRODUCT_README.md",
)

Action = Literal[
    "copied",
    "already_archived",
    "source_missing_archive_ok",
    "noop_empty",
    "skipped_non_real_pr",
]


class ArchiveSeed5Error(RuntimeError):
    """Unrecoverable seed5 product archive failure."""


@dataclass
class ArchiveSeed5Result:
    """Outcome of one archive-seed5-deepswe invocation."""

    ok: bool
    action: Action
    source_dir: Path
    archive_dir: Path
    source_pack_count: int
    archive_pack_count: int
    archived_task_ids: list[str] = field(default_factory=list)
    has_pack_manifest: bool = False
    has_tasks: bool = False
    source_track: str = "real_pr"
    archive_report_path: Path | None = None
    archive_readme_path: Path | None = None
    product_cleared: bool = False
    reason: str = ""
    timestamp_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "source_dir": str(self.source_dir),
            "archive_dir": str(self.archive_dir),
            "source_pack_count": self.source_pack_count,
            "archive_pack_count": self.archive_pack_count,
            "archived_task_ids": list(self.archived_task_ids),
            "has_pack_manifest": self.has_pack_manifest,
            "has_tasks": self.has_tasks,
            "source_track": self.source_track,
            "archive_report_path": (
                str(self.archive_report_path) if self.archive_report_path else None
            ),
            "archive_readme_path": (
                str(self.archive_readme_path) if self.archive_readme_path else None
            ),
            "product_cleared": self.product_cleared,
            "reason": self.reason,
            "timestamp_utc": self.timestamp_utc,
            "product_surface": "datasets/deepswe_v1",
            "archive_surface": "datasets/deepswe_v1_seed5_archive",
            "hybrid_archive_separate": "datasets/deepswe_v1_hybrid_archive",
            "hybrid_claimed_as_product": False,
            "seed5_claimed_as_current_product": False,
        }


def _detect_product_track(root: Path) -> str:
    """Best-effort product track label for honesty docs."""
    manifest = root / "pack_manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            track = data.get("product_track") or data.get("source_track")
            if isinstance(track, str) and track.strip():
                return track.strip()
            if data.get("hybrid") is True:
                return "hybrid_curated"
            tracks = data.get("source_tracks")
            if isinstance(tracks, dict) and tracks:
                vals = {str(v) for v in tracks.values()}
                if len(vals) == 1:
                    return next(iter(vals))
    # Infer from task.toml samples
    tasks = root / "tasks"
    if tasks.is_dir():
        for child in sorted(tasks.iterdir()):
            toml = child / "task.toml"
            if not toml.is_file():
                continue
            text = toml.read_text(encoding="utf-8", errors="replace")
            if 'source_track = "real_pr"' in text or "source_track = 'real_pr'" in text:
                return "real_pr"
            if "hybrid_curated" in text:
                return "hybrid_curated"
    return "unknown"


def _write_seed5_docs(
    archive_dir: Path,
    *,
    source_dir: Path,
    pack_count: int,
    task_ids: list[str],
    action: Action,
    timestamp: str,
    source_track: str,
) -> tuple[Path, Path]:
    readme = archive_dir / "ARCHIVE_README.md"
    report = archive_dir / "archive_report.json"
    readme_text = f"""# Seed5 real_pr product archive (historical)

> **Not current product.** This tree freezes the **prior honesty product**
> seed (M13 real_pr N≈{pack_count}) before the M14 live-mine overwrite of
> ``datasets/deepswe_v1``. Hybrid motors remain under
> ``datasets/deepswe_v1_hybrid_archive/`` (never mixed here).

| Field | Value |
|---|---|
| Archive path | `{archive_dir.as_posix()}` |
| Source path at archive time | `{source_dir.as_posix()}` |
| Live product path (post M14 ship) | `datasets/deepswe_v1` |
| Pack count | **{pack_count}** |
| Source track | `{source_track}` (historical seed) |
| Archived (UTC) | `{timestamp}` |
| Last archive action | `{action}` |

## Honesty

- This is the **seed5 / prior product** archive — not hybrid, not live-mine N.
- Do **not** fold seed5 archive pack counts into live product N after M14 promote.
- Overwrite of product requires seed5 archive ok **and** dual-truth
  ``gate_audit`` pass (VAL-LSHIP-001 / VAL-LSHIP-007).

## Sample pack ids

{chr(10).join(f"- `{tid}`" for tid in task_ids[:20]) or "- _(none)_"}
"""
    readme.write_text(readme_text, encoding="utf-8")
    payload = {
        "ok": True,
        "action": action,
        "timestamp_utc": timestamp,
        "source_dir": str(source_dir.resolve()) if source_dir.exists() else str(source_dir),
        "archive_dir": str(archive_dir.resolve()),
        "pack_count": pack_count,
        "task_ids": task_ids,
        "source_track": source_track,
        "status": "historical_seed5_archive",
        "product_surface": "datasets/deepswe_v1",
        "archive_surface": "datasets/deepswe_v1_seed5_archive",
        "hybrid_archive_surface": "datasets/deepswe_v1_hybrid_archive",
        "seed5_claimed_as_current_product": False,
        "product_cleared_by_archive_step": False,
        "note": (
            "Prior real_pr product seed archived before M14 live overwrite. "
            "Not counted as live product N. Hybrid archive remains separate."
        ),
        "required_evidence": {
            "pack_manifest": (archive_dir / "pack_manifest.json").is_file(),
            "tasks": pack_count > 0,
            "archive_readme": True,
        },
    }
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return readme, report


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=False)


def archive_seed5_deepswe(
    *,
    source_dir: Path | str = DEFAULT_SOURCE,
    archive_dir: Path | str = DEFAULT_SEED5_ARCHIVE,
    force_recopy: bool = False,
    require_real_pr: bool = True,
) -> ArchiveSeed5Result:
    """Idempotently archive prior real_pr product seed → deepswe_v1_seed5_archive.

    Args:
        source_dir: Current product path (default ``datasets/deepswe_v1``).
        archive_dir: Destination (default ``datasets/deepswe_v1_seed5_archive``).
        force_recopy: Recopy even when archive already looks complete.
        require_real_pr: When True, refuse to archive hybrid_curated corpora as seed5.

    Returns:
        :class:`ArchiveSeed5Result`.
    """
    source = Path(source_dir)
    archive = Path(archive_dir)
    timestamp = datetime.now(UTC).isoformat()

    src_count, src_ids = count_task_packs(source)
    src_has = has_archive_evidence(source) or (
        source.is_dir() and (src_count > 0 or (source / "pack_manifest.json").is_file())
    )
    arch_count, arch_ids = count_task_packs(archive)
    arch_has = has_archive_evidence(archive)
    track = (
        _detect_product_track(source)
        if src_has
        else (_detect_product_track(archive) if arch_has else "real_pr")
    )

    # Archive already complete and source empty → ok.
    if arch_has and not src_has and src_count == 0:
        readme, report = _write_seed5_docs(
            archive,
            source_dir=source,
            pack_count=arch_count,
            task_ids=arch_ids,
            action="source_missing_archive_ok",
            timestamp=timestamp,
            source_track=track or "real_pr",
        )
        return ArchiveSeed5Result(
            ok=True,
            action="source_missing_archive_ok",
            source_dir=source,
            archive_dir=archive,
            source_pack_count=0,
            archive_pack_count=arch_count,
            archived_task_ids=arch_ids,
            has_pack_manifest=(archive / "pack_manifest.json").is_file(),
            has_tasks=arch_count > 0,
            source_track=track or "real_pr",
            archive_report_path=report,
            archive_readme_path=readme,
            reason=(
                "Seed5 archive already holds prior product packs; "
                "source empty or already overwritten."
            ),
            timestamp_utc=timestamp,
        )

    # Idempotent already_archived
    if arch_has and not force_recopy and arch_count > 0:
        need_refresh = src_count > arch_count and src_has
        if not need_refresh:
            readme, report = _write_seed5_docs(
                archive,
                source_dir=source,
                pack_count=arch_count,
                task_ids=arch_ids,
                action="already_archived",
                timestamp=timestamp,
                source_track=track or "real_pr",
            )
            return ArchiveSeed5Result(
                ok=True,
                action="already_archived",
                source_dir=source,
                archive_dir=archive,
                source_pack_count=src_count,
                archive_pack_count=arch_count,
                archived_task_ids=arch_ids,
                has_pack_manifest=(archive / "pack_manifest.json").is_file(),
                has_tasks=arch_count > 0,
                source_track=track or "real_pr",
                archive_report_path=report,
                archive_readme_path=readme,
                reason=(
                    f"Idempotent no-op: seed5 archive already contains prior "
                    f"product evidence (N={arch_count}). Hybrid archive separate."
                ),
                timestamp_utc=timestamp,
            )

    # Nothing to archive
    if not src_has and not arch_has:
        return ArchiveSeed5Result(
            ok=True,
            action="noop_empty",
            source_dir=source,
            archive_dir=archive,
            source_pack_count=0,
            archive_pack_count=0,
            reason="No product seed at source and seed5 archive empty; nothing to archive.",
            timestamp_utc=timestamp,
        )

    if require_real_pr and track == "hybrid_curated":
        # Do not misfile hybrid motors as seed5 (hybrid has its own archive).
        return ArchiveSeed5Result(
            ok=False,
            action="skipped_non_real_pr",
            source_dir=source,
            archive_dir=archive,
            source_pack_count=src_count,
            archive_pack_count=arch_count,
            archived_task_ids=arch_ids,
            source_track=track,
            reason=(
                "Source looks hybrid_curated; refuse seed5 real_pr archive. "
                "Use archive-hybrid-deepswe for hybrid motors."
            ),
            timestamp_utc=timestamp,
        )

    if not src_has or src_count == 0:
        raise ArchiveSeed5Error(
            f"Cannot archive seed5: source {source} has no pack_manifest/tasks "
            f"(pack_count={src_count}) and archive is incomplete."
        )

    archive.parent.mkdir(parents=True, exist_ok=True)
    staging = archive.parent / f".{archive.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        _copy_tree(source, staging)
        st_count, st_ids = count_task_packs(staging)
        if st_count <= 0 or not (staging / "pack_manifest.json").is_file():
            raise ArchiveSeed5Error(
                "Seed5 staging missing pack_manifest.json or tasks/ after copy "
                f"(pack_count={st_count})."
            )
        # Drop ephemeral work residue if any
        for noise in ("_work", "work", ".work"):
            p = staging / noise
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        if archive.exists():
            shutil.rmtree(archive)
        staging.rename(archive)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    arch_count, arch_ids = count_task_packs(archive)
    if not has_archive_evidence(archive):
        raise ArchiveSeed5Error(
            f"Post-copy seed5 verification failed under {archive}; "
            "pack_manifest/tasks evidence missing."
        )
    track = _detect_product_track(archive) or track or "real_pr"
    readme, report = _write_seed5_docs(
        archive,
        source_dir=source,
        pack_count=arch_count,
        task_ids=arch_ids,
        action="copied",
        timestamp=timestamp,
        source_track=track,
    )
    return ArchiveSeed5Result(
        ok=True,
        action="copied",
        source_dir=source,
        archive_dir=archive,
        source_pack_count=src_count,
        archive_pack_count=arch_count,
        archived_task_ids=arch_ids,
        has_pack_manifest=True,
        has_tasks=True,
        source_track=track,
        archive_report_path=report,
        archive_readme_path=readme,
        reason=(
            f"Copied prior product seed N={arch_count} from {source} → {archive}. "
            "Product path left intact until gate_audit + live promote. "
            "Hybrid archive remains separate (VAL-LSHIP-001)."
        ),
        timestamp_utc=timestamp,
    )


def require_seed5_archived(
    *,
    archive_dir: Path | str = DEFAULT_SEED5_ARCHIVE,
    min_packs: int = 1,
) -> ArchiveSeed5Result:
    """Refuse product overwrite when seed5 archive evidence is missing."""
    archive = Path(archive_dir)
    n, ids = count_task_packs(archive)
    if not has_archive_evidence(archive) or n < min_packs:
        raise ArchiveSeed5Error(
            f"product overwrite refuses missing seed5 archive under {archive} "
            f"(pack_count={n} < min={min_packs}); archive prior product first "
            "(VAL-LSHIP-001)"
        )
    return ArchiveSeed5Result(
        ok=True,
        action="already_archived",
        source_dir=DEFAULT_SOURCE,
        archive_dir=archive,
        source_pack_count=0,
        archive_pack_count=n,
        archived_task_ids=ids,
        has_pack_manifest=True,
        has_tasks=True,
        reason=f"seed5 archive ok N={n}",
        timestamp_utc=datetime.now(UTC).isoformat(),
    )


__all__ = [
    "DEFAULT_SEED5_ARCHIVE",
    "DEFAULT_SOURCE",
    "ArchiveSeed5Error",
    "ArchiveSeed5Result",
    "archive_seed5_deepswe",
    "require_seed5_archived",
]
