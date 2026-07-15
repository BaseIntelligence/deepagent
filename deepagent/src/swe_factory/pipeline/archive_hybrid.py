"""Archive hybrid ``datasets/deepagent_v1`` before real-PR product overwrite.

VAL-RPR-001 / VAL-RSHIP-001 (partial pre-ship): copy the sealed hybrid motor
corpus (``source_track=hybrid_curated``, N≈113) to
``datasets/deepagent_v1_hybrid_archive/`` as a complete restorable tree.

Rules:
- Never delete product packs without a verified archive first.
- Archive is idempotent: re-run is a no-op when the archive already holds
  pack_manifest + tasks evidence equivalent to (or superseding) the source.
- Product ``datasets/deepagent_v1`` is **not** cleared here; clearing/overwrite
  is reserved for the real-PR ship feature after archive ok.
- Archive docs must label the corpus historical/hybrid, never as current product.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

DEFAULT_SOURCE = Path("datasets/deepagent_v1")
DEFAULT_ARCHIVE = Path("datasets/deepagent_v1_hybrid_archive")

# Root files copied when present (plus the tasks/ tree).
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
)

Action = Literal[
    "copied",
    "already_archived",
    "source_missing_archive_ok",
    "noop_empty",
]


class ArchiveHybridError(RuntimeError):
    """Unrecoverable hybrid archive failure."""


@dataclass
class ArchiveHybridResult:
    """Outcome of one archive-hybrid-deepagent invocation."""

    ok: bool
    action: Action
    source_dir: Path
    archive_dir: Path
    source_pack_count: int
    archive_pack_count: int
    archived_task_ids: list[str] = field(default_factory=list)
    has_pack_manifest: bool = False
    has_tasks: bool = False
    archive_report_path: Path | None = None
    archive_readme_path: Path | None = None
    product_cleared: bool = False
    product_is_current_hybrid: bool = True
    hybrid_claimed_as_product: bool = False
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
            "archive_report_path": (
                str(self.archive_report_path) if self.archive_report_path else None
            ),
            "archive_readme_path": (
                str(self.archive_readme_path) if self.archive_readme_path else None
            ),
            "product_cleared": self.product_cleared,
            "product_is_current_hybrid": self.product_is_current_hybrid,
            "hybrid_claimed_as_product": self.hybrid_claimed_as_product,
            "reason": self.reason,
            "timestamp_utc": self.timestamp_utc,
            # Honesty machine fields for ship gates
            "product_surface": "datasets/deepagent_v1",
            "archive_surface": "datasets/deepagent_v1_hybrid_archive",
            "source_track_archived": "hybrid_curated",
            "current_product_claim": "none_after_archive_step",
        }


def count_task_packs(root: Path) -> tuple[int, list[str]]:
    """Return (count, sorted task ids) under ``root/tasks``."""
    tasks = root / "tasks"
    if not tasks.is_dir():
        return 0, []
    ids = sorted(p.name for p in tasks.iterdir() if p.is_dir() and not p.name.startswith("."))
    return len(ids), ids


def has_archive_evidence(root: Path) -> bool:
    """True when archive root holds pack_manifest and at least one task pack."""
    if not root.is_dir():
        return False
    manifest = root / "pack_manifest.json"
    n, _ = count_task_packs(root)
    return manifest.is_file() and n > 0


def inventory_corpus(root: Path) -> dict[str, Any]:
    """Lightweight hybrid corpus inventory (counts + flags only)."""
    n, ids = count_task_packs(root)
    manifest_path = root / "pack_manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
    hybrid_flag = None
    if isinstance(manifest, dict):
        hybrid_flag = manifest.get("hybrid")
        if hybrid_flag is None:
            hybrid_flag = manifest.get("source_track")
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "pack_count": n,
        "task_ids_sample": ids[:5],
        "task_ids_count": n,
        "has_pack_manifest": manifest_path.is_file(),
        "has_provenance": (root / "PROVENANCE.md").is_file(),
        "has_report": (root / "report.md").is_file(),
        "has_tasks": n > 0,
        "manifest_count": (int(manifest.get("count", 0)) if isinstance(manifest, dict) else None),
        "manifest_hybrid": hybrid_flag,
        "manifest_product_surface": (
            manifest.get("product_surface") if isinstance(manifest, dict) else None
        ),
    }


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy directory tree ``src`` → ``dst``, overwriting destination if present."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=False)


def _write_archive_docs(
    archive_dir: Path,
    *,
    source_dir: Path,
    pack_count: int,
    task_ids: list[str],
    action: Action,
    timestamp: str,
) -> tuple[Path, Path]:
    """Write ARCHIVE_README.md + archive_report.json under the archive root."""
    readme = archive_dir / "ARCHIVE_README.md"
    report = archive_dir / "archive_report.json"

    readme_text = f"""# Hybrid-archive DeepAgent corpus (historical)

> **Not product.** This tree is the **archived hybrid motor corpus**
> (``source_track=hybrid_curated``). It is **not** the current certified
> product surface and must never be claimed as ``real_pr`` or as the live
> ``datasets/deepagent_v1`` corpus after the real-PR rebaseline.

| Field | Value |
|---|---|
| Archive path | `{archive_dir.as_posix()}` |
| Source path at archive time | `{source_dir.as_posix()}` |
| Product path (post-archive) | `datasets/deepagent_v1` (**real_pr only after ship**) |
| Pack count | **{pack_count}** |
| Source track | `hybrid_curated` (historical) |
| Archived (UTC) | `{timestamp}` |
| Last archive action | `{action}` |

## Paths

- **Archive (this tree):** `datasets/deepagent_v1_hybrid_archive/`
- **Product (real-PR only after ship):** `datasets/deepagent_v1/`
- **Non-product fixtures:** `datasets/harbor_v1/`, `datasets/v1/`

## Evidence layout

```text
pack_manifest.json     # hybrid keep inventory
PROVENANCE.md          # one hybrid row per keep
report.md              # historical hybrid ship report
tasks/<pack_id>/       # full Harbor pack trees
ARCHIVE_README.md      # this file
archive_report.json    # machine-readable archive summary
```

## Honesty

- Hybrid motors are **historical archive debt**, not the Real-PR product.
- Do not fold archive pack counts into product N.
- Product rewrite / clear of `datasets/deepagent_v1` happens only **after** this
  archive is verified (ship feature). This archive step does **not** delete
  packs without archival evidence.

## Sample pack ids

{chr(10).join(f"- `{tid}`" for tid in task_ids[:10]) or "- _(none)_"}
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
        "source_track": "hybrid_curated",
        "status": "historical_archive",
        "product_surface": "datasets/deepagent_v1",
        "archive_surface": "datasets/deepagent_v1_hybrid_archive",
        "hybrid_claimed_as_current_product": False,
        "product_cleared_by_archive_step": False,
        "note": (
            "Hybrid corpus archived. Not current product. "
            "Real-PR packs replace datasets/deepagent_v1 only after ship."
        ),
        "required_evidence": {
            "pack_manifest": (archive_dir / "pack_manifest.json").is_file(),
            "tasks": pack_count > 0,
        },
    }
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return readme, report


def archive_hybrid_deepagent(
    *,
    source_dir: Path | str = DEFAULT_SOURCE,
    archive_dir: Path | str = DEFAULT_ARCHIVE,
    force_recopy: bool = False,
) -> ArchiveHybridResult:
    """Idempotently archive hybrid deepagent_v1 → deepagent_v1_hybrid_archive.

    Args:
        source_dir: Product hybrid path (default ``datasets/deepagent_v1``).
        archive_dir: Destination archive (default ``datasets/deepagent_v1_hybrid_archive``).
        force_recopy: Recopy source even when archive already looks complete.

    Returns:
        :class:`ArchiveHybridResult` with action + evidence flags.

    Raises:
        ArchiveHybridError: when source has hybrid content that cannot be
            both archived and verified.
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

    # Case 1: archive already complete and source empty / missing → ok.
    if arch_has and not src_has and src_count == 0:
        readme, report = _write_archive_docs(
            archive,
            source_dir=source,
            pack_count=arch_count,
            task_ids=arch_ids,
            action="source_missing_archive_ok",
            timestamp=timestamp,
        )
        return ArchiveHybridResult(
            ok=True,
            action="source_missing_archive_ok",
            source_dir=source,
            archive_dir=archive,
            source_pack_count=0,
            archive_pack_count=arch_count,
            archived_task_ids=arch_ids,
            has_pack_manifest=(archive / "pack_manifest.json").is_file(),
            has_tasks=arch_count > 0,
            archive_report_path=report,
            archive_readme_path=readme,
            product_cleared=False,
            product_is_current_hybrid=False,
            hybrid_claimed_as_product=False,
            reason=(
                "Archive already holds hybrid pack_manifest/tasks; "
                "source empty or missing (downstream may have cleared product)."
            ),
            timestamp_utc=timestamp,
        )

    # Case 2: archive complete, source still hybrid — default is already_archived.
    if arch_has and not force_recopy and arch_count > 0:
        # Prefer not to thrash a good archive; refresh docs only.
        # If source still has packs and count differs upward, recopy for completeness.
        need_refresh = src_count > arch_count and src_has
        if not need_refresh:
            readme, report = _write_archive_docs(
                archive,
                source_dir=source,
                pack_count=arch_count,
                task_ids=arch_ids,
                action="already_archived",
                timestamp=timestamp,
            )
            return ArchiveHybridResult(
                ok=True,
                action="already_archived",
                source_dir=source,
                archive_dir=archive,
                source_pack_count=src_count,
                archive_pack_count=arch_count,
                archived_task_ids=arch_ids,
                has_pack_manifest=(archive / "pack_manifest.json").is_file(),
                has_tasks=arch_count > 0,
                archive_report_path=report,
                archive_readme_path=readme,
                product_cleared=False,
                product_is_current_hybrid=src_count > 0,
                hybrid_claimed_as_product=False,
                reason=(
                    "Idempotent no-op: archive already contains hybrid pack "
                    f"evidence (N={arch_count}). Product not claimed as current "
                    "certified real_pr corpus by this archive step."
                ),
                timestamp_utc=timestamp,
            )

    # Case 3: nothing to archive and no archive yet → noop.
    if not src_has and not arch_has:
        return ArchiveHybridResult(
            ok=True,
            action="noop_empty",
            source_dir=source,
            archive_dir=archive,
            source_pack_count=0,
            archive_pack_count=0,
            archived_task_ids=[],
            has_pack_manifest=False,
            has_tasks=False,
            archive_report_path=None,
            archive_readme_path=None,
            product_cleared=False,
            product_is_current_hybrid=False,
            hybrid_claimed_as_product=False,
            reason="No hybrid corpus at source and archive empty; nothing to archive.",
            timestamp_utc=timestamp,
        )

    # Case 4: copy source → archive (requires source hybrid evidence).
    if not src_has or src_count == 0:
        raise ArchiveHybridError(
            f"Cannot archive: source {source} has no hybrid pack_manifest/tasks "
            f"evidence (pack_count={src_count}) and archive is incomplete."
        )

    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and force_recopy:
        # Recopy whole tree
        pass
    # Stage into temp sibling then replace for partial-safety.
    staging = archive.parent / f".{archive.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        _copy_tree(source, staging)
        # Drop live-only workdirs that are not corpus evidence (optional).
        # Always ensure pack_manifest + tasks are present post-copy.
        st_count, st_ids = count_task_packs(staging)
        if st_count <= 0 or not (staging / "pack_manifest.json").is_file():
            raise ArchiveHybridError(
                "Archive staging missing pack_manifest.json or tasks/ after copy "
                f"(pack_count={st_count}). Refusing incomplete archive."
            )
        if archive.exists():
            shutil.rmtree(archive)
        staging.rename(archive)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    arch_count, arch_ids = count_task_packs(archive)
    readme, report = _write_archive_docs(
        archive,
        source_dir=source,
        pack_count=arch_count,
        task_ids=arch_ids,
        action="copied",
        timestamp=timestamp,
    )

    if not has_archive_evidence(archive):
        raise ArchiveHybridError(
            f"Post-copy verification failed under {archive}; pack_manifest/tasks evidence missing."
        )

    return ArchiveHybridResult(
        ok=True,
        action="copied",
        source_dir=source,
        archive_dir=archive,
        source_pack_count=src_count,
        archive_pack_count=arch_count,
        archived_task_ids=arch_ids,
        has_pack_manifest=(archive / "pack_manifest.json").is_file(),
        has_tasks=arch_count > 0,
        archive_report_path=report,
        archive_readme_path=readme,
        product_cleared=False,
        product_is_current_hybrid=True,
        hybrid_claimed_as_product=False,
        reason=(
            f"Copied hybrid corpus N={arch_count} from {source} → {archive}. "
            "Product path left intact (clear/overwrite only in real ship). "
            "Hybrid not claimed as current real_pr product."
        ),
        timestamp_utc=timestamp,
    )


def archive_result_asdict(result: ArchiveHybridResult) -> dict[str, Any]:
    """JSON-friendly view used by CLI."""
    return result.to_dict()


__all__ = [
    "DEFAULT_ARCHIVE",
    "DEFAULT_SOURCE",
    "ArchiveHybridError",
    "ArchiveHybridResult",
    "archive_hybrid_deepagent",
    "archive_result_asdict",
    "count_task_packs",
    "has_archive_evidence",
    "inventory_corpus",
]
