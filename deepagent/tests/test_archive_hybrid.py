"""Hybrid archive of datasets/deepagent_v1 (VAL-RPR-001 / VAL-RSHIP-001)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.pipeline.archive_hybrid import (
    archive_hybrid_deepagent,
    count_task_packs,
    has_archive_evidence,
    inventory_corpus,
)

runner = CliRunner()


def _seed_hybrid_tree(root: Path, *, n_packs: int = 3) -> list[str]:
    """Write a minimal hybrid corpus skeleton with pack_manifest + tasks."""
    tasks = root / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    for i in range(1, n_packs + 1):
        tid = f"harbor-py-orders-v{i}-deepagent_hybrid_seed_{i}"
        pack = tasks / tid
        pack.mkdir(parents=True, exist_ok=True)
        (pack / "task.toml").write_text(
            f'schema_version = "1.1"\n'
            f'[metadata]\ntask_id = "{tid}"\n'
            f'source_track = "hybrid_curated"\n'
            f'repository_url = "https://github.com/example/repo.git"\n'
            f'base_commit_hash = "{"a" * 40}"\n',
            encoding="utf-8",
        )
        (pack / "instruction.md").write_text("fix hybrid pack\n", encoding="utf-8")
        (pack / "solution").mkdir()
        (pack / "solution" / "solution.patch").write_text(
            "diff --git a/a.py b/a.py\n", encoding="utf-8"
        )
        (pack / "tests").mkdir()
        (pack / "tests" / "test.patch").write_text("diff --git a/t.py b/t.py\n", encoding="utf-8")
        ids.append(tid)

    manifest = {
        "count": n_packs,
        "product_surface": "datasets/deepagent_v1",
        "hybrid": True,
        "task_ids": ids,
        "refuse_fake": True,
        "historical_fixtures_only": ["datasets/harbor_v1", "datasets/v1"],
    }
    (root / "pack_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "PROVENANCE.md").write_text(
        "# PROVENANCE — hybrid_curated\n\n"
        + "\n".join(f"| `{tid}` | hybrid_curated |" for tid in ids)
        + "\n",
        encoding="utf-8",
    )
    (root / "report.md").write_text(
        "# DeepAgent v1 ship report (hybrid historical)\n"
        f"- Certified packs: **{n_packs}** hybrid_curated\n",
        encoding="utf-8",
    )
    return ids


def test_count_and_evidence_helpers(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert count_task_packs(empty) == (0, [])
    assert has_archive_evidence(empty) is False

    src = tmp_path / "deepagent_v1"
    ids = _seed_hybrid_tree(src, n_packs=2)
    n, got = count_task_packs(src)
    assert n == 2
    assert got == sorted(ids)
    assert has_archive_evidence(src) is True
    inv = inventory_corpus(src)
    assert inv["pack_count"] == 2
    assert inv["has_pack_manifest"] is True
    assert inv["manifest_hybrid"] is True


def test_archive_hybrid_copies_manifest_and_tasks(tmp_path: Path) -> None:
    src = tmp_path / "deepagent_v1"
    arch = tmp_path / "deepagent_v1_hybrid_archive"
    ids = _seed_hybrid_tree(src, n_packs=4)

    result = archive_hybrid_deepagent(source_dir=src, archive_dir=arch)
    assert result.ok
    assert result.action == "copied"
    assert result.archive_pack_count == 4
    assert result.source_pack_count == 4
    assert result.product_cleared is False  # never delete without ship step
    assert result.hybrid_claimed_as_product is False
    assert result.has_pack_manifest is True
    assert result.has_tasks is True
    assert sorted(result.archived_task_ids) == sorted(ids)

    assert (arch / "pack_manifest.json").is_file()
    assert (arch / "PROVENANCE.md").is_file()
    assert (arch / "report.md").is_file()
    assert (arch / "ARCHIVE_README.md").is_file()
    assert (arch / "archive_report.json").is_file()
    for tid in ids:
        assert (arch / "tasks" / tid / "task.toml").is_file()

    # Source left intact (clear only in ship feature)
    assert count_task_packs(src)[0] == 4

    readme = (arch / "ARCHIVE_README.md").read_text(encoding="utf-8")
    assert "Not product" in readme or "not" in readme.lower()
    assert "hybrid" in readme.lower()
    assert "real_pr" in readme.lower()
    assert "deepagent_v1_hybrid_archive" in readme

    report = json.loads((arch / "archive_report.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["pack_count"] == 4
    assert report["source_track"] == "hybrid_curated"
    assert report["hybrid_claimed_as_current_product"] is False
    assert report["product_cleared_by_archive_step"] is False
    assert report["status"] == "historical_archive"


def test_archive_hybrid_idempotent(tmp_path: Path) -> None:
    src = tmp_path / "deepagent_v1"
    arch = tmp_path / "deepagent_v1_hybrid_archive"
    _seed_hybrid_tree(src, n_packs=3)

    first = archive_hybrid_deepagent(source_dir=src, archive_dir=arch)
    assert first.action == "copied"
    mtime1 = (arch / "pack_manifest.json").stat().st_mtime_ns

    second = archive_hybrid_deepagent(source_dir=src, archive_dir=arch)
    assert second.ok
    assert second.action == "already_archived"
    assert second.archive_pack_count == 3
    mtime2 = (arch / "pack_manifest.json").stat().st_mtime_ns
    # Manifest not recopied on idempotent re-run
    assert mtime1 == mtime2

    # Force recopy rebuilds archive
    third = archive_hybrid_deepagent(source_dir=src, archive_dir=arch, force_recopy=True)
    assert third.ok
    assert third.action == "copied"
    assert third.archive_pack_count == 3


def test_archive_survives_source_clear(tmp_path: Path) -> None:
    """If product already empty but archive holds hybrid, re-run is safe."""
    src = tmp_path / "deepagent_v1"
    arch = tmp_path / "deepagent_v1_hybrid_archive"
    _seed_hybrid_tree(src, n_packs=2)
    archive_hybrid_deepagent(source_dir=src, archive_dir=arch)

    # Simulate ship feature clearing product after verified archive
    shutil.rmtree(src)
    src.mkdir()

    result = archive_hybrid_deepagent(source_dir=src, archive_dir=arch)
    assert result.ok
    assert result.action == "source_missing_archive_ok"
    assert result.archive_pack_count == 2
    assert has_archive_evidence(arch)


def test_noop_empty_source_and_archive(tmp_path: Path) -> None:
    src = tmp_path / "deepagent_v1"
    arch = tmp_path / "deepagent_v1_hybrid_archive"
    src.mkdir()
    result = archive_hybrid_deepagent(source_dir=src, archive_dir=arch)
    assert result.ok
    assert result.action == "noop_empty"
    assert result.archive_pack_count == 0


def test_cli_archive_hybrid(tmp_path: Path) -> None:
    src = tmp_path / "deepagent_v1"
    arch = tmp_path / "deepagent_v1_hybrid_archive"
    _seed_hybrid_tree(src, n_packs=2)

    res = runner.invoke(
        app,
        [
            "archive-hybrid-deepagent",
            "--source",
            str(src),
            "--archive",
            str(arch),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["ok"] is True
    assert payload["action"] == "copied"
    assert payload["archive_pack_count"] == 2
    assert payload["hybrid_claimed_as_product"] is False
    assert payload["product_cleared"] is False
    assert payload["current_product_claim"] == "none_after_archive_step"
    assert (arch / "pack_manifest.json").is_file()
    assert count_task_packs(arch)[0] == 2

    # Idempotent CLI re-run
    res2 = runner.invoke(
        app,
        [
            "archive-hybrid-deepagent",
            "--source",
            str(src),
            "--archive",
            str(arch),
            "--json",
        ],
    )
    assert res2.exit_code == 0, res2.output
    payload2 = json.loads(res2.output)
    assert payload2["action"] == "already_archived"


def test_cli_help_lists_archive_command() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "archive-hybrid-deepagent" in res.output
    res2 = runner.invoke(app, ["archive-hybrid-deepagent", "--help"])
    assert res2.exit_code == 0
    assert "deepagent_v1_hybrid_archive" in res2.output
    assert "hybrid" in res2.output.lower()


def test_result_dict_honesty_fields(tmp_path: Path) -> None:
    src = tmp_path / "deepagent_v1"
    arch = tmp_path / "deepagent_v1_hybrid_archive"
    _seed_hybrid_tree(src, n_packs=1)
    result = archive_hybrid_deepagent(source_dir=src, archive_dir=arch)
    d = result.to_dict()
    assert d["hybrid_claimed_as_product"] is False
    assert d["source_track_archived"] == "hybrid_curated"
    assert "deepagent_v1_hybrid_archive" in d["archive_surface"]
    assert d["current_product_claim"] == "none_after_archive_step"
