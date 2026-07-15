"""Prior real_pr product seed5 archive (VAL-LSHIP-001)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.pipeline.archive_seed5 import (
    archive_seed5_deepswe,
    require_seed5_archived,
)

runner = CliRunner()


def _seed_real_pr_tree(root: Path, *, n_packs: int = 5) -> list[str]:
    tasks = root / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    for i in range(1, n_packs + 1):
        tid = f"realpr-demo-{i}"
        pack = tasks / tid
        pack.mkdir(parents=True, exist_ok=True)
        (pack / "task.toml").write_text(
            'schema_version = "1.1"\n'
            f'[metadata]\ntask_id = "{tid}"\n'
            'source_track = "real_pr"\n'
            'repository_url = "https://github.com/example/repo.git"\n'
            f'base_commit_hash = "{"b" * 40}"\n',
            encoding="utf-8",
        )
        (pack / "instruction.md").write_text("demo real_pr pack\n", encoding="utf-8")
        (pack / "solution").mkdir()
        (pack / "solution" / "solution.patch").write_text(
            "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-a\n+b\n",
            encoding="utf-8",
        )
        (pack / "tests").mkdir()
        (pack / "tests" / "test.patch").write_text(
            "diff --git a/tests/t.py b/tests/t.py\n",
            encoding="utf-8",
        )
        ids.append(tid)
    manifest = {
        "count": n_packs,
        "product_surface": "datasets/deepswe_v1",
        "product_track": "real_pr",
        "task_ids": ids,
        "hybrid_claimed_as_product": False,
    }
    (root / "pack_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "PROVENANCE.md").write_text(
        "# PROVENANCE — real_pr seed\n\n"
        + "\n".join(f"| `{tid}` | real_pr |" for tid in ids)
        + "\n",
        encoding="utf-8",
    )
    (root / "report.md").write_text(
        f"# DeepSWE seed report\n- Certified packs: **{n_packs}** real_pr\n",
        encoding="utf-8",
    )
    return ids


def test_archive_seed5_copies_and_idempotent(tmp_path: Path) -> None:
    src = tmp_path / "deepswe_v1"
    arch = tmp_path / "deepswe_v1_seed5_archive"
    ids = _seed_real_pr_tree(src, n_packs=5)

    r1 = archive_seed5_deepswe(source_dir=src, archive_dir=arch)
    assert r1.ok
    assert r1.action == "copied"
    assert r1.archive_pack_count == 5
    assert (arch / "pack_manifest.json").is_file()
    assert (arch / "ARCHIVE_README.md").is_file()
    assert (arch / "archive_report.json").is_file()
    assert (arch / "tasks" / ids[0]).is_dir()
    # Source left intact
    assert (src / "tasks" / ids[0]).is_dir()

    r2 = archive_seed5_deepswe(source_dir=src, archive_dir=arch)
    assert r2.ok
    assert r2.action == "already_archived"
    assert r2.archive_pack_count == 5

    require_seed5_archived(archive_dir=arch, min_packs=1)
    report = json.loads((arch / "archive_report.json").read_text(encoding="utf-8"))
    assert report["seed5_claimed_as_current_product"] is False
    assert report["pack_count"] == 5


def test_archive_seed5_cli(tmp_path: Path) -> None:
    src = tmp_path / "deepswe_v1"
    arch = tmp_path / "deepswe_v1_seed5_archive"
    _seed_real_pr_tree(src, n_packs=3)
    res = runner.invoke(
        app,
        [
            "archive-seed5-deepswe",
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
    assert payload["archive_pack_count"] == 3
    assert payload["hybrid_claimed_as_product"] is False
