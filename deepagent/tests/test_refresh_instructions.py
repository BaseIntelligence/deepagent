"""Instruction-only pack re-export / full prompts (VAL-DPRMPT-004)."""

from __future__ import annotations

import json
from pathlib import Path

from swe_factory.pipeline.refresh_instructions import (
    PROMPT_STYLE_FULL_V1,
    material_from_pack,
    refresh_pack_instruction,
    refresh_product_instructions,
    stamp_pack_manifest_prompt_style,
)
from swe_factory.pipeline.ship_real_pr import build_real_pr_agent_instruction

_FULL_SHA = "a" * 40
_LONG_BODY = (
    "This change restores multi-module package contracts that diverged when "
    "pricing and inventory reservation stopped coordinating. Callers that reserve "
    "stock no longer observe consistent totals after checkout mutations."
)

_SOLUTION = (
    "diff --git a/pkg/pricing.py b/pkg/pricing.py\n"
    "--- a/pkg/pricing.py\n"
    "+++ b/pkg/pricing.py\n"
    "@@ -1,3 +1,4 @@\n"
    " def total(items):\n"
    "-    return sum(items)\n"
    "+    return round(sum(items) * 0.9, 2)\n"
    "+\n"
    "diff --git a/pkg/inventory.py b/pkg/inventory.py\n"
    "--- a/pkg/inventory.py\n"
    "+++ b/pkg/inventory.py\n"
    "@@ -1,2 +1,3 @@\n"
    " def reserve(sku, qty):\n"
    "-    return qty\n"
    "+    return max(0, qty)\n"
    "+\n"
)

_TEST_PATCH = (
    "diff --git a/tests/test_checkout.py b/tests/test_checkout.py\n"
    "--- a/tests/test_checkout.py\n"
    "+++ b/tests/test_checkout.py\n"
    "@@ -0,0 +1,3 @@\n"
    "+def test_restored_totals():\n"
    "+    assert True\n"
    "+\n"
)

_STUB = (
    "Merged PR #77 on `https://github.com/owner/demo.git`: Restore multi-module "
    "checkout totals\n\n"
    "Restore multi-file product behavior against the held-out verifier suite. "
    "Affected sources include: pkg/pricing.py, pkg/inventory.py.\n"
    f"Base commit (immutable): `{_FULL_SHA}`.\n"
)


def _write_pack(root: Path, task_id: str = "realpr-demo-77") -> Path:
    pack = root / "tasks" / task_id
    pack.mkdir(parents=True)
    (pack / "instruction.md").write_text(_STUB, encoding="utf-8")
    (pack / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.1"',
                "[task]",
                f'name = "swe-factory/{task_id}"',
                "[metadata]",
                f'task_id = "{task_id}"',
                'language = "python"',
                'repository_url = "https://github.com/owner/demo.git"',
                f'base_commit_hash = "{_FULL_SHA}"',
                'source_track = "real_pr"',
                'license = "MIT"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    sol = pack / "solution"
    sol.mkdir()
    (sol / "solution.patch").write_text(_SOLUTION, encoding="utf-8")
    tests = pack / "tests"
    tests.mkdir()
    (tests / "test.patch").write_text(_TEST_PATCH, encoding="utf-8")
    return pack


def _write_materials(root: Path, task_id: str = "realpr-demo-77", *, body: str = "") -> Path:
    mat = root / task_id
    mat.mkdir(parents=True)
    meta = {
        "task_id": task_id,
        "repo": "owner/demo",
        "url": "https://github.com/owner/demo.git",
        "base": _FULL_SHA,
        "pr": 77,
        "title": "Restore multi-module checkout totals",
        "language": "python",
        "license": "MIT",
        "src": ["pkg/pricing.py", "pkg/inventory.py"],
        "tests": ["tests/test_checkout.py"],
        "source_hunk_count": 4,
    }
    if body:
        meta["body"] = body
    (mat / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (mat / "solution.patch").write_text(_SOLUTION, encoding="utf-8")
    (mat / "test.patch").write_text(_TEST_PATCH, encoding="utf-8")
    return mat


def test_refresh_pack_from_materials_body(tmp_path: Path) -> None:
    """Materials body → full multi-section instruction (no GitHub)."""
    packs = tmp_path / "test_n10"
    mats = tmp_path / "live_materials"
    pack = _write_pack(packs)
    _write_materials(mats, body=_LONG_BODY)

    one = refresh_pack_instruction(
        pack,
        materials_root=mats,
        fetch_github=False,
    )
    assert one.ok, one.error
    assert one.chars_after >= 400
    assert one.body_source == "meta"
    assert one.body_chars == len(_LONG_BODY)
    text = (pack / "instruction.md").read_text(encoding="utf-8")
    lower = text.lower()
    assert "context" in lower
    assert "pr description" in lower
    assert "deliverable" in lower
    assert _LONG_BODY[:40] in text
    assert not text.lstrip().startswith("Merged PR #77 on")


def test_refresh_pack_fetches_github_when_body_missing(tmp_path: Path) -> None:
    """Missing materials body triggers get_pull injection (VAL-DPRMPT-004)."""
    packs = tmp_path / "test_n10"
    mats = tmp_path / "live_materials"
    pack = _write_pack(packs)
    _write_materials(mats, body="")  # no body

    def fake_get_pull(repo: str, number: int) -> dict[str, str]:
        assert repo == "owner/demo"
        assert number == 77
        return {"body": _LONG_BODY, "title": "ok"}

    one = refresh_pack_instruction(
        pack,
        materials_root=mats,
        fetch_github=True,
        get_pull=fake_get_pull,
    )
    assert one.ok, one.error
    assert one.body_source == "github"
    # Body persisted onto materials meta.json
    meta = json.loads((mats / "realpr-demo-77" / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("body") == _LONG_BODY
    text = (pack / "instruction.md").read_text(encoding="utf-8")
    assert "PR description" in text
    assert _LONG_BODY[:30] in text


def test_refresh_root_stamps_manifest_prompt_style(tmp_path: Path) -> None:
    packs = tmp_path / "test_n10"
    mats = tmp_path / "live_materials"
    _write_pack(packs)
    _write_materials(mats, body=_LONG_BODY)
    (packs / "pack_manifest.json").write_text(
        json.dumps(
            {
                "pack_count": 1,
                "packs": [{"task_id": "realpr-demo-77", "certified": True}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = refresh_product_instructions(
        packs,
        materials_root=mats,
        fetch_github=False,
        stamp_manifest=True,
    )
    assert result.ok
    assert result.refreshed == 1
    assert result.prompt_style == PROMPT_STYLE_FULL_V1
    assert result.stamped_manifest is True
    man = json.loads((packs / "pack_manifest.json").read_text(encoding="utf-8"))
    assert man.get("prompt_style") == PROMPT_STYLE_FULL_V1
    assert man["packs"][0].get("prompt_style") == PROMPT_STYLE_FULL_V1


def test_refresh_gold_leak_fails_closed(tmp_path: Path) -> None:
    """If crafted instruction would leak gold, ok=False (builder itself is clean)."""
    packs = tmp_path / "test_n10"
    pack = _write_pack(packs)
    # Inject a unique long marker line into solution that build should not embed
    unique = "atomic_reserve_and_price_with_unique_marker_ZXQ99_never_in_prompt()"
    sol = (pack / "solution" / "solution.patch").read_text(encoding="utf-8")
    (pack / "solution" / "solution.patch").write_text(
        sol + f"+{unique}\n",
        encoding="utf-8",
    )
    mats = tmp_path / "live_materials"
    _write_materials(mats, body=_LONG_BODY)

    one = refresh_pack_instruction(pack, materials_root=mats, fetch_github=False)
    assert one.ok, one.error
    text = (pack / "instruction.md").read_text(encoding="utf-8")
    assert unique not in text
    assert "diff --git" not in text


def test_material_from_pack_empty_body_still_full(tmp_path: Path) -> None:
    packs = tmp_path / "test_n10"
    pack = _write_pack(packs)
    mat, body_res = material_from_pack(pack, fetch_github=False)
    assert body_res.source == "empty"
    text = build_real_pr_agent_instruction(mat)
    assert len(text) >= 400
    assert "context" in text.lower()


def test_stamp_manifest_helper_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    path = root / "pack_manifest.json"
    path.write_text(json.dumps({"packs": []}) + "\n", encoding="utf-8")
    assert stamp_pack_manifest_prompt_style(root) is True
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["prompt_style"] == PROMPT_STYLE_FULL_V1
