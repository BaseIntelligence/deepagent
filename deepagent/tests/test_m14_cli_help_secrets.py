"""M14d honesty CLI: live-mine help discoverability, offline non-product, secrets clean.

Assertions:
- VAL-LX-004 offline unit/fixture non-regression green and never product N
- VAL-LX-005 no secret material (gho_/Bearer/OPENROUTER raw) in ship/mine trees
- VAL-LX-006 live-mine ship path discoverable from CLI help
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.export.leak_scan import scan_export_tree, scan_text_for_secrets
from swe_factory.pipeline.ship_real_pr import is_product_deepagent_dest
from swe_factory.producers.materialize_from_pr import is_fixture_materials_root

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCT = REPO_ROOT / "datasets" / "deepagent_v1"
FIXTURE_MATERIALS = REPO_ROOT / "fixtures" / "real_pr_ship"


def test_top_level_help_surfaces_live_mine_path() -> None:
    """VAL-LX-006: first-visit help shows live-mine / real-pr-pool --live / product dest."""
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0, res.output
    text = res.output
    lowered = text.lower()
    assert "real-pr-pool" in lowered
    assert "ship-deepagent" in lowered
    assert "live-mine" in lowered or "--live" in lowered
    # Product dest honesty + offline engineering-only
    assert "deepagent_v1" in lowered or "product" in lowered
    assert "fixture" in lowered
    assert "live-mine" in lowered
    assert "materialize-from-pr" in lowered or "materialize" in lowered


def test_real_pr_pool_help_surfaces_live_flag() -> None:
    """VAL-LX-006: real-pr-pool --help documents --live vs offline fixture (non-product)."""
    res = runner.invoke(app, ["real-pr-pool", "--help"])
    assert res.exit_code == 0, res.output
    text = res.output
    lowered = text.lower()
    assert "--live" in text or "--offline-fixture/--live" in text or "live" in lowered
    assert "offline" in lowered
    # Offline must be labeled engineering / not product N
    assert "product" in lowered
    assert "engineering" in lowered or "never product" in lowered or "not product" in lowered
    assert "discovery" in lowered or "search" in lowered or "list_pulls" in lowered


def test_ship_deepagent_help_surfaces_live_mine_and_product_honesty() -> None:
    """VAL-LX-006: ship-deepagent --help surfaces --live-mine + product dest refuse fixture."""
    res = runner.invoke(app, ["ship-deepagent", "--help"])
    assert res.exit_code == 0, res.output
    text = res.output
    lowered = text.lower()
    assert "live-mine" in lowered
    assert "deepagent_v1" in lowered
    assert "real_pr" in lowered
    assert "fixture" in lowered
    assert "materials" in lowered
    assert "docker" in lowered
    # Product honesty: refuse hybrid/fake called out in help narrative
    assert "hybrid" in lowered
    assert "min-packs" in lowered or "min_packs" in lowered or "minimum" in lowered


def test_pr_mine_help_lists_offline_and_live() -> None:
    res = runner.invoke(app, ["pr-mine", "--help"])
    assert res.exit_code == 0, res.output
    lowered = res.output.lower()
    assert "offline" in lowered
    assert "live" in lowered or "github" in lowered


def test_materialize_help_lists_live_materials_not_fixture() -> None:
    res = runner.invoke(app, ["materialize-from-pr", "--help"])
    assert res.exit_code == 0, res.output
    lowered = res.output.lower()
    assert "live_materials" in lowered or "live materials" in lowered
    assert "fixture" in lowered
    assert "inventory" in lowered


def test_offline_fixture_materials_never_product_dest() -> None:
    """VAL-LX-004: fixture shortlist is not the product dest materials authority."""
    assert is_fixture_materials_root(FIXTURE_MATERIALS) is True
    assert is_product_deepagent_dest("datasets/deepagent_v1") is True
    assert is_product_deepagent_dest(PRODUCT) is True
    # Offline unit dest stays non-product
    assert is_product_deepagent_dest("datasets/deepagent_v1_offline_only") is False
    assert is_product_deepagent_dest(REPO_ROOT / "datasets" / "deepagent_v1_ut_rpack_m11") is False


def test_product_n_excludes_offline_fixture_identity() -> None:
    """VAL-LX-004: certified product packs do not cite offline-fixture as product N."""
    if not PRODUCT.is_dir():
        return
    manifest = PRODUCT / "pack_manifest.json"
    if not manifest.is_file():
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    ids = data.get("task_ids") or data.get("ids") or []
    assert isinstance(ids, list)
    for tid in ids:
        s = str(tid).lower()
        assert "offline_fixture" not in s
        assert "hybrid" not in s
        assert "motor" not in s


def test_scan_text_detects_m14_secret_shapes() -> None:
    """VAL-LX-005 unit: scanner flags gho_ / Bearer / OPENROUTER raw."""
    assert scan_text_for_secrets("gho_ABCDEFGHIJKLMNOPQRSTUV", rel="x")
    assert scan_text_for_secrets("Bearer abcdefghijklmnopqrstuvwxyz01", rel="y")
    assert scan_text_for_secrets("OPENROUTER_API_KEY=sk-or-v1-" + ("b" * 20), rel="z")
    assert not scan_text_for_secrets("prefer export from gh auth token", rel="ok")


def test_product_ship_tree_secrets_clean() -> None:
    """VAL-LX-005: real product deepagent_v1 ship surface has no raw secrets."""
    if not PRODUCT.is_dir():
        return
    # Scan report/PROVENANCE/ledger-level artifacts + a pack sample
    text_targets: list[Path] = []
    for name in (
        "report.md",
        "PROVENANCE.md",
        "PRODUCT_README.md",
        "pack_manifest.json",
        "ship_summary.json",
        "oracle_evidence.json",
        "pier_evidence.json",
        "ledger_summary.json",
        "gate_audit.jsonl",
        "mine_inventory.json",
    ):
        p = PRODUCT / name
        if p.is_file():
            text_targets.append(p)
    findings: list[str] = []
    for path in text_targets:
        text = path.read_text(encoding="utf-8", errors="ignore")
        findings.extend(scan_text_for_secrets(text, rel=str(path.relative_to(REPO_ROOT))))
    assert findings == [], f"secret findings in product artifacts: {findings}"

    # Full tree scan (skips binaries; catches gho_/Bearer/OPENROUTER assignments)
    result = scan_export_tree(PRODUCT)
    # solution.patch + gold co-location under tasks/*/solution is expected Harbor layout
    # and is NOT agent-visible silver; filter pure solution-potency forbids that predate this
    # wave. VAL-LX-005 cares about token/keys only, so re-filter findings to secrets.
    secret_findings = [
        f
        for f in result.findings
        if "secret" in f.lower()
        or "bearer" in f.lower()
        or "gho_" in f.lower()
        or "api" in f.lower()
        or "key" in f.lower()
        or "token" in f.lower()
    ]
    assert secret_findings == [], f"secret findings: {secret_findings}"


def test_live_materials_and_candidates_secrets_clean() -> None:
    """VAL-LX-005: candidates/live materials evidence sample free of raw tokens."""
    roots = [
        REPO_ROOT / "datasets" / "live_materials",
        REPO_ROOT / "datasets" / "real_pr_pool",
        REPO_ROOT / "datasets" / "real_pr_pool_live_m14",
    ]
    sample_names = (
        "inventory.json",
        "candidates.jsonl",
        "real_pr_pool_report.json",
        "candidates/candidates.jsonl",
    )
    findings: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for name in sample_names:
            p = root / name
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            findings.extend(scan_text_for_secrets(text, rel=str(p.relative_to(REPO_ROOT))))
        # Also sample a few meta/json files one level deep
        for path in sorted(root.rglob("*.json"))[:30]:
            if path.stat().st_size > 200_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            findings.extend(scan_text_for_secrets(text, rel=str(path.relative_to(REPO_ROOT))))
    assert findings == [], f"secret findings in candidates/materials: {findings}"
