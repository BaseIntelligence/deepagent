"""Full DeepSWE-style real_pr agent prompts (VAL-DPRMPT + VAL-DSTYLE).

Stub "Merged PR #N … Restore multi-file…" product instructions are forbidden.
M18 agent-visible instructions are behavior-first DeepSWE rewrites with **no**
PR / repo / SHA provenance (private materials may still store body/meta).
"""

from __future__ import annotations

import json
from pathlib import Path

from swe_factory.harbor.real_pack import scan_instruction_gold_leak
from swe_factory.pipeline.deepswe_prompt import (
    find_provenance_fingerprints,
    has_deepswe_footer,
)
from swe_factory.pipeline.ship_real_pr import (
    RealPrMaterial,
    build_real_pr_agent_instruction,
    build_real_pr_pack_spec,
    load_real_pr_materials,
)
from swe_factory.producers.materialize_from_pr import materialize_merged_pr
from swe_factory.producers.pr_miner import MergedPR, PrFileChange, build_problem_statement

_FULL_SHA = "c" * 40
_MERGE_SHA = "d" * 40

_LONG_BODY = (
    "This change restores multi-module package contracts that diverged when "
    "pricing and inventory reservation stopped coordinating. Callers that reserve "
    "stock no longer observe consistent totals after checkout mutations. Please "
    "restore the original compositional behaviour without weakening unrelated "
    "pass_to_pass coverage or rewriting the held-out verifier suite."
)

_STUB_MARKERS = ("Restore multi-file product behavior against the held-out verifier suite.",)


def _source(path: str, patch: str | None = None) -> PrFileChange:
    body = patch or ("@@ -1,3 +1,4 @@\n def a():\n-    return 1\n+    return 2\n+\n")
    return PrFileChange(path=path, status="modified", patch=body)


def _test_file(path: str, patch: str | None = None) -> PrFileChange:
    body = patch or "@@ -0,0 +1,3 @@\n+def test_restored_totals():\n+    assert True\n+\n"
    return PrFileChange(path=path, status="added", patch=body)


def _merged_pr(*, body: str = _LONG_BODY, number: int = 77) -> MergedPR:
    return MergedPR(
        repo="owner/demo",
        number=number,
        title=f"Restore multi-module checkout totals (PR #{number})",
        body=body,
        base_commit=_FULL_SHA,
        merge_commit_sha=_MERGE_SHA,
        language="python",
        html_url=f"https://github.com/owner/demo/pull/{number}",
        files=(
            _source("pkg/pricing.py"),
            _source("pkg/inventory.py"),
            _test_file("tests/test_checkout.py"),
        ),
        license="MIT",
        merged_at="2026-03-04T00:00:00Z",
        source_hunk_count=4,
    )


def _material(
    *,
    body: str = _LONG_BODY,
    title: str = "Restore multi-module checkout totals",
    solution_extra: str = "",
) -> RealPrMaterial:
    solution = (
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
    if solution_extra:
        solution = solution + solution_extra
    return RealPrMaterial(
        task_id="realpr-demo-77",
        repository_url="https://github.com/owner/demo.git",
        base_commit=_FULL_SHA,
        language="python",
        license="MIT",
        pr_number=77,
        title=title,
        body=body,
        source_files=("pkg/pricing.py", "pkg/inventory.py"),
        test_files=("tests/test_checkout.py",),
        solution_patch=solution,
        test_patch=(
            "diff --git a/tests/test_checkout.py b/tests/test_checkout.py\n"
            "--- a/tests/test_checkout.py\n"
            "+++ b/tests/test_checkout.py\n"
            "@@ -0,0 +1,3 @@\n"
            "+def test_restored_totals():\n"
            "+    assert True\n"
            "+\n"
        ),
        materials_dir="",
        discovery_path="search",
        source_hunk_count=4,
    )


def test_materialize_persists_pr_body_on_meta_json(tmp_path: Path) -> None:
    """VAL-DPRMPT-001: materials meta.json stores non-empty PR body."""
    root = tmp_path / "live_materials"
    pr = _merged_pr(body=_LONG_BODY)
    task = materialize_merged_pr(pr, root, discovery_path="search")

    meta_path = root / task.task_id / "meta.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("body") == _LONG_BODY
    # MaterializedTask also surfaces body for callers.
    assert getattr(task, "body", "") == _LONG_BODY


def test_load_real_pr_materials_includes_body_field(tmp_path: Path) -> None:
    """VAL-DPRMPT-001: RealPrMaterial carries body from meta when present."""
    root = tmp_path / "live_materials"
    pr = _merged_pr(body=_LONG_BODY)
    materialize_merged_pr(pr, root, discovery_path="list_pulls")
    mats = load_real_pr_materials(root, limit=1)
    assert mats
    assert mats[0].body == _LONG_BODY
    assert mats[0].title
    # to_dict exposes body for reports / refresh tools
    dumped = mats[0].to_dict()
    assert dumped.get("body") == _LONG_BODY


def test_build_real_pr_agent_instruction_is_full_not_stub() -> None:
    """VAL-DSTYLE: DeepSWE-true behavior-first prompt, not PR-number stub."""
    mat = _material()
    text = build_real_pr_agent_instruction(mat, force_offline=True)

    # Floor: narrative richness for long-horizon agent consumption
    assert len(text.strip()) >= 200

    lower = text.lower()
    # DeepSWE register (not M17 Context/PR description scaffolding)
    assert find_provenance_fingerprints(text) == []
    assert has_deepswe_footer(text)
    assert "expected outcomes" in lower or "constraint" in lower
    assert "## context" not in lower
    assert "pr description" not in lower
    assert mat.base_commit not in text
    assert "github.com/owner/demo" not in lower

    # Substance from title/body without provenance
    assert "checkout" in lower or "pricing" in lower or "inventory" in lower
    assert "python" in lower or "module" in lower or "constraint" in lower

    # Must not be the short Merged-PR-only stub
    stub_hits = sum(1 for marker in _STUB_MARKERS if text.strip() == marker)
    assert stub_hits == 0
    assert not text.lstrip().startswith("Merged PR #")


def test_build_real_pr_pack_spec_uses_full_instruction_by_default() -> None:
    """VAL-DSTYLE: generate/ship path builder emits DeepSWE-true instruction."""
    mat = _material()
    spec = build_real_pr_pack_spec(mat, force_offline=True)
    instruction = spec.instruction_md
    assert len(instruction.strip()) >= 200
    assert find_provenance_fingerprints(instruction) == []
    assert has_deepswe_footer(instruction)
    assert mat.base_commit not in instruction
    # Still authentic real_pr track on private Harbor metadata
    assert spec.task_toml.metadata.source_track == "real_pr"
    assert "git clone" in (spec.environment_dockerfile or "").lower()


def test_full_instruction_gold_leak_scan_clean() -> None:
    """VAL-DPRMPT-003 / VAL-DSTYLE-005: prompts never embed solution.patch markers."""
    mat = _material()
    unique_gold = "return atomic_reserve_and_price_with_unique_marker_ZXQ99_never_in_prompt()\n"
    mat_leaky_sol = RealPrMaterial(
        task_id=mat.task_id,
        repository_url=mat.repository_url,
        base_commit=mat.base_commit,
        language=mat.language,
        license=mat.license,
        pr_number=mat.pr_number,
        title=mat.title,
        body=mat.body,
        source_files=mat.source_files,
        test_files=mat.test_files,
        solution_patch=(
            "diff --git a/pkg/pricing.py b/pkg/pricing.py\n"
            "--- a/pkg/pricing.py\n"
            "+++ b/pkg/pricing.py\n"
            "@@ -1,1 +1,2 @@\n"
            f"+{unique_gold}"
            "+return 42\n"
        ),
        test_patch=mat.test_patch,
        materials_dir=mat.materials_dir,
        discovery_path=mat.discovery_path,
        source_hunk_count=mat.source_hunk_count,
    )
    instruction = build_real_pr_agent_instruction(mat_leaky_sol, force_offline=True)
    hits = scan_instruction_gold_leak(instruction, mat_leaky_sol.solution_patch)
    assert hits == []
    assert "diff --git" not in instruction
    assert find_provenance_fingerprints(instruction) == []


def test_body_truncated_and_sanitized_in_prompt() -> None:
    """Huge / dirty body is distilled (not unbounded dump) and antifingerprint-clean."""
    huge = ("Paragraph about product behaviour. " * 200) + "TAIL_MARKER_END"
    mat = _material(body=huge)
    text = build_real_pr_agent_instruction(mat, force_offline=True)
    assert (
        "product behaviour" in text.lower()
        or "behavior" in text.lower()
        or "checkout" in text.lower()
    )
    assert len(text) < len(huge) + 2500
    assert len(text) >= 200
    assert find_provenance_fingerprints(text) == []


def test_build_problem_statement_still_usable_and_aligned() -> None:
    """Shared pr_miner helper remains gold-safe (mine path; may keep PR framing)."""
    pr = _merged_pr()
    prompt = build_problem_statement(pr=pr)
    # Mine-side helper may still mention PR; product path uses DeepSWE rewrite.
    assert pr.title in prompt or "Restore multi-module" in prompt
    assert "diff --git" not in prompt
    assert len(prompt) >= 100


def test_empty_body_still_full_multi_section() -> None:
    """When GitHub provides no body, still emit DeepSWE framing (not Merged-PR stub)."""
    mat = _material(body="")
    text = build_real_pr_agent_instruction(mat, force_offline=True)
    assert len(text.strip()) >= 200
    lower = text.lower()
    assert "expected outcomes" in lower or "constraint" in lower
    assert has_deepswe_footer(text)
    assert find_provenance_fingerprints(text) == []
    assert not text.lstrip().startswith("Merged PR #")
