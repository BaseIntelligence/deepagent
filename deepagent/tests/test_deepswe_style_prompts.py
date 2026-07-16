"""DeepSWE-true product prompts (VAL-DSTYLE-001..005).

Antifingerprint regex ban list + offline rewriter golden style fixtures inspired
by datacurve-ai/deep-swe (abs-module-cache-flags, adaptix-name-mapping-aliases).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_factory.harbor.real_pack import scan_instruction_gold_leak
from swe_factory.openrouter import ChatResult, ScriptedChatClient, TokenUsage
from swe_factory.pipeline.deepswe_prompt import (
    DEEPSWE_IMPORTANT_FOOTER,
    PROMPT_STYLE_DEEPSWE_V1,
    build_deepswe_style_instruction,
    build_offline_deepswe_instruction,
    find_provenance_fingerprints,
    has_deepswe_footer,
    persist_agent_instruction,
    read_cached_agent_instruction,
    sanitize_body_for_deepswe,
    style_ok,
)
from swe_factory.pipeline.ship_real_pr import (
    RealPrMaterial,
    build_real_pr_agent_instruction,
    build_real_pr_pack_spec,
)

_FULL_SHA = "c" * 40
_MERGE_SHA = "d" * 40

_LONG_BODY = (
    "This change restores multi-module package contracts that diverged when "
    "pricing and inventory reservation stopped coordinating.\n\n"
    "See also https://github.com/owner/demo/pull/77 and issue "
    "https://github.com/owner/demo/issues/12.\n\n"
    "- [ ] Fix pricing totals after reserve mutations\n"
    "- [x] Keep pass_to_pass inventory inventory.reserve callers green\n"
    "- Align checkout with inventory for multi-SKU carts\n"
    "Closes #12. Merged PR #77 already landed problem statement only.\n"
    f"Base pin was `{_FULL_SHA}` — agents must not see this hash.\n"
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

# Style golden snippets inspired by real deep-swe samples (behavior + outcomes).
_GOLDEN_ADAPTIX_LIKE = (
    "`name_mapping` can rename fields via `map` but cannot accept multiple "
    "alternative input keys for the same field. Add alias support.\n\n"
    "Expected outcomes\n"
    "1. Loading resolves primary key then ordered alias fallback.\n"
    "2. Multi-key conflicts raise ExtraFieldsLoadError.\n"
    "3. Explicit aliases equal to their own primary key error at creation.\n\n"
    "Constraints\n"
    "- Keep public entrypoints stable.\n\n"
    f"{DEEPSWE_IMPORTANT_FOOTER}\n"
)


def _material(
    *,
    body: str = _LONG_BODY,
    title: str = "Restore multi-module checkout totals (PR #77)",
    agent_instruction: str = "",
    materials_dir: str = "",
) -> RealPrMaterial:
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
        solution_patch=_SOLUTION,
        test_patch=_TEST_PATCH,
        materials_dir=materials_dir,
        discovery_path="search",
        source_hunk_count=4,
        agent_instruction=agent_instruction,
    )


# ---------------------------------------------------------------------------
# VAL-DSTYLE-001 antifingerprint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dirty",
    [
        "Merged PR #77 on github",
        "See PR #12 for more",
        "this pull request restores",
        "https://github.com/owner/demo/pull/77",
        "https://github.com/owner/demo/issues/12",
        "source track=real_pr",
        "source_track: real_pr",
        f"Base commit: `{_FULL_SHA}`",
        "Repository URL: https://github.com/owner/demo.git",
        "## Context\nYou are solving a mined PR",
        "## PR description\nraw dump",
        "clone https://github.com/owner/demo.git please",
        "git@github.com:owner/demo.git",
    ],
)
def test_antifingerprint_regex_hits_provenance(dirty: str) -> None:
    hits = find_provenance_fingerprints(dirty)
    assert hits, f"expected fingerprint hit for {dirty!r}"


def test_clean_deepswe_sample_register_has_no_fingerprints() -> None:
    assert find_provenance_fingerprints(_GOLDEN_ADAPTIX_LIKE) == []
    assert style_ok(_GOLDEN_ADAPTIX_LIKE)
    assert has_deepswe_footer(_GOLDEN_ADAPTIX_LIKE)


def test_sanitize_body_strips_links_and_pr_markers() -> None:
    cleaned = sanitize_body_for_deepswe(_LONG_BODY)
    assert "github.com" not in cleaned.lower()
    assert "PR #77" not in cleaned
    assert _FULL_SHA not in cleaned
    assert "pricing" in cleaned.lower() or "inventory" in cleaned.lower()


# ---------------------------------------------------------------------------
# VAL-DSTYLE-003 offline rewriter + VAL-DSTYLE-002 register
# ---------------------------------------------------------------------------


def test_offline_fallback_behavior_first_no_provenance() -> None:
    text = build_offline_deepswe_instruction(
        title="Restore multi-module checkout totals (PR #77)",
        body=_LONG_BODY,
        source_files=("pkg/pricing.py", "pkg/inventory.py"),
        language="python",
    )
    assert len(text.strip()) >= 200
    assert find_provenance_fingerprints(text) == []
    lower = text.lower()
    assert "expected outcomes" in lower
    assert "constraint" in lower
    assert "pr description" not in lower
    assert "## context" not in lower
    assert "merged pr" not in lower
    assert _FULL_SHA not in text
    assert "github.com" not in lower
    assert "owner/demo" not in text  # no raw repo path echo required
    assert has_deepswe_footer(text)
    assert DEEPSWE_IMPORTANT_FOOTER in text
    # Behavior content from title/body still present without PR markers
    assert "checkout" in lower or "pricing" in lower or "inventory" in lower


def test_build_deepswe_force_offline_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    text = build_deepswe_style_instruction(
        title="Add alias support to name_mapping",
        body=(
            "name_mapping can rename fields via map but cannot accept multiple "
            "alternative input keys.\n\n"
            "- Load-only aliases with first-wins-per-field\n"
            "- alias_style generates aliases per field\n"
        ),
        source_files=("src/adaptix/_internal/name_style.py",),
        language="python",
        force_offline=True,
        persist_cache=False,
    )
    assert find_provenance_fingerprints(text) == []
    assert "expected outcomes" in text.lower()
    assert has_deepswe_footer(text)
    assert style_ok(text)


def test_live_path_uses_injected_client_when_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-real")
    good = (
        "Restore multi-module checkout totals so reserve and pricing stay consistent.\n\n"
        "Expected outcomes\n"
        "1. Pricing totals remain consistent after inventory reservation.\n"
        "2. Unrelated inventory callers stay green.\n\n"
        "Constraints\n"
        "- Prefer multi-file product source edits only.\n\n"
        f"{DEEPSWE_IMPORTANT_FOOTER}\n"
    )
    client = ScriptedChatClient(
        responses=[
            ChatResult(
                model="x-ai/grok-4.5",
                text=good,
                usage=TokenUsage(1, 1, 2),
                request_id="r1",
                cost_usd=None,
                finish_reason="stop",
                raw_usage={},
            )
        ]
    )
    text = build_deepswe_style_instruction(
        title="Restore checkout (PR #77)",
        body=_LONG_BODY,
        source_files=("pkg/pricing.py",),
        language="python",
        client=client,
        force_offline=False,
        persist_cache=False,
        env={"OPENROUTER_API_KEY": "test-not-real"},
    )
    assert "Pricing totals remain consistent" in text
    assert find_provenance_fingerprints(text) == []
    assert has_deepswe_footer(text)


def test_live_dirty_response_falls_back_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-real")
    dirty = (
        f"## Context\nMerged PR #77 base {_FULL_SHA}\n"
        "https://github.com/owner/demo/pull/77\n\n"
        f"{DEEPSWE_IMPORTANT_FOOTER}\n"
    )
    client = ScriptedChatClient(
        responses=[
            ChatResult(
                model="x-ai/grok-4.5",
                text=dirty,
                usage=TokenUsage(1, 1, 2),
                request_id="r2",
                cost_usd=None,
                finish_reason="stop",
                raw_usage={},
            )
        ]
    )
    text = build_deepswe_style_instruction(
        title="Restore checkout totals (PR #77)",
        body=_LONG_BODY,
        source_files=("pkg/pricing.py", "pkg/inventory.py"),
        language="python",
        client=client,
        force_offline=False,
        persist_cache=False,
        env={"OPENROUTER_API_KEY": "test"},
    )
    # Offline fallback luck, no fingerprints
    assert find_provenance_fingerprints(text) == []
    assert "https://github.com" not in text
    assert _FULL_SHA not in text
    assert has_deepswe_footer(text)


# ---------------------------------------------------------------------------
# Cache agent_instruction (meta)
# ---------------------------------------------------------------------------


def test_cache_agent_instruction_on_materials_meta(tmp_path: Path) -> None:
    mat = tmp_path / "realpr-demo-77"
    mat.mkdir()
    (mat / "meta.json").write_text(
        json.dumps({"task_id": "realpr-demo-77", "body": _LONG_BODY}, indent=2) + "\n",
        encoding="utf-8",
    )
    instr = build_offline_deepswe_instruction(
        title="Restore checkout",
        body=_LONG_BODY,
        source_files=("pkg/pricing.py",),
        language="python",
    )
    assert persist_agent_instruction(mat, instr) is True
    meta = json.loads((mat / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("agent_instruction")
    assert meta.get("prompt_style") == PROMPT_STYLE_DEEPSWE_V1
    cached = read_cached_agent_instruction(mat)
    assert cached.strip() == instr.strip()

    # Second call uses cache and does not need LLM
    again = build_deepswe_style_instruction(
        title="IGNORED TITLE BECAUSE CACHE",
        body="IGNORED BODY",
        source_files=("pkg/other.py",),
        materials_dir=mat,
        force_offline=True,
        persist_cache=False,
    )
    assert again.strip() == instr.strip()


# ---------------------------------------------------------------------------
# Product path wiring (build_real_pr_*)
# ---------------------------------------------------------------------------


def test_product_instruction_no_provenance_via_pack_spec() -> None:
    mat = _material()
    # force_offline through env isolation on builder wrapper
    text = build_real_pr_agent_instruction(mat, force_offline=True)
    assert find_provenance_fingerprints(text) == []
    assert has_deepswe_footer(text)
    assert "expected outcomes" in text.lower() or style_ok(text)
    # M17 scaffolding gone
    assert "## Context" not in text
    assert "## PR description" not in text
    assert mat.base_commit not in text
    assert "github.com/owner/demo" not in text.lower()
    assert f"#{mat.pr_number}" not in text

    spec = build_real_pr_pack_spec(mat, force_offline=True)
    assert find_provenance_fingerprints(spec.instruction_md) == []
    assert has_deepswe_footer(spec.instruction_md)
    assert spec.task_toml.metadata.source_track == "real_pr"  # private meta ok
    assert mat.base_commit not in spec.instruction_md


def test_gold_leak_still_clean() -> None:
    unique = "return atomic_reserve_and_price_with_unique_marker_ZXQ99_never_in_prompt()\n"
    mat = RealPrMaterial(
        task_id="realpr-demo-77",
        repository_url="https://github.com/owner/demo.git",
        base_commit=_FULL_SHA,
        language="python",
        license="MIT",
        pr_number=77,
        title="Restore multi-module checkout totals",
        body=_LONG_BODY,
        source_files=("pkg/pricing.py", "pkg/inventory.py"),
        test_files=("tests/test_checkout.py",),
        solution_patch=(
            "diff --git a/pkg/pricing.py b/pkg/pricing.py\n"
            "--- a/pkg/pricing.py\n"
            "+++ b/pkg/pricing.py\n"
            "@@ -1,1 +1,2 @@\n"
            f"+{unique}"
            "+return 42\n"
        ),
        test_patch=_TEST_PATCH,
        materials_dir="",
        discovery_path="search",
        source_hunk_count=2,
    )
    instruction = build_real_pr_agent_instruction(mat, force_offline=True)
    hits = scan_instruction_gold_leak(instruction, mat.solution_patch)
    assert hits == []
    assert unique.strip() not in instruction
    assert "diff --git" not in instruction
    assert find_provenance_fingerprints(instruction) == []


def test_footer_always_present() -> None:
    text = build_deepswe_style_instruction(
        title="Tiny change",
        body="",
        force_offline=True,
        persist_cache=False,
    )
    assert has_deepswe_footer(text)
    assert text.strip().endswith("when you are done.")
