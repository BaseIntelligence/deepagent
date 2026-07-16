"""VAL-DHARD-001: prompt–verifier alignment gate (fail-closed product dest).

Covers more-itertools-1136-class misalignment (version/export instruction while
F2P/gold quiz windowed / unique_everseen runtime) and an aligned control pack.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from swe_factory.pipeline.prompt_alignment import (
    REASON_PROMPT_ALIGN_OK,
    REASON_PROMPT_ALIGN_SKIPPED,
    REASON_PROMPT_NO_RUNTIME_CLAIM,
    REASON_PROMPT_VERSION_ONLY,
    PromptVerifierMisalignRejected,
    analyze_instruction_claims,
    analyze_verifier_signals,
    check_prompt_verifier_alignment,
    is_alignment_enforced_dest,
    refuse_prompt_verifier_misalign,
    summarize_test_intent,
)
from swe_factory.pipeline.ship_real_pr import (
    ProductPromptAlignRejected,
    RealPrMaterial,
    build_real_pr_pack_spec,
)
from swe_factory.pipeline.ship_real_pr import (
    refuse_prompt_verifier_misalign as ship_refuse_prompt_align,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures (1136-like + aligned behavioural control)
# ---------------------------------------------------------------------------

_MISALIGNED_INSTRUCTION_1136 = textwrap.dedent(
    """\
    Prepare the codebase for the next major release by bumping the package version
    and ensuring the public API surface is consistent and correct across the module.

    ## Expected outcomes
    1. The package version string is updated to `11.0.0` wherever it is exposed
       (e.g., `__version__` in the package's `__init__.py`).
    2. All public functions defined in the `more` and `recipes` modules are correctly
       exported via `__all__`, and there are no stale entries referencing removed or
       renamed callables.
    3. Importing the package and inspecting its version reports `11.0.0`.
    4. The existing test suite continues to pass without regressions.

    ## Constraints
    - Do not change the runtime behavior of existing iterators, recipes, or helper
      functions unless required to keep `__all__` and actual definitions in sync.
    - Keep the version string in a single canonical location if the project already
      centralizes it; avoid duplicating it inconsistently across files.
    - Maintain backward compatibility for all documented, public names.
    - Follow the existing code style and formatting conventions used throughout
      the modules.

    ## Implementation notes
    - Verify that every name listed in `__all__` resolves to an actual object in the
      module, and that every intended public object appears in `__all__`.
    - If the version is referenced elsewhere (documentation strings, metadata helpers),
      ensure those references are consistent with the new value.

    IMPORTANT: Please work on this in a new branch from main and commit everything
    when you are done.
    """
)

_TEST_PATCH_1136 = textwrap.dedent(
    """\
    diff --git a/tests/test_more.py b/tests/test_more.py
    --- a/tests/test_more.py
    +++ b/tests/test_more.py
    @@ -829,7 +829,6 @@ def test_basic(self):
             (3, [(1, 2, 3), (2, 3, 4), (3, 4, 5)]),
             (2, [(1, 2), (2, 3), (3, 4), (4, 5)]),
             (1, [(1,), (2,), (3,), (4,), (5,)]),
    -            (0, [()]),
         ):
             with self.subTest(n=n):
                 actual = list(mi.windowed(iterable, n))
    @@ -865,9 +864,11 @@ def test_fillvalue_step(self):
         expected = [(1, 2, 3), (4, 5, '!')]
         self.assertEqual(actual, expected)
     
    -    def test_negative(self):
    +    def test_invalid_n(self):
    +        with self.assertRaises(ValueError):
    +            list(mi.windowed([1, 2, 3, 4, 5], 0))  # n is zero
         with self.assertRaises(ValueError):
    -            list(mi.windowed([1, 2, 3, 4, 5], -1))
    +            list(mi.windowed([1, 2, 3, 4, 5], -1))  # n is negative
     
         def test_empty_seq(self):
             actual = list(mi.windowed([], 3))
    diff --git a/tests/test_recipes.py b/tests/test_recipes.py
    --- a/tests/test_recipes.py
    +++ b/tests/test_recipes.py
    @@ -410,15 +410,36 @@ def test_custom_key(self):
         u = mi.unique_everseen('aAbACCc', key=str.lower)
         self.assertEqual(list('abC'), list(u))
     
    -    def test_unhashable(self):
    -        iterable = ['a', [1, 2, 3], [1, 2, 3], 'a']
    -        u = mi.unique_everseen(iterable)
    -        self.assertEqual(list(u), ['a', [1, 2, 3]])
    -
    -    def test_unhashable_key(self):
    -        iterable = ['a', [1, 2, 3], [1, 2, 3], 'a']
    -        u = mi.unique_everseen(iterable, key=lambda x: x)
    -        self.assertEqual(list(u), ['a', [1, 2, 3]])
    +    def test_unhashable_lists(self):
    +        iterable = ['a', [1, 2, 3], [1, 2, 3], 'a']
    +        with self.assertRaises(TypeError):
    +            list(mi.unique_everseen(iterable))
    +        self.assertEqual(
    +            list(mi.unique_everseen(iterable, key=str)),
    +            ['a', [1, 2, 3]],
    +        )
    +
    +    def test_unhashable_sets(self):
    +        iterable = ['a', {1}, {1}, 'a']
    +        with self.assertRaises(TypeError):
    +            list(mi.unique_everseen(iterable))
    +        self.assertEqual(list(mi.unique_everseen(iterable, key=str)), ['a', {1}])
    +
    +    def test_unhashable_dicts(self):
    +        iterable = ['a', {1: 2}, {1: 2}, 'a']
    +        with self.assertRaises(TypeError):
    +            list(mi.unique_everseen(iterable))
    +        self.assertEqual(
    +            list(mi.unique_everseen(iterable, key=str)),
    +            ['a', {1: 2}],
    +        )
    """
)

# Use single-quoted outer triple strings so patch bodies may contain """ docstrings.
_SOLUTION_PATCH_1136 = textwrap.dedent(
    '''\
    diff --git a/more_itertools/__init__.py b/more_itertools/__init__.py
    --- a/more_itertools/__init__.py
    +++ b/more_itertools/__init__.py
    @@ -3,4 +3,4 @@
     from .more import *  # noqa
     from .recipes import *  # noqa

    -__version__ = '10.8.0'
    +__version__ = '11.0.0'
    diff --git a/more_itertools/more.py b/more_itertools/more.py
    --- a/more_itertools/more.py
    +++ b/more_itertools/more.py
    @@ -1045,11 +1045,8 @@ def windowed(seq, n, fillvalue=None, step=1):
             >>> list(windowed(chain(padding, iterable), 3))
             [(None, None, 1), (None, 1, 2), (1, 2, 3), (2, 3, 4)]
         """
    -    if n < 0:
    -        raise ValueError('n must be >= 0')
    -    if n == 0:
    -        yield ()
    -        return
    +    if n <= 0:
    +        raise ValueError('n must be > 0')
         if step < 1:
             raise ValueError('step must be >= 1')
    diff --git a/more_itertools/recipes.py b/more_itertools/recipes.py
    --- a/more_itertools/recipes.py
    +++ b/more_itertools/recipes.py
    @@ -640,12 +640,18 @@ def unique_everseen(iterable, key=None):
         """
         seen = set()
         seen_add = seen.add
    -    if key is None:
    -        for element in iterable:
    -            k = element
    -            if k not in seen:
    -                seen_add(k)
    -                yield element
    +    if key is None:
    +        for element in iterable:
    +            # Unhashable elements raise TypeError without a key.
    +            if element not in seen:
    +                seen_add(element)
    +                yield element
         else:
             for element in iterable:
                 k = key(element)
                 if k not in seen:
                     seen_add(k)
                     yield element
    '''
)

_ALIGNED_INSTRUCTION = textwrap.dedent(
    """\
    # Add JSON Schema generation for item classes

    `itemadapter` supports several item types. Add a `get_json_schema()` function
    that, given a supported item class, returns a JSON Schema (`dict`) describing
    that class's fields and their types.

    ## Expected outcomes
    1. A public `get_json_schema(item_class)` callable is available that accepts a
       supported item class and returns a JSON Schema as a plain `dict`.
    2. The result has `"type": "object"` at the top level, with a `"properties"`
       mapping keyed by field name.
    3. Field Python types are mapped to their JSON Schema equivalents where
       determinable (e.g. `str` → `{"type": "string"}`).
    4. Container types with parameterized element types (e.g. `list[int]`)
       populate the corresponding nested schema (e.g. `"items"`).
    5. Unsupported item classes raise `TypeError` with a clear message.

    ## Constraints
    - Do not break existing public APIs; only add the new helper and its tests.
    - Follow the existing code style used throughout the package.

    IMPORTANT: Please work on this in a new branch from main and commit everything
    when you are done.
    """
)

_ALIGNED_TEST_PATCH = textwrap.dedent(
    """\
    diff --git a/tests/test_json_schema.py b/tests/test_json_schema.py
    --- a/tests/test_json_schema.py
    +++ b/tests/test_json_schema.py
    @@ -0,0 +1,40 @@
    +import pytest
    +from itemadapter.utils import get_json_schema
    +
    +
    +def test_basic_dataclass_schema():
    +    schema = get_json_schema(Item)
    +    assert schema["type"] == "object"
    +    assert "properties" in schema
    +    assert schema["properties"]["name"]["type"] == "string"
    +
    +
    +def test_list_container_items():
    +    schema = get_json_schema(ListItem)
    +    assert schema["properties"]["tags"]["type"] == "array"
    +    assert schema["properties"]["tags"]["items"]["type"] == "string"
    +
    +
    +def test_unsupported_raises_typeerror():
    +    with pytest.raises(TypeError):
    +        get_json_schema(object)
    """
)

_ALIGNED_SOLUTION = textwrap.dedent(
    """\
    diff --git a/itemadapter/utils.py b/itemadapter/utils.py
    --- a/itemadapter/utils.py
    +++ b/itemadapter/utils.py
    @@ -0,0 +1,25 @@
    +def get_json_schema(item_class):
    +    if not is_supported(item_class):
    +        raise TypeError(f"unsupported: {item_class!r}")
    +    props = {}
    +    for name, typ in fields_of(item_class):
    +        props[name] = type_to_schema(typ)
    +    return {"type": "object", "properties": props}
    diff --git a/itemadapter/__init__.py b/itemadapter/__init__.py
    --- a/itemadapter/__init__.py
    +++ b/itemadapter/__init__.py
    @@ -1,3 +1,4 @@
     from itemadapter.adapter import ItemAdapter
    +from itemadapter.utils import get_json_schema
    """
)

_SHA40 = "a" * 40


def _material(
    *,
    task_id: str,
    instruction: str,
    test_patch: str,
    solution_patch: str,
    source_files: tuple[str, ...] = ("pkg/a.py", "pkg/b.py"),
    test_files: tuple[str, ...] = ("tests/test_x.py",),
) -> RealPrMaterial:
    return RealPrMaterial(
        task_id=task_id,
        repository_url="https://github.com/example/repo",
        base_commit=_SHA40,
        language="python",
        license="MIT",
        pr_number=1136,
        title="fake title",
        source_files=source_files,
        test_files=test_files,
        solution_patch=solution_patch,
        test_patch=test_patch,
        body="",
        agent_instruction=instruction,
    )


# ---------------------------------------------------------------------------
# Unit: mechanical claims / signals
# ---------------------------------------------------------------------------


def test_analyze_instruction_claims_1136_version_no_runtime() -> None:
    claims = analyze_instruction_claims(_MISALIGNED_INSTRUCTION_1136)
    assert claims.no_runtime_claim is True
    assert claims.version_export_claim is True
    assert claims.version_export_only is True
    assert claims.multi_behavior_ask is False


def test_analyze_instruction_claims_aligned_multi_behavior() -> None:
    claims = analyze_instruction_claims(_ALIGNED_INSTRUCTION)
    assert claims.no_runtime_claim is False
    assert claims.version_export_only is False
    assert claims.behavior_outcome_count >= 2
    assert claims.multi_behavior_ask is True


def test_analyze_verifier_signals_1136_runtime() -> None:
    sig = analyze_verifier_signals(_TEST_PATCH_1136, solution_patch=_SOLUTION_PATCH_1136)
    assert sig.runtime_behavior_changes is True
    assert sig.version_only is False
    assert sig.gold_runtime_delta is True
    # At least one of the known behavioural classes should fire.
    assert sig.behavior_class_count >= 1
    assert any(
        c.startswith("windowed") or c.startswith("unique_everseen") or c == "error_contract"
        for c in sig.behavior_classes
    )


def test_analyze_verifier_signals_aligned_schema() -> None:
    sig = analyze_verifier_signals(_ALIGNED_TEST_PATCH, solution_patch=_ALIGNED_SOLUTION)
    assert sig.runtime_behavior_changes is True
    assert sig.version_only is False


# ---------------------------------------------------------------------------
# Unit: core check + refuse
# ---------------------------------------------------------------------------


def test_check_misaligned_1136_refuses_with_stable_code() -> None:
    result = check_prompt_verifier_alignment(
        _MISALIGNED_INSTRUCTION_1136,
        test_patch=_TEST_PATCH_1136,
        solution_patch=_SOLUTION_PATCH_1136,
        f2p_node_ids=[
            "tests.test_more.TestWindowed.test_invalid_n",
            "tests.test_recipes.TestUniqueEverseen.test_unhashable_lists",
            "tests.test_recipes.TestUniqueEverseen.test_unhashable_sets",
        ],
    )
    assert result.ok is False
    assert result.reason_code in {
        REASON_PROMPT_NO_RUNTIME_CLAIM,
        REASON_PROMPT_VERSION_ONLY,
    }
    assert "runtime" in result.detail.lower() or "version" in result.detail.lower()


def test_check_aligned_pack_passes() -> None:
    result = check_prompt_verifier_alignment(
        _ALIGNED_INSTRUCTION,
        test_patch=_ALIGNED_TEST_PATCH,
        solution_patch=_ALIGNED_SOLUTION,
        f2p_node_ids=[
            "tests.test_json_schema::test_basic_dataclass_schema",
            "tests.test_json_schema::test_list_container_items",
            "tests.test_json_schema::test_unsupported_raises_typeerror",
        ],
    )
    assert result.ok is True
    assert result.reason_code == REASON_PROMPT_ALIGN_OK


def test_refuse_prompt_verifier_misalign_product_dest() -> None:
    with pytest.raises(PromptVerifierMisalignRejected) as ei:
        refuse_prompt_verifier_misalign(
            _MISALIGNED_INSTRUCTION_1136,
            test_patch=_TEST_PATCH_1136,
            solution_patch=_SOLUTION_PATCH_1136,
            dest="datasets/deepagent_v1",
            offline_only=False,
            task_id="realpr-more-itertools-1136",
        )
    assert ei.value.reason_code in {
        REASON_PROMPT_NO_RUNTIME_CLAIM,
        REASON_PROMPT_VERSION_ONLY,
    }
    assert "VAL-DHARD-001" in str(ei.value)


def test_refuse_prompt_align_live_generate_test_n10() -> None:
    with pytest.raises(PromptVerifierMisalignRejected) as ei:
        refuse_prompt_verifier_misalign(
            _MISALIGNED_INSTRUCTION_1136,
            test_patch=_TEST_PATCH_1136,
            solution_patch=_SOLUTION_PATCH_1136,
            dest="datasets/test_n10",
            task_id="realpr-more-itertools-1136",
        )
    assert ei.value.reason_code


def test_refuse_skipped_on_offline_dest_without_force() -> None:
    res = refuse_prompt_verifier_misalign(
        _MISALIGNED_INSTRUCTION_1136,
        test_patch=_TEST_PATCH_1136,
        solution_patch=_SOLUTION_PATCH_1136,
        dest="/tmp/offline_only_ut",
        offline_only=False,
    )
    assert res.ok is True
    assert res.reason_code == REASON_PROMPT_ALIGN_SKIPPED


def test_refuse_force_over_offline() -> None:
    with pytest.raises(PromptVerifierMisalignRejected):
        refuse_prompt_verifier_misalign(
            _MISALIGNED_INSTRUCTION_1136,
            test_patch=_TEST_PATCH_1136,
            solution_patch=_SOLUTION_PATCH_1136,
            dest="/tmp/offline_only_ut",
            force=True,
        )


def test_aligned_product_dest_passes() -> None:
    res = refuse_prompt_verifier_misalign(
        _ALIGNED_INSTRUCTION,
        test_patch=_ALIGNED_TEST_PATCH,
        solution_patch=_ALIGNED_SOLUTION,
        dest="datasets/deepagent_v1",
        task_id="realpr-itemadapter-101",
    )
    assert res.ok is True
    assert res.reason_code == REASON_PROMPT_ALIGN_OK


def test_is_alignment_enforced_dest() -> None:
    assert is_alignment_enforced_dest("datasets/deepagent_v1") is True
    assert is_alignment_enforced_dest("datasets/test_n10") is True
    assert is_alignment_enforced_dest("datasets/prod_hard_keep") is True
    assert is_alignment_enforced_dest("datasets/deepagent_v1_offline_only") is False
    assert is_alignment_enforced_dest("/tmp/ut_offline") is False
    assert is_alignment_enforced_dest("datasets/deepagent_v1", offline_only=True) is False


def test_summarize_test_intent_no_node_id_dump() -> None:
    items = summarize_test_intent(_TEST_PATCH_1136)
    joined = " ".join(items)
    assert "test_invalid_n" in joined or "behaviour_class:" in joined
    # Must not invent gold solution bodies into the summary list.
    assert "n must be > 0" not in joined


# ---------------------------------------------------------------------------
# Integration: build_real_pr_pack_spec product refuse
# ---------------------------------------------------------------------------


def test_build_real_pr_pack_spec_refuses_misaligned_product_dest() -> None:
    mat = _material(
        task_id="realpr-more-itertools-1136-synth",
        instruction=_MISALIGNED_INSTRUCTION_1136,
        test_patch=_TEST_PATCH_1136,
        solution_patch=_SOLUTION_PATCH_1136,
        source_files=("more_itertools/__init__.py", "more_itertools/more.py"),
        test_files=("tests/test_more.py", "tests/test_recipes.py"),
    )
    with pytest.raises((ProductPromptAlignRejected, PromptVerifierMisalignRejected)) as ei:
        build_real_pr_pack_spec(
            mat,
            force_offline=True,
            dest="datasets/deepagent_v1",
            offline_only=False,
        )
    msg = str(ei.value).lower()
    assert "align" in msg or "prompt" in msg or "version" in msg or "runtime" in msg
    code = getattr(ei.value, "reason_code", "")
    assert code in {
        REASON_PROMPT_NO_RUNTIME_CLAIM,
        REASON_PROMPT_VERSION_ONLY,
        "prompt_verifier_misalign",
    } or code.startswith("prompt_")


def test_build_real_pr_pack_spec_refuses_live_generate_dest() -> None:
    mat = _material(
        task_id="realpr-more-itertools-1136-synth",
        instruction=_MISALIGNED_INSTRUCTION_1136,
        test_patch=_TEST_PATCH_1136,
        solution_patch=_SOLUTION_PATCH_1136,
        source_files=("more_itertools/__init__.py", "more_itertools/more.py"),
    )
    with pytest.raises(ProductPromptAlignRejected):
        build_real_pr_pack_spec(
            mat,
            force_offline=True,
            dest="datasets/test_n10",
            offline_only=False,
        )


def test_build_real_pr_pack_spec_aligned_product_passes() -> None:
    mat = _material(
        task_id="realpr-itemadapter-101-synth",
        instruction=_ALIGNED_INSTRUCTION,
        test_patch=_ALIGNED_TEST_PATCH,
        solution_patch=_ALIGNED_SOLUTION,
        source_files=("itemadapter/utils.py", "itemadapter/__init__.py"),
        test_files=("tests/test_json_schema.py",),
    )
    spec = build_real_pr_pack_spec(
        mat,
        force_offline=True,
        dest="datasets/deepagent_v1",
        offline_only=False,
    )
    assert "get_json_schema" in spec.instruction_md
    assert spec.task_id == "realpr-itemadapter-101-synth"


def test_build_real_pr_pack_spec_offline_allows_misalign_without_dest() -> None:
    """Historical offline unit path (no dest) must not break engineering builds."""
    mat = _material(
        task_id="realpr-more-itertools-1136-synth",
        instruction=_MISALIGNED_INSTRUCTION_1136,
        test_patch=_TEST_PATCH_1136,
        solution_patch=_SOLUTION_PATCH_1136,
        source_files=("more_itertools/__init__.py", "more_itertools/more.py"),
    )
    # No dest → gate not enforced (unit fixture ship path).
    spec = build_real_pr_pack_spec(mat, force_offline=True)
    assert spec.task_id == "realpr-more-itertools-1136-synth"


def test_ship_refuse_reexport() -> None:
    """ship_real_pr re-exports refuse_prompt_verifier_misalign for auditors."""
    assert ship_refuse_prompt_align is refuse_prompt_verifier_misalign


# ---------------------------------------------------------------------------
# Optional live pack smoke when datasets/test_n10 is present
# ---------------------------------------------------------------------------


def test_live_1136_pack_misaligns_when_present() -> None:
    pack = Path("datasets/test_n10/tasks/realpr-more-itertools-1136")
    if not pack.is_dir():
        pytest.skip("test_n10 1136 pack not present on disk")
    from swe_factory.pipeline.prompt_alignment import alignment_result_from_pack_dir

    result = alignment_result_from_pack_dir(pack)
    assert result.ok is False
    assert result.reason_code in {
        REASON_PROMPT_NO_RUNTIME_CLAIM,
        REASON_PROMPT_VERSION_ONLY,
        "prompt_empty_behavior_ask_vs_f2p",
    }


def test_live_aligned_pack_passes_when_present() -> None:
    pack = Path("datasets/test_n10/tasks/realpr-itemadapter-101")
    if not pack.is_dir():
        pytest.skip("test_n10 itemadapter pack not present on disk")
    from swe_factory.pipeline.prompt_alignment import alignment_result_from_pack_dir

    result = alignment_result_from_pack_dir(pack)
    assert result.ok is True, result.detail
