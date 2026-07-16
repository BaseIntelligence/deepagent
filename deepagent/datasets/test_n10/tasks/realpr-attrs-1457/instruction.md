# Make `kw_only=True` behavior consistent with dataclasses

## Context
You are solving a **long-horizon multi-file** software engineering task mined from
a real merged pull request on a public repository.

- **Repository URL:** `https://github.com/python-attrs/attrs.git`
- **Base commit (immutable):** `af349876f4fb5df2b71a3f4878239b775bc89230`
- **Language:** `python`
- **Merged PR:** `#1457` — Make `kw_only=True` behavior consistent with dataclasses
- **Source track:** `real_pr` (agent environment is a clean clone at the base SHA)

Cross-module product behaviour is composed across independent source files rather
than a single helper. A regression was fixed upstream by multi-file changes that
touched at least two product sources. Your job is to restore that intended
contract from the agent-visible tree alone.

Affected product source modules include:
`src/attr/_make.py`, `src/attr/_next_gen.py`

## PR description
# Summary

This PR resolve #980.

### Before this change
When `kw_only=True` is set on a class, all attributes are made keyword-only, including those from base classes. If an attribute sets `kw_only=False`, that setting is ignored and it is still made keyword-only.

### After this change
When `kw_only=True` is set on a class, only the attributes defined in that class that doesn't explicitly set `kw_only=False` are made keyword-only. Notably this is also the behavior for `dataclasses`.

See `TestKeywordOnlyAttributes.{test_kw_only_inheritance,test_kw_only_inheritance_force_kw_only}` for an example of the old and new behaviors.

### Implementation
The default for `kw_only` in `attr.ib`/`attrs.field` is changed from False to None. When set to None, the attribute's `kw_only` mirrors the value set on the class.

An additional back-compat `force_kw_only` argument is added to `attr.s`/`attrs.define`. When set to True, the old behavior is restored for the class. The default is False (new behavior) for `attrs.define` and True (old behavior) for `attr.s`.

### Backwards compatibility
This change makes certain attributes that were previously kw-only no longer kw-only. As such, all call sites where we construct an instance of an attrs class should be backwards compatible.

If no attribute in an attrs class and all of its base classes explicitly sets `kw_only=False`, then that class is also backwards compatible.

If there are `kw_only=False` attributes, then it is possible for the class to fail on import due to attribute ordering. For example:
```python
@attrs.define(kw_only=True)
class A:
    a: int
    b: int = attrs.field(default=1, kw_only=False)

@attrs.define
class B(A):
    c: int
```
Previously, `B.__init__` would have the signature `(self, c, *, a, b=...)`, but with the new behavior this is wrong because `c` follows `b`, a non-kw-only attribute with a default.

My hope is that this is rare enough --- since `kw_only=False` was the default on `attr.ib`/`attrs.field` and d…

## Behavioural requirements
1. Restore the original multi-module contracts so the held-out **fail_to_pass**
   cases pass when your solution is applied.
2. Do **not** remove, skip, rename, or rewrite existing tests as a "fix". The
   graded suite is enforced by a separate verifier image; plastic diffs that
   weaken coverage score 0.
3. Prefer a minimal multi-file unified-diff style change under the repository
   root. Paths should look like `--- a/<rel>` / `+++ b/<rel>` relative product
   paths (the harness materializes your work as `model.patch`).
4. Keep **pass_to_pass** behaviour intact for unrelated modules and branches.
5. Hard product track requires a multi-file solution (≥2 product source files).
   Single-hunk NotImplemented stubs or docs-only edits are not acceptable.
6. Do not invent secrets, API keys, or vendor credentials in the tree.

The held-out verifier suite defines the graded **fail_to_pass** set (node ids live only in the hidden tests/config, not in this prompt). Your multi-file source patch must flip every fail-to-pass case red → green while **pass_to_pass** regressions stay green.

## Deliverable
Work on a **new branch** from the pinned base checkout. Implement the multi-file
source fix that restores the green behavioural contract against the held-out
verifier suite. Commit when done and leave a clean porcelain tree so the grader
can harvest `model.patch`.

IMPORTANT: Please work on this in a new branch from the base commit and commit
everything when you are done. Do not weaken pass_to_pass coverage.
