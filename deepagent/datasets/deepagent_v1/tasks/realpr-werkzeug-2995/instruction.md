# restrict containers accepted by multi

## Context
You are solving a **long-horizon multi-file** software engineering task mined from
a real merged pull request on a public repository.

- **Repository URL:** `https://github.com/pallets/werkzeug.git`
- **Base commit (immutable):** `1a1728ed88939ca68928dade168e1989be062c6f`
- **Language:** `python`
- **Merged PR:** `#2995` — restrict containers accepted by multi
- **Source track:** `real_pr` (agent environment is a clean clone at the base SHA)

Cross-module product behaviour is composed across independent source files rather
than a single helper. A regression was fixed upstream by multi-file changes that
touched at least two product sources. Your job is to restore that intended
contract from the agent-visible tree alone.

Affected product source modules include:
`src/werkzeug/datastructures/headers.py`, `src/werkzeug/datastructures/structures.py`

## PR description
When inlining the type annotations, I noticed that `iter_multi_items` and related methods were annotated to accept `collections.abc.Iterable`, but actually called `isinstance(value, (list, tuple)`. I expanded this to `isinstance(value, Container) and not isinstance(value, str)`. However, this incorrectly matched and iterated over `bytes`, `bytearray`, `memoryview`, `array`, etc, all of which should be treated as single values. Rather than trying to build up an allow list, I've gone back to restricting to `(list, tuple, set)`, adding `set` since it was the one built-in collection type missing that would make sense to use.

fixes #2994

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
