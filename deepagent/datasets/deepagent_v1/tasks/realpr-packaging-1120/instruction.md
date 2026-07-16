# Implement Specifiers and SpecifierSets filtering using ranges

## Context
You are solving a **long-horizon multi-file** software engineering task mined from
a real merged pull request on a public repository.

- **Repository URL:** `https://github.com/pypa/packaging.git`
- **Base commit (immutable):** `ff14df979f865165553999d9d1a111feec6f4843`
- **Language:** `python`
- **Merged PR:** `#1120` — Implement Specifiers and SpecifierSets filtering using ranges
- **Source track:** `real_pr` (agent environment is a clean clone at the base SHA)

Cross-module product behaviour is composed across independent source files rather
than a single helper. A regression was fixed upstream by multi-file changes that
touched at least two product sources. Your job is to restore that intended
contract from the agent-visible tree alone.

Affected product source modules include:
`benchmarks/specifiers.py`, `src/packaging/_ranges.py`, `src/packaging/specifiers.py`

## PR description
This is a ~~proof of concept of~~ switching the internal mechanics of Specifier and SpecifierSet to using intervals instead of iterative version filters. In general this speeds up any mildly complex specifier and can make very complex specifiers significantly faster (more than 10x).

However this is a large overhaul, first it is intended we land https://github.com/pypa/packaging/pull/1119 which implements just enough of this to introduce a new `is_unsatisfiable` method.

This implementation is designed to make it easy to build a public API on top of this, I imagine having some `to_range()` method which returns a PEP 440 compliant `VersionRange` object that can be manipulated with set algebra.

~~However, to make this more performant, especially for very simple one off cases where it is currently slower, the internal machinery probably needs to move away from bounds logic, implementing hot paths and side channels.  Will leave this in draft until after https://github.com/pypa/packaging/pull/1119 lands.~~

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
