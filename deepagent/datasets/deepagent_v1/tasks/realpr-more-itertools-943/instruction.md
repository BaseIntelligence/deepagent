# Changes for release 10.6.0

## Context
You are solving a **long-horizon multi-file** software engineering task mined from
a real merged pull request on a public repository.

- **Repository URL:** `https://github.com/more-itertools/more-itertools.git`
- **Base commit (immutable):** `f36c88fe03688fa442154ef14f429bcfa4c38525`
- **Language:** `python`
- **Merged PR:** `#943` — Changes for release 10.6.0
- **Source track:** `real_pr` (agent environment is a clean clone at the base SHA)

Cross-module product behaviour is composed across independent source files rather
than a single helper. A regression was fixed upstream by multi-file changes that
touched at least two product sources. Your job is to restore that intended
contract from the agent-visible tree alone.

Affected product source modules include:
`docs/conf.py`, `more_itertools/more.py`, `more_itertools/recipes.py`

## PR description
This PR has the changed for the next release.

The [changelog](https://github.com/more-itertools/more-itertools/blob/28f5e8f142d191626280e98ebc4ce3773c0db9cd/docs/versions.rst) has the highlights.

@debruijn, I apologize for not deciding this sooner, but I yanked `derangments` and `distinct_derangements` once I started reformatting the docstrings. I'd like to include versions of these in a future release, but I came to the conclusion that a different behavior is more desirable. I'll put more details in #937. Sorry!

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
