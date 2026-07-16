# improve subdomain and host matching

## Context
You are solving a **long-horizon multi-file** software engineering task mined from
a real merged pull request on a public repository.

- **Repository URL:** `https://github.com/pallets/werkzeug.git`
- **Base commit (immutable):** `cb307c144e7b9092bf72b1a1dba5281e7c6ff838`
- **Language:** `python`
- **Merged PR:** `#3006` — improve subdomain and host matching
- **Source track:** `real_pr` (agent environment is a clean clone at the base SHA)

Cross-module product behaviour is composed across independent source files rather
than a single helper. A regression was fixed upstream by multi-file changes that
touched at least two product sources. Your job is to restore that intended
contract from the agent-visible tree alone.

Affected product source modules include:
`src/werkzeug/routing/map.py`, `src/werkzeug/routing/rules.py`

## PR description
`Map` takes `subdomain_matching`, moving the behavior out of Flask pallets/flask#5634. It's enabled by default to match current behavior. If it and `host_matching` are disabled, the request's `Host` doesn't factor into routing at all.

`bind_to_environ` `server_name` is not used if `host_matching` is enabled, otherwise it would restrict routing to only that host. If `subdomain_matching` is enabled and a subdomain couldn't be detected, `default_subdomain` is used if set, rather than always `"<invalid>"`.

This did not affect any existing tests. Leaving as draft until I have a chance to write tests and docs for all this. I also want to consider the use of `"<invalid>"` more, whether it should always be used (current behavior) or never be used (further than this PR).

fixes #3005

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
