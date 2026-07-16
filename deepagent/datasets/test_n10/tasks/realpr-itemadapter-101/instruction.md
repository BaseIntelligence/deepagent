# Add get_json_schema()

## Context
You are solving a **long-horizon multi-file** software engineering task mined from
a real merged pull request on a public repository.

- **Repository URL:** `https://github.com/scrapy/itemadapter.git`
- **Base commit (immutable):** `f7860b6ec7c5b49f623ecd1f67e73877f08039b6`
- **Language:** `python`
- **Merged PR:** `#101` — Add get_json_schema()
- **Source track:** `real_pr` (agent environment is a clean clone at the base SHA)

Cross-module product behaviour is composed across independent source files rather
than a single helper. A regression was fixed upstream by multi-file changes that
touched at least two product sources. Your job is to restore that intended
contract from the agent-visible tree alone.

Affected product source modules include:
`itemadapter/_imports.py`, `itemadapter/_json_schema.py`, `itemadapter/adapter.py`, `itemadapter/utils.py`

## PR description
Post-merge work:

- [x] Create issues to eventually address the following:
  - [x] [Support converting Python regexp patterns to JSON Schema patterns where possible, instead of ignoring any incompatible pattern however easy it would be to convert it](https://github.com/scrapy/itemadapter/issues/102).
  - [x] [Support recursive type definition (e.g. an item class has a field of its same type) by implementing $refs support, and use $refs as well when a given type is used more than once in the schema](https://github.com/scrapy/itemadapter/issues/103).
  - [x] [Improve JSON Schema generation](https://github.com/scrapy/itemadapter/issues/104).
    We can take some notes from https://github.com/pydantic/pydantic/blob/d156ba08c140ee1e2b931120cb150080843476fe/pydantic/json_schema.py#L1064.
    Maybe we can come up with an implementation that is flexible enough to support itemadapter and pydantic cases and split it into a separate library? Maybe learn lessons from JSON serializers out there, or even reuse one?

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
