# Release 3.4.6

## Context
You are solving a **long-horizon multi-file** software engineering task mined from
a real merged pull request on a public repository.

- **Repository URL:** `https://github.com/jawah/charset_normalizer.git`
- **Base commit (immutable):** `7411396ebd495e1abc28f5682975b5c662b2ff35`
- **Language:** `python`
- **Merged PR:** `#715` — Release 3.4.6
- **Source track:** `real_pr` (agent environment is a clean clone at the base SHA)

Cross-module product behaviour is composed across independent source files rather
than a single helper. A regression was fixed upstream by multi-file changes that
touched at least two product sources. Your job is to restore that intended
contract from the agent-visible tree alone.

Affected product source modules include:
`_mypyc_hook/backend.py`, `noxfile.py`, `src/charset_normalizer/api.py`, `src/charset_normalizer/cd.py`, `src/charset_normalizer/cli/__main__.py`, `src/charset_normalizer/constant.py`, `src/charset_normalizer/legacy.py`, `src/charset_normalizer/md.py`, `src/charset_normalizer/models.py`, `src/charset_normalizer/utils.py`, `src/charset_normalizer/version.py`

## PR description
## [3.4.6](https://github.com/Ousret/charset_normalizer/compare/3.4.5...3.4.6) (2026-03-15)

### Changed
- Flattened the logic in `charset_normalizer.md` for higher performance. Removed `eligible(..)` and `feed(...)`
  in favor of `feed_info(...)`.
- Raised upper bound for mypy[c] to 1.20, for our optimized version.
- Updated `UNICODE_RANGES_COMBINED` using Unicode blocks v17.

### Fixed
- Edge case where noise difference between two candidates can be almost insignificant. (#672)
- CLI `--normalize` writing to wrong path when passing multiple files in. (#702)

### Misc
- Freethreaded pre-built wheels now shipped in PyPI starting with 3.14t. (#616)

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
