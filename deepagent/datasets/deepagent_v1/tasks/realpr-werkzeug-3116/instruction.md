# move structured header parsing to class methods

## Context
You are solving a **long-horizon multi-file** software engineering task mined from
a real merged pull request on a public repository.

- **Repository URL:** `https://github.com/pallets/werkzeug.git`
- **Base commit (immutable):** `a792bc2d1ebd52abfd285db847ef0fd42a911df9`
- **Language:** `python`
- **Merged PR:** `#3116` — move structured header parsing to class methods
- **Source track:** `real_pr` (agent environment is a clean clone at the base SHA)

Cross-module product behaviour is composed across independent source files rather
than a single helper. A regression was fixed upstream by multi-file changes that
touched at least two product sources. Your job is to restore that intended
contract from the agent-visible tree alone.

Affected product source modules include:
`src/werkzeug/datastructures/accept.py`, `src/werkzeug/datastructures/auth.py`, `src/werkzeug/datastructures/cache_control.py`, `src/werkzeug/datastructures/csp.py`, `src/werkzeug/datastructures/etag.py`, `src/werkzeug/datastructures/file_storage.py`, `src/werkzeug/datastructures/headers.py`, `src/werkzeug/datastructures/range.py`, `src/werkzeug/datastructures/structures.py`, `src/werkzeug/http.py`, `src/werkzeug/sansio/http.py`, `src/werkzeug/sansio/request.py` (+4 more)

## PR description
Some header classes, such as `Authorization`, already had `from_header` class method and `to_header` method. Many were missing the `from_header` class method. These all had corresponding `parse_thing_header` methods in `http`, which created a lot of circular imports and made type annotations more complicated. Given that the parse functions only returned the classes anyway (not some simpler representation like a plain list or dict), there was not really a benefit to keeping the parsing logic in one place while all the other logic was in another.

Searching with sourcegraph doesn't turn up much of anything, either old projects or copies of things. I have a feeling the vast majority of uses are through `Request` and `Response` attributes (which doesn't change with this).

- `dump_csp_header` -> `ContentSecurityPolicy.to_header`
- `parse_accept_header` -> `Accept.from_header`
- `parse_cache_control_header` -> `RequestCacheControl.from_header`
- `parse_content_range_header` -> `ContentRange.from_header`
- `parse_csp_header` -> `ContentSecurityPolicy.from_header`
- `parse_etags` -> `ETags.from_header`
- `parse_if_range_header` -> `IfRange.from_header`
- `parse_range_header` -> `Range.from_header`
- `parse_set_header` -> `HeaderSet.from_header`

See https://github.com/pallets/werkzeug/pull/2619 for the precursor to this, `Authorization` was updated in 2023 with the plan that it would continue on to these other structured headers.

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
