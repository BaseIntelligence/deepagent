# refactor `EnvironBuilder` file handling and related code

## Context
You are solving a **long-horizon multi-file** software engineering task mined from
a real merged pull request on a public repository.

- **Repository URL:** `https://github.com/pallets/werkzeug.git`
- **Base commit (immutable):** `70551309d170d43696fff527cd5b5893421996ba`
- **Language:** `python`
- **Merged PR:** `#3101` — refactor `EnvironBuilder` file handling and related code
- **Source track:** `real_pr` (agent environment is a clean clone at the base SHA)

Cross-module product behaviour is composed across independent source files rather
than a single helper. A regression was fixed upstream by multi-file changes that
touched at least two product sources. Your job is to restore that intended
contract from the agent-visible tree alone.

Affected product source modules include:
`src/werkzeug/datastructures/file_storage.py`, `src/werkzeug/formparser.py`, `src/werkzeug/test.py`, `src/werkzeug/wrappers/request.py`

## PR description
This started off as me addressing an observation in #3092 that `EnvironBuilder` could be a context manager, and turned into a giant refactor and docs revision as I traced through more of the code and tests. Steps along the way are split into commits. Pretty much the entire thing relates to avoiding `ResourceWarnings` during testing, in one way or another.

`EnvironBuilder` can now be used as a `with` context manager. This ensures any files in `builder.files` are closed. `builder.input_stream` is not closed, as it's assumed to be managed externally in a test. The way `form`, `files`, and `input_stream` interact is a lot cleaner now, the multidicts are cleared when setting the stream, rather than shuffling `None` checks around, and clearing the file multidict closes the files. The same applies to how `args` and `query_string` interact. The docs have been rewritten to make all of the behaviors and relationships between all these args and attrs much clearer.

While adding the `close` and `clear` behavior to `FileMultiDict`, I noticed that unlike `FileStorage`, `add_file` does not detect the filename from an IO object, which means it won't guess the content type either. Refactored so both places do the detection.

I then when through the tests to find everywhere that was using `EnvironBuilder`, and switch it to a context manager anywhere files are involved. Then did the same with `Request.from_values`.

I noticed that one test was still showing a `ResourceWarning`. `test_premature_end_of_file` was testing that the parser would handle if the stream ended partway through a multipart file. It does handle it, by raising an error that's then silenced higher up. But because it failed before filling in `request.files`, but after opening the file to populate it, `request.close` could not close the file later during the test. To fix that, `MultiPartParser` is a context manager as well now. It tracks files as it creates them, and `__exit__` will close them if the context is exitin…

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
