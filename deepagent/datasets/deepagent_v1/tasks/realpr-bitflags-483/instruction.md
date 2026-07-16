# realpr-bitflags-483

## Context
You are solving a **long-horizon multi-file** software engineering task mined from
a real merged pull request on a public repository.

- **Repository URL:** `https://github.com/bitflags/bitflags.git`
- **Base commit (immutable):** `4ed9ffa949970239cd2d87c775e9fdcf9c438fb5`
- **Language:** `rust`
- **Merged PR:** `#483` — realpr-bitflags-483
- **Source track:** `real_pr` (agent environment is a clean clone at the base SHA)

Cross-module product behaviour is composed across independent source files rather
than a single helper. A regression was fixed upstream by multi-file changes that
touched at least two product sources. Your job is to restore that intended
contract from the agent-visible tree alone.

Affected product source modules include:
`src/lib.rs`, `src/public.rs`, `src/tests.rs`

## PR description
Closes #470 

This PR introduces a `#[flag_name]` attribute you can apply to flags to give them a different name. I've tried to avoid adding these kinds of attributes in the past, but there's no way to get around this one with `bitflags`, so we support it. It lets you write:

```rust
bitflags! {
    pub struct MyFlags: u8 {
        #[flag_name = "a"]
        const A = 1;
    }
}
```

There isn't any validation done on the `flag_name` given, so you could pick strange names like `_`, or `a | b` or `  a`, etc. I need to add some more test coverage before this is ready.

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
