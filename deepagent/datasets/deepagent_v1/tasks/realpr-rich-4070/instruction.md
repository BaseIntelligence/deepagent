# perf: reduce Console and RichHandler import time by deferring unused imports

## Context
You are solving a **long-horizon multi-file** software engineering task mined from
a real merged pull request on a public repository.

- **Repository URL:** `https://github.com/Textualize/rich.git`
- **Base commit (immutable):** `fc41075a3206d2a5fd846c6f41c4d2becab814fa`
- **Language:** `python`
- **Merged PR:** `#4070` — perf: reduce Console and RichHandler import time by deferring unused imports
- **Source track:** `real_pr` (agent environment is a clean clone at the base SHA)

Cross-module product behaviour is composed across independent source files rather
than a single helper. A regression was fixed upstream by multi-file changes that
touched at least two product sources. Your job is to restore that intended
contract from the agent-visible tree alone.

Affected product source modules include:
`rich/_emoji_replace.py`, `rich/console.py`, `rich/emoji.py`, `rich/logging.py`, `rich/protocol.py`, `rich/repr.py`, `rich/segment.py`, `rich/syntax.py`, `rich/theme.py`

## PR description
## E2E Results

| Import | master | This PR | Speedup |
|---|---|---|---|
| `from rich.console import Console` | 78.1ms | 52.2ms | **1.50x faster** |
| `from rich.logging import RichHandler` | 99.4ms | 56.9ms | **1.75x faster** |
| `import rich` | 18.2ms | 18.3ms | (already lean) |

## Summary

Defer module-level imports that are only needed in specific code paths, move annotation-only imports to `TYPE_CHECKING`, and remove dead code:

| File | Change | Savings |
|---|---|---|
| `logging.py` | Defer `Traceback` to `emit()` (only when `rich_tracebacks=True`) | ~20ms |
| `logging.py` | `from __future__ import annotations` + `TYPE_CHECKING` for `Console`, `ConsoleRenderable`, `Highlighter`, `FormatTimeCallable` | ~6ms |
| `logging.py` | Replace `pathlib.Path` → `os.path.basename` (also a minor runtime win) | ~4-5ms |
| `console.py` | Eliminate `import inspect`; replace `isclass` → `isinstance(x, type)`, `currentframe` → `sys._getframe` | ~10ms |
| `console.py` | Defer `pretty` to `Console.print()` | ~3-5ms |
| `console.py` | Defer `scope` to `Console.log()` (only when `log_locals=True`) | ~3-5ms |
| `console.py` | Defer `getpass` to `Console.input()` (only when `password=True`) | ~2ms |
| `console.py` | Defer `html.escape` and `zlib` to export methods | ~2.3ms |
| `console.py` | Remove dead `_svg_hash` function (unused since 113997ac, fixes latent NameError) | cleanup |
| `segment.py` | Remove dead `logging` import (`getLogger` assigned but never used) | ~2-3ms |
| `theme.py` | Defer `configparser` to `Theme.from_file()` | ~1.5ms |
| `syntax.py` | Move `Console`, `ConsoleOptions`, `JustifyMethod`, `RenderResult` to `TYPE_CHECKING` | eliminates console.py from syntax import chain |
| `protocol.py` | Replace `from inspect import isclass` → `isinstance(x, type)` | prepares for dataclasses removal |
| `repr.py` | Defer `import inspect` to `auto_rich_repr()` | prepares for dataclasses removal |

All deferred imports are still available at runtime — they're loaded when the c…

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
