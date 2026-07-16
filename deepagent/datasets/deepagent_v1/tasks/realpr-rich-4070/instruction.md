# Reduce import time for Console and RichHandler by deferring unused imports

Importing `Console` and `RichHandler` from this library currently pulls in a large chain of modules eagerly, even though many of them are only needed on specific code paths. Since this library is vendored by tools like `pip`, these imports run on nearly every command invocation, so trimming the import graph has broad startup-time benefits.

Refactor the codebase so that imports only used in narrow code paths are deferred to the point of use, annotation-only imports are moved under `TYPE_CHECKING`, and dead imports/code are removed — all without changing any public runtime behavior.

## Expected outcomes

1. `from rich.console import Console` and `from rich.logging import RichHandler` import noticeably faster because unused transitive modules are no longer loaded eagerly.
2. In `logging.py`, `Traceback` is imported lazily inside the emit path and only loaded when rich tracebacks are actually enabled. Type-only references (`Console`, `ConsoleRenderable`, `Highlighter`, `FormatTimeCallable`) are moved under `TYPE_CHECKING`, and file-name extraction uses `os.path.basename` instead of constructing a `pathlib.Path`.
3. In `console.py`, the `inspect` module is no longer imported: replace `isclass(x)` with `isinstance(x, type)` and `currentframe()` with `sys._getframe(...)`. The `pretty`, `scope`, `getpass`, `html.escape`, and `zlib` dependencies are imported lazily within the methods that use them (`print`, `log` when locals are logged, `input` when reading a password, and the HTML/SVG export methods). The unused `_svg_hash` helper is removed.
4. `segment.py` no longer imports `logging` (the logger was assigned but never used).
5. `theme.py` imports `configparser` lazily inside `Theme.from_file()`.
6. `syntax.py` moves `Console`, `ConsoleOptions`, `JustifyMethod`, and `RenderResult` under `TYPE_CHECKING` so importing `Syntax` no longer drags in the console module.
7. `protocol.py` replaces `from inspect import isclass` usage with `isinstance(x, type)`, and `repr.py` imports `inspect` lazily inside `auto_rich_repr()`.

## Constraints

- All deferred symbols must still be fully available at runtime; loading them on demand must not change observable behavior or error semantics.
- Files relying on postponed evaluation of annotations must include `from __future__ import annotations` so `TYPE_CHECKING`-only imports remain valid in signatures.
- Do not alter any public API, method signatures, or return values.
- The existing test suite must continue to pass, `mypy` must report no new issues, and formatting must remain clean.

## Implementation notes

- Prefer moving imports to the innermost scope where they are genuinely required, rather than the top of the module.
- Guard truly conditional imports (e.g. `Traceback`, `getpass`, `scope`) behind the runtime flag or branch that needs them.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
