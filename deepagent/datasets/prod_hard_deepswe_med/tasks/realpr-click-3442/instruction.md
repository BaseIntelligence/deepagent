Merge Main into Stable. Affected product modules include `docs/conf.py`, `examples/validation/validation.py`, `src/click/__init__.py`, `src/click/_compat.py`, `src/click/_termui_impl.py`, `src/click/_textwrap.py`, `src/click/_utils.py`, `src/click/_winconsole.py` (+10 more).

Expected outcomes
1. Restore the intended multi-module contracts so product behaviour matches the described problem across the affected source modules.
2. Keep unrelated public APIs and pass behaviour intact; do not remove, skip, or rewrite tests to force a green run.
3. Prefer a focused multi-file source change that addresses the behaviour in pure python without inventing secrets or credentials.

Constraints
- Do not embed or rely on held-out verifier sources; implement production behaviour only.
- Prefer minimal multi-file edits under the repository root.
- Language focus: python.
- Do not invent secrets, API keys, or vendor credentials.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
