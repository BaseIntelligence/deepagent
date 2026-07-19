feat: add public PEP 440 compliance `VersionRange` class. This supersedes #1182 with a smaller, more focused implementation. `Specifier` and `SpecifierSet` already run their PEP 440 matching through a shared internal range engine (`_ranges.py`). A specifier becomes a list of bound intervals, and membership, filtering, and intersection all run on those intervals. That is the state on main today. This PR exposes that capability as a public set-algebra object. `VersionRange` is a second consumer of the same engine: it reuses the engine's bound construction, intersection, membership test, filtering, and pre-release handling, and adds the two operations only it needs, union and complement, plus a canonical interval form. `SpecifierSet` gets one new meth… Affected product modules include `src/packaging/_ranges.py`, `src/packaging/ranges.py`, `src/packaging/specifiers.py`.

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
