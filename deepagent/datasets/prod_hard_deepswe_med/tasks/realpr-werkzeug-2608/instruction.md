replace calls to `werkzeug.urls` with `urllib.parse`. Use `urllib.parse` functions instead of our own implementation. Deprecate all of `werkzeug.urls` except for `uri_to_iri` and `iri_to_uri`. My benchmark shows a 35% speedup in routing and responses, 8% from replacing most calls, the rest from refactoring the implementations of the `iri` functions. , The only thing that still needed an (internal) wrapper was `urlencode`, since the router and test client might pass `MultiDict` or `dict` to it, and also expect `None` values to be discarded. Since I was replacing all uses of `quote`, I also took the opportunity to review what characters are being treated as safe from percent encoding. We were not being particularly consistent or correct about it.… Affected product modules include `examples/couchy/utils.py`, `examples/shortly/shortly.py`, `examples/shorty/utils.py`, `examples/simplewiki/utils.py`, `src/werkzeug/formparser.py`, `src/werkzeug/middleware/http_proxy.py`, `src/werkzeug/routing/converters.py`, `src/werkzeug/routing/map.py` (+8 more).

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
