# Refactor cookie parsing/dumping and the test client cookie handling

The cookie utilities and the test client's cookie support need a modernization pass. The goal is faster, cleaner cookie handling that aligns with real browser behavior and drops the dependency on `http.cookiejar` in the test client.

## Expected outcomes

1. Reimplement `parse_cookie` and `dump_cookie` for efficiency, using a streamlined tokenizing approach. Results should be functionally equivalent to before while being significantly faster.
2. `dump_cookie` must NOT set `path="/"` by default. Clients already assume `/` when the attribute is omitted, so omitting it keeps `Set-Cookie` headers smaller. Only emit a `path` attribute when one is explicitly provided.
3. `dump_cookie` must not reject domains that lack a dot (e.g. `localhost` is valid). If the given domain has a leading dot, strip it. Whether a client does exact-origin matching should depend on whether a domain was provided at all, not on dot presence.
4. Rework the test client so cookie storage and selection no longer rely on `http.cookiejar`. Implement domain and path matching to decide which cookies are sent with a given request.
5. Add a `get_cookie` method to the test client to inspect a stored cookie. The returned `Cookie` object must expose `decoded_key` and `decoded_values` attributes holding the server-side (decoded) values.
6. `Cookie` objects must always carry a `domain` and `path`, defaulting to `localhost` and `/` respectively when not explicitly set.
7. Add an `origin_only` parameter to the test client's `set_cookie` to control whether the domain must match exactly. `origin_only=True` corresponds to a browser receiving a cookie with no domain attribute (exact-host matching).

## Constraints

- Deprecate passing `bytes` to `parse_cookie` and `dump_cookie`; these should accept text and warn when given bytes.
- Deprecate the `charset` parameter of `dump_cookie`. Encoding should move toward always using UTF-8; emit a deprecation warning when `charset` is used.
- Deprecate the leading positional `server_name` parameter of the test client's `set_cookie` and `delete_cookie`. Callers should use the `domain` parameter instead.
- Deprecate most parameters of `delete_cookie` (those that only made sense alongside `server_name`); deletion should be driven by `domain`/`path`.
- Emit `DeprecationWarning` for all deprecated paths, and keep the old behavior working until removal so existing callers don't break immediately.

## Implementation notes

- Keep the parse/dump refactor confined to the internal helpers and the sansio HTTP/response layers plus the public `http` module; avoid changing unrelated public signatures beyond the deprecations described.
- Verify against Firefox/Chrome behavior for `localhost` and dot-less domains: cookies should be settable and matched correctly.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
