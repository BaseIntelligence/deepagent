# Improve subdomain and host matching in the routing map

The URL routing `Map` needs clearer, more predictable control over how a request's `Host` header influences matching. Currently subdomain handling is entangled with host detection and can produce surprising results (for example, forcing an `"<invalid>"` subdomain when detection fails, or letting a bound `server_name` restrict routing even when host matching is active).

## Expected outcomes

1. `Map.__init__` accepts a new `subdomain_matching` keyword argument that controls whether the request's subdomain is considered during routing. It defaults to enabled so existing behavior is preserved.
2. When both `subdomain_matching` and `host_matching` are disabled, the request's `Host` header does not factor into routing at all — matching proceeds without deriving or comparing any subdomain/host component.
3. In `bind_to_environ`, the `server_name` value is ignored when `host_matching` is enabled, so binding does not incorrectly restrict routing to a single host. When `host_matching` is disabled, `server_name` continues to be used as before to derive the subdomain.
4. When `subdomain_matching` is enabled but a subdomain cannot be detected from the environment, fall back to `default_subdomain` if it is set, instead of unconditionally substituting `"<invalid>"`. The `"<invalid>"` placeholder should only be used when no usable default is available.

## Constraints

- Keep the default behavior backward compatible: a `Map` constructed without the new argument must behave exactly as it does today.
- Do not break the interaction between `host_matching` and `subdomain_matching`; the two flags represent distinct, non-overlapping strategies and should not both drive matching simultaneously in a conflicting way.
- Confine changes to the routing map and rule logic; do not alter unrelated public API surfaces.
- Preserve existing type hints and follow the module's current style.

## Implementation notes

- The `subdomain_matching` flag lives on the `Map` instance and should be threaded through to the binding/matching paths where the subdomain is derived and compared.
- Review the code path in `bind_to_environ` that computes the effective subdomain from `server_name` and the request host, and gate the `server_name`-based restriction behind the `host_matching` check.
- Add focused tests covering: subdomain matching disabled + host matching disabled (Host ignored), `server_name` ignored under host matching, and the `default_subdomain` fallback when detection fails.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
