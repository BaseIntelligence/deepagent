# Move structured HTTP header parsing into the header classes

The structured header types in `werkzeug.datastructures` (things like `Accept`, `RequestCacheControl`, `ContentRange`, `ContentSecurityPolicy`, `ETags`, `IfRange`, `Range`, `HeaderSet`) currently rely on standalone `parse_*_header` functions living in `werkzeug.http`. This split creates circular imports between `http` and the datastructures modules and complicates type annotations, since those parse functions only ever return the header classes themselves — there's no simpler intermediate representation that justifies keeping the logic separate.

`Authorization` already follows the better pattern: it owns a `from_header` classmethod and a `to_header` method. Bring the remaining structured header classes in line with that design.

## Expected outcomes

1. Each of the following structured header classes gains a `from_header` classmethod that parses a raw header string (returning `None` for empty/absent input where the old function did) and produces an instance of that class:
   - `Accept` (replacing `parse_accept_header`)
   - `RequestCacheControl` (replacing `parse_cache_control_header`)
   - `ContentRange` (replacing `parse_content_range_header`)
   - `ContentSecurityPolicy` (replacing `parse_csp_header`)
   - `ETags` (replacing `parse_etags`)
   - `IfRange` (replacing `parse_if_range_header`)
   - `Range` (replacing `parse_range_header`)
   - `HeaderSet` (replacing `parse_set_header`)
2. `ContentSecurityPolicy` gains a `to_header` method, replacing `dump_csp_header`.
3. All internal callers — including `sansio.request`, `sansio.response`, `sansio.http`, `wrappers.response`, `serving`, and the datastructures modules themselves — use the new class methods instead of the old module-level functions.
4. The parsing/dumping behavior is preserved exactly: the same inputs yield the same parsed values and the same serialized header strings as before.
5. Circular imports between `werkzeug.http` and the datastructures modules are eliminated as a result of the move.

## Constraints

- Public access through `Request` and `Response` attributes must continue to work unchanged; do not alter those attribute APIs or their return types.
- Where an `Accept` subtype or specific cache-control variant was returned before (e.g. request vs. response cache control), preserve that same behavior via the appropriate class's `from_header`.
- Keep argument names and semantics consistent with the previous parse functions (e.g. optional `on_update` callbacks, class selection for `Accept`).
- Do not change the serialized output format of any `to_header` implementation.

## Implementation notes

- The old `parse_*` functions were thin wrappers that constructed and returned these classes, so the logic can move directly onto the classes with minimal transformation.
- Prefer classmethods that accept the raw header value as their first parameter, mirroring the existing `Authorization.from_header` signature.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
