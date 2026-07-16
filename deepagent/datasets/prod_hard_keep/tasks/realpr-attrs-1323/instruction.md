# Deprecate the `hash` argument in favor of `unsafe_hash`

Our class-decorator API currently exposes a `hash` argument for controlling automatic `__hash__` generation. To align with the naming used elsewhere in the ecosystem (and to make the danger of the operation explicit), we want the parameter to be called `unsafe_hash` instead. The old name should continue to work for now, but should surface a deprecation warning so users have time to migrate.

## Expected outcomes

1. Add an `unsafe_hash` argument to the relevant class-building decorator(s) that behaves identically to the existing `hash` argument.
2. When a caller passes `hash`, emit a `DeprecationWarning` explaining that `hash` is deprecated and that `unsafe_hash` should be used instead. The warning must point at the caller's code (appropriate stack level).
3. Passing both `hash` and `unsafe_hash` in the same call is an error and should raise a clear exception describing the conflict.
4. When neither is passed, behavior is unchanged (hashing decision falls back to the existing default logic).
5. `unsafe_hash` accepts the same value domain as the old `hash` argument (`True`, `False`, or `None`) and produces the same resulting `__hash__` behavior for each value.

## Constraints

- Do not remove or break the existing `hash` argument; existing code that relies on it must keep working (just with a warning).
- The deprecation warning text should be user-actionable and mention both the deprecated name and the replacement.
- Keep the public signatures backwards compatible; only additive changes plus the warning are expected.
- Ensure the internal machinery that decides whether to generate `__hash__` reads from a single normalized value, regardless of which argument name the user supplied.

## Implementation notes

- Normalize the two arguments into one internal value early, raising on conflict before any hashing logic runs.
- Use `warnings.warn(..., DeprecationWarning, stacklevel=...)` with a stacklevel that reports the user's call site rather than internal frames.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
