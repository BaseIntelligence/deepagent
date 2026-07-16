# Make `kw_only=True` behave consistently with `dataclasses`

The `attrs` library currently over-applies class-level `kw_only=True`. When a class is decorated with `kw_only=True`, *every* attribute — including those inherited from base classes and those that explicitly set `kw_only=False` — is forced to be keyword-only. This diverges from the standard library `dataclasses`, where an explicit per-field opt-out is respected, and it silently ignores a user's intent.

Rework the keyword-only propagation so that class-level `kw_only=True` only affects the attributes defined directly on that class, and honors any field that explicitly opts out with `kw_only=False`, while preserving backward compatibility through an opt-in flag.

## Expected outcomes

1. When a class sets `kw_only=True`, only the attributes **defined on that class** that do not explicitly set `kw_only=False` become keyword-only. Inherited attributes retain the kw-only status they had on their defining class.
2. A field that explicitly sets `kw_only=False` remains positional even when its class is decorated with `kw_only=True`.
3. The field-level `kw_only` default changes from `False` to `None`. When `None`, an attribute's keyword-only status mirrors the value set at the class level; an explicit `True`/`False` always takes precedence.
4. Add a `force_kw_only` parameter to the class decorators. When `True`, the previous behavior is restored: all attributes (including inherited and `kw_only=False` ones) become keyword-only under a class-level `kw_only=True`.
5. `force_kw_only` defaults to `False` (new behavior) for the modern `define`-style API, and `True` (legacy behavior) for the classic `attr.s`-style API, so existing classic usage is unchanged by default.
6. Generated `__init__` signatures reflect the corrected ordering. For example, a modern-API class with `kw_only=True` and a field explicitly marked `kw_only=False` keeps that field positional, and normal positional-ordering rules (non-default before default) apply to it.

## Constraints

- Do not break construction call sites: attributes that were previously keyword-only and become positional must still accept keyword arguments.
- Explicit per-field `kw_only` values must never be overridden by class-level settings under the new (non-forced) behavior.
- Keep the legacy classic-API path behaving exactly as before by defaulting it to the forced mode.
- A class where an explicitly positional field with a default is followed by a non-default field must fail loudly at definition time with the usual attribute-ordering error, rather than silently producing an incorrect signature.
- Public type stubs (`.pyi`) for both the classic and modern APIs must expose the new `force_kw_only` parameter and the `kw_only` default of `None`, and the modern-API stubs must be re-exported consistently.

## Implementation notes

- Treat field-level `kw_only=None` as "inherit from class"; resolve it during attribute collection so inherited attributes keep the status computed on their own class.
- Add coverage demonstrating both the corrected inheritance behavior and the restored legacy behavior via `force_kw_only`, e.g. companion tests exercising kw-only inheritance with and without forcing.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
