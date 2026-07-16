# Implement `|` and `|=` operators for MultiDict and Headers

The `MultiDict` and `Headers` datastructures should support the union operators `|` and `|=`, mirroring the semantics that Python's built-in `dict` gained. Right now these types can only be merged via method calls; they should behave naturally with the operators too, while respecting the immutability and identity guarantees of their subclasses.

## Expected outcomes

1. `MultiDict` and `Headers` support `a | b`, returning a new instance that combines the contents of both operands. The `|` operator only accepts another mapping on the right-hand side; anything else should result in `NotImplemented` (raising `TypeError` at the operator level).
2. `MultiDict` and `Headers` support `a |= b` for in-place updates. Unlike `|`, the in-place variant is more permissive: it accepts either a mapping or an iterable of key/value pairs, matching what the corresponding `update` behavior already allows.
3. `UpdateDictMixin` implements `|=` so that in-place union routes through its update mechanism (triggering the on-update callback where applicable).
4. `ImmutableDictMixin` and `ImmutableHeadersMixin` disallow `|=`, raising the same immutability error these types already raise for mutating operations.
5. `EnvironHeaders` disallows both `|` and `|=`. Since its `copy` cannot produce a mutable independent instance, the non-mutating `|` operator is also unsupported for this type.

## Constraints

- `|` must not mutate either operand; it produces a fresh object of the appropriate type.
- The result type and preserved multi-value semantics of `|` on a `MultiDict` should match the type's normal copy/update behavior (multiple values per key are retained).
- Right-hand operands that are not valid for a given operator should yield `NotImplemented` rather than a bespoke error, so Python's operator protocol can fall back correctly.
- Immutable variants must reject `|=` consistently with their existing mutation-blocking exceptions; do not silently succeed.

## Implementation notes

- Add `__or__` and `__ior__` to the relevant classes in the datastructures package (headers, mixins, and the core structures module).
- For `__or__`, start from a copy of the left operand and apply the right mapping; for `__ior__`, delegate to the existing update path.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
