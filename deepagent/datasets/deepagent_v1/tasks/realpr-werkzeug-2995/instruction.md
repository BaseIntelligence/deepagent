# Restrict container types accepted by multi-value helpers

The header and multi-dict data structures include helpers (such as `iter_multi_items` and related methods) that decide whether a supplied value represents multiple items to iterate over, or a single value to store as-is.

The current logic for detecting "multiple items" is inconsistent with its type annotations and misbehaves on binary-like types. An overly broad check using `Container` incorrectly treats values like `bytes`, `bytearray`, `memoryview`, and `array` as iterables of items, when each of these should be stored as a single value. Conversely, an overly narrow check (only `list` and `tuple`) omits `set`, which is a reasonable built-in collection to pass in.

## Expected outcomes

1. When a value is a `list`, `tuple`, or `set`, the helper treats it as a collection and iterates over its members to produce multiple items.
2. When a value is any other type — including `str`, `bytes`, `bytearray`, `memoryview`, `array`, or arbitrary objects — it is treated as a single value and stored/yielded intact.
3. The type annotations on the affected methods accurately reflect the runtime behaviour (i.e. they should not claim to accept an arbitrary `Iterable` when only specific collection types are handled as multi-value inputs).

## Constraints

- Do not attempt to build an allow/deny list of binary or sequence types; restrict acceptance to the concrete built-in collection types `list`, `tuple`, and `set`.
- Preserve existing behaviour for `str` and for scalar values; they must continue to be treated as single values.
- Keep the changes confined to the header and multi-dict data structure modules and their helper logic.

## Implementation notes

Replace any `isinstance(value, Container)`-style detection with a direct `isinstance(value, (list, tuple, set))` check, and update the corresponding annotations so they match what is actually accepted.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
