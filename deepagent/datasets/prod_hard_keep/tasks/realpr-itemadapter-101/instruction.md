# Add JSON Schema generation for item classes

`itemadapter` supports several item types (dictionaries, `scrapy.Item`, `dataclass`-based classes, `attrs`-based classes, and Pydantic models). We want a way to introspect an item class and produce a JSON Schema document describing its structure. This is useful for validation, documentation, and interoperability tooling.

Add a `get_json_schema()` function that, given a supported item class, returns a JSON Schema (`dict`) describing that class's fields and their types.

## Expected outcomes

1. A public `get_json_schema(item_class)` callable is available from `itemadapter.utils` (and re-exported at package level) that accepts a supported item class and returns a JSON Schema as a plain `dict`.
2. The result has `"type": "object"` at the top level, with a `"properties"` mapping keyed by field name.
3. Field Python types are mapped to their JSON Schema equivalents where determinable (e.g. `str` → `{"type": "string"}`, `int` → `{"type": "integer"}`, `float` → `{"type": "number"}`, `bool` → `{"type": "boolean"}`, `list`/sequence types → `{"type": "array"}`, `dict`/mapping types → `{"type": "object"}`, `None`/`Optional` handled appropriately).
4. Container types with parameterized element types (e.g. `list[int]`) populate the corresponding nested schema (e.g. `"items"`).
5. Field metadata that maps naturally onto JSON Schema keywords is honored — e.g. per-field constraints such as minimum/maximum, min/max length, enum choices, descriptions, titles, and default values are emitted where available from the underlying item definition.
6. Required fields (those without defaults) are collected into a top-level `"required"` list.
7. The function works consistently across all supported item types, deriving field information via the appropriate adapter for each class.

## Constraints

- Do not raise on field types that cannot be represented in JSON Schema; skip or fall back gracefully to a permissive schema for those fields rather than failing the whole call.
- Regular-expression patterns that cannot be safely represented as JSON Schema `pattern` strings should be ignored rather than emitted incorrectly.
- Recursive or self-referential type definitions (a class with a field of its own type) do not need full `$ref` support in this iteration; avoid infinite recursion by handling such cases defensively.
- Keep type-detection imports isolated so the feature degrades gracefully when optional dependencies (attrs, pydantic, scrapy) are not installed.
- Do not change existing public adapter behavior; this is additive.

## Implementation notes

- Centralize the schema-building logic (e.g. in a dedicated module) and expose it through the existing adapter/utility layer so each supported item type reuses the same conversion code.
- Reuse the existing adapter machinery to enumerate fields and read per-field metadata rather than special-casing each item type inline.
- Add tests covering each supported item type and the mapping of primitive, container, optional, and constrained fields.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
