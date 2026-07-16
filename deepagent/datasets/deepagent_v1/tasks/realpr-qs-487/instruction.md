# Add `allowEmptyArrays` option for query string serialization

Our query string library currently drops empty arrays during stringification. When an object contains a key whose value is an empty array, that key disappears entirely from the output rather than being represented. Some consumers need a way to preserve these empty arrays so the serialized form round-trips more faithfully.

Introduce a new `allowEmptyArrays` option that, when enabled, keeps empty-array values in the output instead of silently omitting them.

## Expected outcomes

1. Add an `allowEmptyArrays` option (boolean) to the stringify entry point. It defaults to `false`, preserving current behavior — empty arrays are omitted from the output.
2. When `allowEmptyArrays` is set to `true`, a key mapped to an empty array must be emitted with an empty bracket notation (e.g. `foo[]=`), rather than being dropped.
3. The option must interact sensibly with existing options (e.g. `encode`, `arrayFormat`, nested objects). Keys with empty arrays should still respect encoding and prefixing rules.
4. Parsing behavior should remain consistent so that supported serialized forms continue to be interpreted correctly.

## Constraints

- Do not change the default output for any existing input; the new behavior must be opt-in only.
- Non-empty arrays and all other value types must serialize exactly as before, regardless of the new option's value.
- Validate the option as a boolean and fall back to the default when it is not explicitly provided.
- Keep the option threaded correctly through recursive/nested serialization so empty arrays are preserved at any depth.

## Implementation notes

- The relevant logic lives in the stringify path; the empty-array short-circuit that currently returns nothing should be gated on the new option.
- Add test coverage demonstrating both states: with the option off (empty arrays omitted) and on (empty arrays retained as `key[]=`).

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
