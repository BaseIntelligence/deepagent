# Add `allowEmptyArrays` Option to `parse` and `stringify`

The query-string library needs to support representing empty arrays as values within objects. Currently, when an object contains a property whose value is an empty array, there is no way to preserve that emptiness through a `stringify`/`parse` round-trip — the empty array simply disappears or is not encoded in a recoverable way.

Add a new boolean option, `allowEmptyArrays`, to both the `parse` and `stringify` functions so that empty arrays can be explicitly emitted and correctly reconstructed.

## Expected outcomes

1. `stringify` accepts an `allowEmptyArrays` option (default `false`). When enabled and a property value is an empty array, the output includes a representation for that key that encodes its empty-array nature (e.g. `foo[]`), rather than silently omitting the key.
2. `parse` accepts an `allowEmptyArrays` option (default `false`). When enabled, a key encoded as an empty array (e.g. `foo[]` with no value) is parsed back into an empty array `[]` rather than an empty string, `undefined`, or a single-element array.
3. With both options enabled, a value like `{ foo: [] }` survives a full `stringify` → `parse` round-trip and remains an empty array.
4. When the option is `false` or omitted, existing behavior is unchanged for all inputs, including objects that contain empty arrays.

## Constraints

- The option name must be exactly `allowEmptyArrays` and must default to `false` in both `parse` and `stringify`.
- Validate the option: if a non-boolean value is supplied, throw a `TypeError` with a clear message, consistent with how other boolean options in these functions are validated.
- Do not alter the treatment of non-empty arrays under any array-format setting; the new behavior applies only to arrays with zero elements.
- Preserve compatibility with the other existing options (array formats, filtering, encoding, `strictNullHandling`, etc.) — `allowEmptyArrays` must compose cleanly with them.
- Keep the encoded form consistent between `stringify` and `parse` so round-trips are lossless.

## Implementation notes

- In `stringify`, detect the empty-array case before the normal array-serialization path and emit the appropriate key marker.
- In `parse`, recognize the empty-array marker during key/value processing and assign an empty array to the resulting object.
- Read the option from the normalized options object alongside the other boolean flags and apply the same validation pattern.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
