# Fix round-tripping of encoded dots in keys

The query-string library supports an `allowDots` option that lets nested objects be represented using dot notation in keys (e.g. `a.b=c` ↔ `{ a: { b: 'c' } }`). However, when a key itself contains a literal dot, round-tripping is broken: stringifying and then re-parsing does not recover the original key, because the literal dot in the key is indistinguishable from a dot used as a nesting separator.

We need a way to encode literal dots in keys during `stringify` and decode them back during `parse`, so that keys containing dots survive a full round trip.

## Expected outcomes

1. Add an `encodeDotInKeys` option to `stringify`. When enabled, literal dots that appear inside key names are percent-encoded (as `%2E`) so they are not later interpreted as nesting separators.
2. Add a `decodeDotInKeys` option to `parse`. When enabled, the encoded dots produced above are decoded back into literal dots in the resulting key names.
3. With both options enabled alongside `allowDots`, a value like `{ "name.obj": { first: "John", last: "Doe" } }` must round-trip exactly:
   ```js
   qs.parse(
     qs.stringify({ "name.obj": { first: "John", last: "Doe" } }, { allowDots: true, encodeDotInKeys: true }),
     { allowDots: true, decodeDotInKeys: true }
   )
   // => { 'name.obj': { first: 'John', last: 'Doe' } }
   ```
4. Without these options, the existing (lossy for dotted keys) behavior stays unchanged.

## Constraints

- `encodeDotInKeys` and `decodeDotInKeys` default to `false`; existing behavior must be fully preserved when they are not set.
- Encoding of literal dots applies only to key names, not to values, and only to the key segments — dots used as legitimate nesting separators must remain functional.
- Validate the interaction with `allowDots`: enabling `encodeDotInKeys`/`decodeDotInKeys` implies dot-based nesting is in play, so guard against contradictory option combinations and throw a clear `TypeError` when they conflict (e.g. `encodeDotInKeys: true` without dot handling enabled).
- Nested keys that themselves contain dots must be distinguished from nesting dots after decoding — the decoded literal dot must never be re-split into further nesting.
- Keep both `parse` and `stringify` symmetric so any input stringified with `encodeDotInKeys: true` parses cleanly with `decodeDotInKeys: true`.

## Implementation notes

- Use `%2E` as the encoded form of a literal dot so it is preserved through standard percent-decoding paths.
- Thread the new options through the existing option-normalization/defaults logic in both `parse` and `stringify`, and update any relevant option validation.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
