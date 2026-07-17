# Add `encodeDotKeys` / `decodeDotKeys` options for dot-in-key handling

The query-string library needs a way to round-trip object keys that contain literal `.` characters. Currently, when serializing nested objects, dots are used as structural separators (or brackets, depending on options), so a key like `foo.bar` inside an object cannot be distinguished from nested access when using dot notation. We want opt-in support to encode literal dots in keys during stringification and decode them back during parsing.

## Expected outcomes

1. Add a new `encodeDotKeys` boolean option to the stringify function. When enabled, any literal `.` character that appears within an object key (as opposed to a structural separator) is percent-encoded so that it survives a round trip and is not interpreted as a nesting delimiter.
2. Add a corresponding `decodeDotKeys` boolean option to the parse function. When enabled, encoded dots within keys are decoded back into literal `.` characters after the structure has been resolved, restoring the original key names.
3. A value stringified with `encodeDotKeys: true` and then parsed with `decodeDotKeys: true` must reproduce the original object, including keys that contain dots.
4. Both options default to `false`, preserving all existing behavior for callers that do not opt in.

## Constraints

- Do not change the default output of `stringify` or the default result of `parse`; the new behavior must be strictly opt-in via the new option flags.
- `encodeDotKeys` must only affect literal dots that are part of a key name, never structural dots used for nesting notation.
- `decodeDotKeys` should operate on key names only and must not alter values or structural parsing.
- Validate the option types where the library already validates other boolean options, and reject invalid combinations consistently with existing option-validation behavior.
- Keep the two options independent of the existing dot-notation handling so they can be combined with other configuration without surprising interactions.

## Implementation notes

- Focus changes on the parse and stringify modules.
- Consider how dot-encoding interacts with the existing `allowDots` behavior and make sure the two features compose sensibly rather than conflicting.
- Add tests covering keys with single and multiple dots, keys that are only a dot, and full round-trip scenarios combining both options.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
