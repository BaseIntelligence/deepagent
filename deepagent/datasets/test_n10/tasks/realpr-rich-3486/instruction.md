# Fine-Grained Error Location Highlighting in Tracebacks

Modern Python interpreters (3.11+) expose column-level position information for each frame in a traceback via PEP 657. When an exception occurs on a line containing multiple expressions or operations, the interpreter can pinpoint the exact span of code responsible. Our traceback rendering currently only shows the offending line without indicating *where* on that line the error originated.

Enhance the traceback formatter so it visually highlights the precise segment of source code that triggered the error, using the column offset data available on the frame objects.

## Expected outcomes

1. When rendering a traceback frame, the specific character range within the source line that caused the error is visually distinguished from the rest of the line (e.g. via a distinct style applied to that span).
2. A dedicated style entry exists for this highlight so users can theme or override it, consistent with how other traceback styles are defined.
3. The column position data is read from frame metadata when present and correctly maps to the start and end columns on the relevant line(s).
4. When the interpreter does not provide position data (older Python versions, or frames lacking the information), rendering falls back to the previous behavior with no highlight and no errors.
5. The syntax rendering path accepts and applies the highlighted range so the emphasis appears within the already-formatted source snippet.

## Constraints

- Guard all access to the new position attributes; they must not be assumed present. Behavior on Python versions without PEP 657 support must remain unchanged.
- Handle edge cases in the reported ranges gracefully: missing end offsets, ranges spanning line boundaries, and zero-width or out-of-bounds columns should not raise exceptions.
- Do not alter the existing structure or ordering of unrelated traceback output; the highlight is additive.
- Keep the new style overridable through the standard style definitions rather than hard-coding colors inline.

## Implementation notes

- The relevant column data lives on the frame/summary objects surfaced during traceback extraction; capture start line, end line, start column, and end column where available.
- Thread the highlight range through to the syntax renderer so it can apply the style to the correct token span when producing the code snippet.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
