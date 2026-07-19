# Handle Grapheme Clusters in Cell Width Calculation

The library currently measures the display width of text on a per-codepoint basis. This breaks for modern Unicode text where a single visible character (a *grapheme cluster*) is composed of multiple codepoints — for example emoji with skin-tone or variation-selector modifiers, flag sequences, ZWJ (zero-width joiner) sequences, and combining marks. When these are measured codepoint-by-codepoint, the reported cell width is wrong, which corrupts alignment, wrapping, truncation, and table/panel rendering.

Improve width handling so that text is measured in terms of grapheme clusters rather than individual codepoints, producing correct terminal cell widths for composed characters.

## Expected outcomes

1. Text width measurement groups codepoints into grapheme clusters and returns the correct total cell width for the cluster (e.g. a base emoji plus modifiers occupies the width of the rendered glyph, not the sum of its parts).
2. Combining characters and zero-width joiners contribute zero additional width when part of a cluster.
3. Emoji sequences (ZWJ sequences, variation-selector-16 presentation, skin-tone modifiers, regional-indicator flag pairs) are treated as a single unit with the expected width.
4. Existing width results for plain ASCII, ordinary wide (CJK) characters, and simple combining cases remain correct — no regressions in alignment, wrapping, or truncation.
5. Unicode data tables needed for classification (cell-width ranges and grapheme-breaking properties) are available in a structured, version-addressable form so behaviour can track a chosen Unicode version.

## Constraints

- Keep the public width API stable; callers should not need to change how they request the width of a string.
- Do not add third-party runtime dependencies for Unicode handling.
- Generated Unicode data modules must be pure data (plain Python literals/tables) and importable without side effects.
- Performance for the common case (ASCII / short strings) must stay fast; avoid per-call table rebuilding or heavy allocation on hot paths.
- Maintain support across the currently supported Python versions.

## Implementation notes

- Provide a grapheme segmentation step that walks a string and yields clusters, then compute width by classifying each cluster once (not each codepoint independently).
- Store cell-width ranges and grapheme-break property data in dedicated data modules keyed by Unicode version, with a small registry so the active version can be selected.
- Prefer bisection over sorted range tables for codepoint lookups rather than large dict membership checks.
- Add tests covering plain text, wide CJK, combining marks, ZWJ emoji, flags, and variation selectors to lock in correct widths.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
