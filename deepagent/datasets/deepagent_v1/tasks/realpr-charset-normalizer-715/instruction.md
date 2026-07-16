# Improve charset detection performance and fix output/scoring edge cases

This task covers a set of performance and correctness improvements to the charset detection library. The work touches the "mess detector" internals, candidate scoring, the command-line interface, and the Unicode range data.

## Expected outcomes

1. Refactor the mess-detection plugin architecture in `md.py` so per-character analysis follows a single, flattened code path. Replace the previous two-step `eligible(...)` / `feed(...)` interface on the detector plugins with a single combined method (e.g. `feed_info(...)`) that both determines eligibility and records the character in one call. All existing detector plugins and the aggregation logic that drives them must be updated to use the new interface, and the observable mess-ratio results must remain equivalent to before.

2. Fix a scoring edge case where the noise/mess difference between two candidate encodings can be almost insignificant. When two candidates are effectively tied on mess ratio, tie-breaking must be deterministic and produce the more appropriate result rather than being sensitive to negligible floating-point differences.

3. Fix the CLI `--normalize` behaviour when multiple input files are passed. Each normalized output must be written next to its own corresponding input file, not to a single shared or incorrect path. Verify that running normalization on several files at once produces one correctly named output per input.

4. Update the combined Unicode range table (`UNICODE_RANGES_COMBINED`) to reflect the current Unicode block definitions, keeping the range names and boundaries consistent with the rest of the detection logic.

5. Raise the supported upper bound for the `mypy`/`mypyc` build dependency so the optimized compiled build can use the newer toolchain.

## Constraints

- Preserve the public API of the package; the plugin interface change is internal only and must not alter externally observable detection results beyond the intended edge-case fixes.
- The flattened `md.py` logic should be measurably faster or at least no slower; do not regress detection accuracy on existing behaviour.
- The CLI fix must handle both single-file and multi-file invocations correctly.
- Keep the version identifier consistent across the package's version metadata.

## Implementation notes

- The mess-detector plugins each currently expose separate eligibility and feed steps; collapsing them removes redundant per-character branching and function-call overhead.
- For the tie-breaking fix, compare mess ratios with an appropriate tolerance and fall back to secondary, deterministic criteria when the difference is within that tolerance.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
