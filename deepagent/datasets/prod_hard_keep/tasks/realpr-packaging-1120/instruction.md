# Represent version specifiers as intervals for faster filtering

The current implementation of `Specifier` and `SpecifierSet` filters candidate versions by iterating and testing each version against the specifier logic. For anything beyond trivial specifiers this scales poorly, and complex specifier sets become very expensive to evaluate.

Rework the internal mechanics so that a specifier (and a set of specifiers) is modeled as one or more **version ranges** (intervals with lower/upper bounds and inclusivity flags). Filtering and containment can then be computed via interval math rather than per-version iteration. This should meaningfully speed up moderately complex specifiers and dramatically speed up very complex ones, while keeping all existing public behavior identical.

## Expected outcomes
1. Introduce an internal range representation (e.g. a module providing a `VersionRange`-style type plus interval/set operations) capable of expressing the bounds implied by any PEP 440 specifier operator (`==`, `!=`, `~=`, `<`, `<=`, `>`, `>=`, `===`, and prefix/`.*` matching).
2. Convert individual `Specifier` operators into their equivalent range(s), and combine the specifiers in a `SpecifierSet` by intersecting their ranges.
3. Add an `is_unsatisfiable` capability that reports when a `SpecifierSet` reduces to an empty range (no version can ever satisfy it).
4. Keep the existing public API and results unchanged: `contains`, `filter`, `__contains__`, membership of prereleases, and equality/serialization must all behave exactly as before.
5. Design the range machinery so a future public `to_range()`-style API returning a manipulable PEP 440 range object could be layered on top cleanly.

## Constraints
- Do not change any public method signatures or observable outputs of `Specifier` / `SpecifierSet`.
- Preserve prerelease handling semantics (including `prereleases=` overrides and the default inclusion rules).
- `!=` and `~=` and prefix-match operators must produce the correct (possibly disjoint or split) interval representation.
- Correctness takes priority over micro-optimization; interval logic must exactly match the previous iterative behavior for all operator combinations.
- All existing tests must continue to pass without modification.

## Implementation notes
- Place the reusable interval/range primitives in a dedicated internal module rather than inline in the specifiers module.
- Add a benchmark script exercising simple, moderately complex, and very complex specifier sets so the performance characteristics of the interval approach can be measured against candidate version lists.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
