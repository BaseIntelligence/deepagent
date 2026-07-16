# Prepare `more_itertools` for the next release

We are cutting a new release of `more_itertools`. Update the library and its documentation so that the public API and version metadata are consistent and ready to ship.

## Expected outcomes

1. Remove the `derangements` and `distinct_derangements` functions entirely. Delete their implementations, their entries in `__all__`, any references in docstrings, and any documentation entries that expose them. After this change, importing these names must fail and they must not appear in the public API.
2. Bump the version metadata so the documentation reflects the new release version (10.6.0). Ensure the version string used by the docs configuration is updated accordingly.
3. Review and reformat the docstrings across the changed modules for consistency — consistent parameter descriptions, example formatting, and cross-reference style — without altering runtime behavior of the functions you keep.

## Constraints

- Do not silently deprecate `derangements` / `distinct_derangements` with a shim or warning; they should be removed cleanly as if they never existed in this release.
- Keep all other public functions and their behavior unchanged; only docstring wording/formatting may change for those.
- Ensure `__all__` remains sorted/consistent with the module's existing convention after removing entries.
- The documentation build must not reference the removed functions anywhere (autodoc directives, tables, or narrative text).
- Version metadata must be updated in exactly one authoritative place used by the docs so there is no mismatch.

## Implementation notes

The removed functions live in the main module; make sure to sweep both the implementation module and the recipes module for any lingering references (imports, `see also` cross-links, doctest examples). The docs configuration file holds the version string that needs bumping.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
