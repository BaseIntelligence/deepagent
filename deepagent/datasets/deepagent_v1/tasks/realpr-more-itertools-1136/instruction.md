Prepare the codebase for the next major release by bumping the package version and ensuring the public API surface is consistent and correct across the module.

## Expected outcomes
1. The package version string is updated to `11.0.0` wherever it is exposed (e.g., `__version__` in the package's `__init__.py`).
2. All public functions defined in the `more` and `recipes` modules are correctly exported via `__all__`, and there are no stale entries referencing removed or renamed callables.
3. Importing the package and inspecting its version reports `11.0.0`.
4. The existing test suite continues to pass without regressions.

## Constraints
- Do not change the runtime behavior of existing iterators, recipes, or helper functions unless required to keep `__all__` and actual definitions in sync.
- Keep the version string in a single canonical location if the project already centralizes it; avoid duplicating it inconsistently across files.
- Maintain backward compatibility for all documented, public names.
- Follow the existing code style and formatting conventions used throughout the modules.

## Implementation notes
- Verify that every name listed in `__all__` resolves to an actual object in the module, and that every intended public object appears in `__all__`.
- If the version is referenced elsewhere (documentation strings, metadata helpers), ensure those references are consistent with the new value.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
