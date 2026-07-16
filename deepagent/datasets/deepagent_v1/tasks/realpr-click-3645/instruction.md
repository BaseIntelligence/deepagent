# Synchronize CLI Behavior Across Core, Decorators, and Shell Completion

The command-line framework's core module, decorator layer, and shell-completion logic have drifted out of alignment. Recent fixes and refinements that landed independently need to be reconciled so the public API behaves consistently across all three areas. Your task is to reconcile these modules so their shared behaviors, edge-case handling, and public contracts agree.

## Expected outcomes

1. Command and group construction in the core module correctly propagates all options, parameters, and context settings so that behavior matches what the decorator layer advertises when defining commands, groups, options, and arguments.
2. Decorators produce command objects whose attributes (name, help text, parameters, callbacks) are fully consistent with how the core executes and introspects them—no attribute set by a decorator should be silently ignored or overridden during command construction.
3. Shell completion returns correct suggestions for commands, subcommands, options, and argument values, including nested groups, and reflects any parameter metadata defined via the decorators.
4. Existing public function and class signatures remain backward compatible; callers relying on current behavior continue to work without changes.
5. The full existing test suite passes, and any behavior that was previously inconsistent between these modules is resolved in favor of the documented/intended behavior.

## Constraints

- Do not break the public API of the affected modules; keep exported names, signatures, and default argument values stable.
- Avoid introducing new external dependencies.
- Keep changes focused on reconciling behavior between core, decorators, and shell completion; do not perform unrelated refactors.
- Preserve support for nested command groups and lazily-loaded subcommands in completion output.
- Ensure completion logic degrades gracefully when a shell integration or completion context is unavailable.

## Implementation notes

- Pay attention to how parameter defaults, `expose_value`, hidden flags, and context settings flow from decorators into the constructed command objects, since these are common sources of divergence.
- When generating completions, resolve the active command path first, then enumerate the relevant parameters and choices from the resolved command rather than re-deriving them.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
