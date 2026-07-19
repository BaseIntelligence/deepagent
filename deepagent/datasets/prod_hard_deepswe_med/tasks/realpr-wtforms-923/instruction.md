# SelectField Backward Compatibility and Choice Callback Enhancements

The form field processing pipeline needs a follow-up hook, and the `SelectField` / `DataList` choice handling needs to be more flexible while preserving compatibility with older WTForms behavior that downstream projects depend on.

## Problem

Currently `SelectField` and `DataList` resolve their `choices` too early, so callable choices cannot inspect sibling field data. In addition, recent refactors to the choice representation broke behaviors that external projects rely on: `iter_choices` returning plain tuples, `render_options` accepting `(value, label, selected)`, and the `has_groups`/`iter_groups` helpers for optgroup rendering. Choices also cannot currently be supplied as a dict.

## Expected outcomes

1. Add a `post_process` step that runs immediately after `process` for fields, and propagate the call through the form-level processing so nested and enclosed fields are also post-processed.
2. Allow the `choices` argument of `SelectField` and `DataList` to be a callable, and support callables that optionally accept `form` and/or `field` parameters. Resolve these callables during `post_process` so the callback can read the `.data` of other fields on the form.
3. Support supplying `choices` as a dict (mapping used to build the option list / optgroups), in addition to the existing iterable forms.
4. Restore `has_groups()` and `iter_groups()` so grouped (optgroup) choices render correctly.
5. Maintain backward compatibility for consumers that:
   - expect `iter_choices` to yield tuples, and
   - call `render_options` with positional `value, label, selected` rather than a single choice object.
6. Represent an individual choice using a `NamedTuple`-based type so choices behave like tuples (indexable/unpackable) while still carrying named fields.

## Constraints

- A `NamedTuple` cannot override `__new__`; introduce a small private helper type to provide default values for the tuple members rather than fighting the NamedTuple machinery.
- Deprecated tuple/positional behaviors must continue to work but should be clearly marked as deprecation-era compatibility so they can be removed cleanly in a future major release. Emit deprecation warnings where appropriate.
- Callable-choice resolution must not run during `__init__`; it must be deferred to `post_process` so cross-field data is available.
- Do not break the existing public API: existing calls to `SelectField(...)`, `DataList(...)`, `iter_choices()`, and `render_options()` must keep working unchanged.
- Keep the callable-arg detection robust (inspect signature/arity) so that callbacks taking zero, one, or two of `form`/`field` all work.

## Implementation notes

- Touch the field processing core so `post_process` is a first-class step; ensure the base `Form.process` invokes it on all bound fields, including `FormField` and `FieldList` children.
- Keep the compatibility shims isolated and commented so they are easy to locate and delete later.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
