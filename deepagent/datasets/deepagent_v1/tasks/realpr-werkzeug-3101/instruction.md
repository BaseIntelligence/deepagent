# Improve File Handling and Resource Management in Werkzeug Test Utilities

The test client's `EnvironBuilder`, the multipart form parser, and `FileMultiDict` currently leak open file handles, producing `ResourceWarning`/pytest unraisable warnings during tests. The interactions between `form`, `files`, `input_stream`, `args`, and `query_string` on `EnvironBuilder` are also confusing and rely on scattered `None` checks. This task cleans up file lifecycle management and clarifies these attribute relationships.

## Expected outcomes

1. `EnvironBuilder` supports use as a context manager (`with EnvironBuilder(...) as builder:`). On exit, all files in `builder.files` are closed. `builder.input_stream` is intentionally *not* closed, since it is assumed to be managed by the caller.
2. Setting `input_stream` clears the `form` and `files` multidicts; setting `form`/`files` (or otherwise populating a request body) is mutually exclusive with a raw `input_stream`. Clearing the `files` multidict must close the file handles it held.
3. The same mutual-exclusion behavior applies between `args` and `query_string`: setting one clears/derives the other cleanly rather than juggling `None` sentinels.
4. `FileMultiDict` gains `close()` and `clear()` behavior so that dropping stored files closes them. `FileMultiDict.add_file` must detect a filename from a file-like object when one isn't given, and use it to guess the content type — matching the detection already done by `FileStorage`. Factor this detection so both code paths share it.
5. `MultiPartParser` works as a context manager. It tracks the files it opens while parsing; if the context exits due to an exception (e.g. a stream that ends partway through a multipart file), the partially-created files are closed so they don't leak.
6. `Request.from_values` and any internal test helpers that build environments with files must use the context-manager form so files are reliably closed.
7. Remove the legacy `SpooledTemporaryFile` compatibility shim (the old `TemporaryFile` + `BytesIO` reimplementation for environments lacking `SpooledTemporaryFile`). Always use the standard library `SpooledTemporaryFile`.
8. Running the full test suite produces no `ResourceWarning`s and no pytest unraisable warnings.

## Constraints

- Preserve existing public behavior for parsing and request construction; the premature-end-of-file case must still be handled the same way (error raised and silenced higher up), just without leaking a file handle.
- `input_stream` must remain caller-managed and never auto-closed by the builder.
- Keep backward-compatible construction: passing `form`/`files`, `data`, or `input_stream` should continue to work, with the mutual-exclusion rules enforced consistently.
- Do not drop support for guessing content type from a filename where it already worked.

## Implementation notes

Update docstrings for `EnvironBuilder` and the affected attributes/args to clearly describe how `form`, `files`, `input_stream`, `args`, and `query_string` relate and override one another. Centralize filename/content-type detection so `FileMultiDict.add_file` and `FileStorage` don't diverge.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
