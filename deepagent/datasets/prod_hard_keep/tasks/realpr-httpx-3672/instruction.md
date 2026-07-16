# Improve Server Connection Handling

The HTTP server and parser layers need better handling of connection lifecycle, keep-alive semantics, and clean shutdown. Currently, requests aren't always fully consumed on keep-alive connections, the parser lacks a way to signal whether a connection should persist, and server shutdown leaves streams open and surfaces spurious exceptions.

Apply these changes consistently across both the synchronous (`httpx`) and asynchronous (`ahttpx`) implementations of the parser, pool, network, and server modules.

## Expected outcomes

1. Add a `keep_alive` property (or attribute) to `HTTPParser` that reports whether the parsed message indicates the connection should be kept alive, based on the HTTP version and `Connection` header semantics.
2. When handling a keep-alive connection, the server must always read the current request body to completion before processing the next request on the same connection, so leftover body bytes never corrupt subsequent request parsing.
3. Rename the parser's `complete` method to `reset`, updating all call sites. The renamed method should clear per-message parser state so the same parser instance can be reused for the next request on a persistent connection.
4. On server exit, ensure all active connection streams are closed cleanly rather than being left dangling.
5. Server shutdown must not surface a `KeyboardInterrupt` to the caller; shutting down via keyboard interrupt should terminate gracefully and quietly.

## Constraints

- Keep the sync and async implementations behaviorally identical; the async versions should mirror the sync logic with `await` where appropriate.
- Preserve existing public APIs other than the documented `complete` → `reset` rename; update every internal reference to the old name.
- Do not change the wire-level behavior of well-formed HTTP/1.0 and HTTP/1.1 requests beyond the keep-alive and completion handling described here.
- Reading a request to completion on keep-alive must not block indefinitely when there is no more data expected (e.g., no body or already fully consumed).

## Implementation notes

- `keep_alive` should default to the version-appropriate behavior: HTTP/1.1 keeps alive unless `Connection: close`, while HTTP/1.0 closes unless `Connection: keep-alive`.
- The graceful shutdown path should catch `KeyboardInterrupt` at the server's run/serve boundary, close streams, and return normally.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
