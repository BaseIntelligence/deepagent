# Preserve buffered response data during connection Upgrade and CONNECT

When handling HTTP/1.1 connection upgrades (via the `Upgrade` header) or `CONNECT` tunneling, the HTTP/1.1 transport hands the raw underlying stream back to the caller so they can continue reading/writing on it directly. The current implementation assumes no response body data has already been buffered internally at that point. This assumption breaks when the parser has already read bytes past the response headers into its internal buffer: those bytes are silently lost when the stream is handed off, corrupting the caller's view of the connection.

Fix the handoff so that any data already read into the internal buffer is not discarded and is made available to the consumer of the upgraded/tunneled stream.

## Expected outcomes

1. For a `101 Switching Protocols` response with connection upgrade, if the HTTP/1.1 reader has buffered bytes beyond the response headers, those buffered bytes must be delivered to the caller before any further reads from the raw network stream.
2. For a successful `CONNECT` request establishing a tunnel, any bytes already buffered internally must likewise be preserved and surfaced ahead of subsequent stream reads.
3. Reading from the returned network stream after an upgrade/CONNECT yields the exact byte sequence the server sent, regardless of how the internal buffer was populated during header parsing.
4. The behavior is identical across the async and sync implementations.

## Constraints

- Apply the fix consistently to both the async and sync HTTP/1.1 code paths so they stay in lockstep.
- Do not change the public API or the type of object returned to the caller for upgrade/CONNECT scenarios.
- Behavior for the common case (no leftover buffered data) must remain unchanged.
- Add tests covering both the upgrade and CONNECT paths where the internal buffer contains data at handoff time, and verify the caller reads the correct bytes.

## Implementation notes

- The internal read buffer used during header/line parsing is the source of the leftover bytes; wrap or prepend it so those bytes are yielded first when the caller reads from the handed-off stream.
- Ensure both `httpcore/_async/http11.py` and `httpcore/_sync/http11.py` receive equivalent changes; the sync module is typically generated/mirrored from the async one, so keep them in sync.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
