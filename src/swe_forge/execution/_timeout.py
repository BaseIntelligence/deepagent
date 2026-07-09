"""Shared timeout provenance and process-tree reaping for Docker execs.

Exit statuses are target-owned data.  In particular, a target can intentionally
return 124 or 137, so neither exec path may infer a timeout from an exit code
or elapsed duration.  Instead, the wrapper starts the target in a new session
and its watchdog emits one opaque marker immediately before killing that session.
Only that marker establishes deadline provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

# The random suffix makes an accidental target-produced marker infeasible. The
# marker is a complete stderr line, removed before normal output is returned.
TIMEOUT_MARKER_PREFIX = "__SWE_FORGE_EXEC_TIMEOUT__:"

# Extra wall-clock budget beyond the requested timeout before the async read is
# force-closed. The watchdog reaps the session at the deadline and the stream
# normally closes on its own. This only bounds a broken HTTP stream.
EXEC_TIMEOUT_GRACE_SECONDS = 5.0


def new_timeout_marker() -> str:
    """Return one opaque marker that only this invocation's watchdog emits."""
    return f"{TIMEOUT_MARKER_PREFIX}{uuid4().hex}"


def wrap_command_with_timeout(
    cmd: Sequence[str], timeout: float, *, marker: str | None = None
) -> list[str]:
    """Wrap ``cmd`` in a new session with a marker-emitting kill watchdog.

    A supervising shell launches the target under ``setsid --wait`` and records
    the target's session-leader PID. Its watchdog prints ``marker`` before
    SIGKILLing that target process group on expiry, reaping the direct command
    and ordinary descendants without killing the watchdog before it can publish
    provenance. The exact marker, not a target exit status, is the timeout
    proof. ``cmd`` remains positional arguments after ``--`` so no target
    argument is shell-expanded.
    """
    watchdog_marker = marker or new_timeout_marker()
    script = "".join(
        (
            "marker=$1 timeout=$2 pidfile=$3; shift 3; ",
            'rm -f "$pidfile"; ',
            "setsid --wait sh -c "
            '\'echo "$$" > "$1"; shift; exec "$@"\' '
            '"swe-forge-exec-target" "$pidfile" "$@" & ',
            "runner=$!; ",
            'until [ -s "$pidfile" ] || ! kill -0 "$runner" 2>/dev/null; do '
            "sleep 0.01; done; ",
            'if [ ! -s "$pidfile" ]; then wait "$runner"; exit $?; fi; ',
            'target_pid=$(cat "$pidfile"); ',
            '(sleep "$timeout"; printf "%s\\n" "$marker" >&2; '
            'kill -KILL -"$target_pid" 2>/dev/null || :) & ',
            "watchdog=$!; ",
            'wait "$runner"; status=$?; ',
            'kill "$watchdog" 2>/dev/null || :; ',
            'wait "$watchdog" 2>/dev/null || :; ',
            'rm -f "$pidfile"; ',
            'exit "$status"',
        )
    )
    return [
        "sh",
        "-c",
        script,
        "swe-forge-exec-watchdog",
        watchdog_marker,
        _format_duration(timeout),
        f"/tmp/.swe-forge-exec-{watchdog_marker}.pid",
        *cmd,
    ]


def outer_read_deadline(timeout: float) -> float:
    """Wall-clock budget for the async read before it is force-abandoned."""
    return timeout + EXEC_TIMEOUT_GRACE_SECONDS


def remove_timeout_marker(output: str, marker: str) -> tuple[str, bool]:
    """Remove watchdog marker lines and report whether this watchdog fired."""
    lines = output.splitlines(keepends=True)
    kept = [line for line in lines if line.rstrip("\r\n") != marker]
    return "".join(kept), len(kept) != len(lines)


def _format_duration(timeout: float) -> str:
    """Format a seconds value for coreutils ``timeout`` (accepts fractions)."""
    if timeout <= 0:
        return "0"
    return f"{timeout:.3f}".rstrip("0").rstrip(".")
