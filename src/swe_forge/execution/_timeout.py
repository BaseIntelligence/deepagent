"""Shared in-container timeout enforcement for the Docker exec paths.

The async Docker exec helpers (``commands.exec_in_container`` and
``DockerClient.exec``) stream their output over an HTTP read that can block
indefinitely and is not reliably cancellable. To make the requested ``timeout``
actually enforcing -- and to reap the in-container process *tree* rather than
leaving a hung command running -- we wrap the in-container command under
coreutils ``timeout --signal=KILL <t>``. The Docker daemon launches ``timeout``
as the exec's main process; on expiry it SIGKILLs its whole process group,
killing the entire tree at the kernel level even when the Python-side read is
uncancellable. coreutils ``timeout`` exits with 124/137 on expiry, which the
caller detects to surface a clean timeout instead of a misleading result.
"""

from __future__ import annotations

from collections.abc import Sequence

# coreutils ``timeout`` exit statuses meaning the command was forcibly stopped
# because it ran past the deadline:
#   124 -> timed out and was signalled (the default TERM path)
#   137 -> sent SIGKILL (128 + 9); always the case with ``--signal=KILL``
TIMEOUT_EXIT_CODES = frozenset({124, 137})

# Deadlocked / busy-loop processes ignore SIGTERM, so we send SIGKILL, which the
# kernel cannot block, guaranteeing the in-container tree is reaped on expiry.
_KILL_SIGNAL = "KILL"

# Extra wall-clock budget beyond the requested timeout before the async read is
# force-abandoned. The in-container coreutils ``timeout`` reaps the process at
# the deadline and the stream then closes on its own, so this grace only bounds
# the pathological case where the HTTP read itself never returns after the reap.
EXEC_TIMEOUT_GRACE_SECONDS = 5.0

# Fraction of the requested timeout a forcibly-stopped command must have run for
# before a 124/137 exit is attributed to the injected timeout (rather than an
# unrelated OOM / external ``kill -9``). A coreutils-reaped command runs for
# roughly the full timeout, so it comfortably clears this floor.
_TIMEOUT_DURATION_FRACTION = 0.5


def wrap_command_with_timeout(cmd: Sequence[str], timeout: float) -> list[str]:
    """Prefix ``cmd`` with coreutils ``timeout --signal=KILL <timeout>``.

    On expiry ``timeout`` SIGKILLs its whole process group, reaping the
    in-container process tree at the kernel level. For a command that finishes
    in time ``timeout`` is transparent (it forwards stdout/stderr and the
    command's own exit code), so the normal, non-timeout contract is unchanged.
    """
    return ["timeout", "--signal", _KILL_SIGNAL, _format_duration(timeout), *cmd]


def outer_read_deadline(timeout: float) -> float:
    """Wall-clock budget for the async read before it is force-abandoned."""
    return timeout + EXEC_TIMEOUT_GRACE_SECONDS


def is_timeout_exit(exit_code: int, duration: float, timeout: float) -> bool:
    """Return True if the exit code + elapsed time indicate the timeout fired.

    Requires both a coreutils-``timeout`` exit status (124/137) *and* that the
    command actually ran for a meaningful fraction of the deadline, so an early
    OOM / external kill that happens to share exit 137 is not misread as a
    timeout.
    """
    if exit_code not in TIMEOUT_EXIT_CODES:
        return False
    return duration >= timeout * _TIMEOUT_DURATION_FRACTION


def _format_duration(timeout: float) -> str:
    """Format a seconds value for coreutils ``timeout`` (accepts fractions)."""
    if timeout <= 0:
        return "0"
    return f"{timeout:.3f}".rstrip("0").rstrip(".")
