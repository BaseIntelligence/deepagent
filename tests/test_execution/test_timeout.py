"""Offline timeout provenance and cancellation-cleanup tests.

The shell watchdog emits an opaque marker only when *it* reaches the deadline.
Consequently a target that returns 124 or 137 is an ordinary command result,
while a watchdog marker raises the documented timeout.  The response reader is
also closed and awaited on both deadline and outer-call cancellation.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from swe_forge.execution import _timeout
from swe_forge.execution._timeout import (
    TIMEOUT_MARKER_PREFIX,
    outer_read_deadline,
    remove_timeout_marker,
    wrap_command_with_timeout,
)
from swe_forge.execution import commands
from swe_forge.execution.commands import exec_in_container


class TestTimeoutHelpers:
    """Pure-function tests for the shared timeout helpers."""

    def test_wrap_starts_a_new_session_and_preserves_command(self):
        marker = f"{TIMEOUT_MARKER_PREFIX}test"
        wrapped = wrap_command_with_timeout(
            ["bash", "-c", "echo hi"], 30.0, marker=marker
        )
        assert wrapped[:2] == ["sh", "-c"]
        assert "setsid --wait" in wrapped[2]
        assert marker in wrapped
        # The target command is passed without shell interpolation as the
        # wrapper's argument tail.
        assert wrapped[-3:] == ["bash", "-c", "echo hi"]

    def test_wrap_formats_fractional_and_integral_durations(self):
        marker = f"{TIMEOUT_MARKER_PREFIX}test"
        assert "3" in wrap_command_with_timeout(["x"], 3.0, marker=marker)
        assert "2.5" in wrap_command_with_timeout(["x"], 2.5, marker=marker)
        assert "0" in wrap_command_with_timeout(["x"], 0, marker=marker)

    def test_outer_deadline_adds_grace(self):
        assert outer_read_deadline(10.0) == 10.0 + _timeout.EXEC_TIMEOUT_GRACE_SECONDS

    def test_only_exact_watchdog_marker_proves_timeout(self):
        marker = f"{TIMEOUT_MARKER_PREFIX}test"
        clean, timed_out = remove_timeout_marker(f"target output\n{marker}\n", marker)
        assert clean == "target output\n"
        assert timed_out is True

        # Exit codes and elapsed duration have no role in provenance. A target
        # may use either conventional timeout status itself.
        clean, timed_out = remove_timeout_marker("target output\n", marker)
        assert clean == "target output\n"
        assert timed_out is False


class _TimeoutMockResponse:
    """Async-context-manager stand-in for an aiodocker ``_query`` response."""

    def __init__(
        self,
        *,
        json_data: dict | None = None,
        read_bytes: bytes = b"",
        read_delay: float = 0.0,
    ) -> None:
        self._json = json_data
        self._read_bytes = read_bytes
        self._read_delay = read_delay
        self.read_started = asyncio.Event()
        self.closed = False

    async def __aenter__(self) -> "_TimeoutMockResponse":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.closed = True
        return False

    async def json(self) -> dict:
        return self._json or {}

    async def read(self) -> bytes:
        self.read_started.set()
        if self._read_delay:
            await asyncio.sleep(self._read_delay)
        return self._read_bytes

    def close(self) -> None:
        self.closed = True


class _TimeoutMockDocker:
    """Minimal aiodocker stand-in driving the exec_in_container control flow."""

    def __init__(
        self,
        *,
        exit_code: int = 0,
        read_bytes: bytes = b"",
        read_delay: float = 0.0,
    ) -> None:
        self._exit_code = exit_code
        self._read_bytes = read_bytes
        self._read_delay = read_delay
        self.closed = False
        self.exec_options: dict | None = None
        self.start_response: _TimeoutMockResponse | None = None

    def _query(self, path, method, headers=None, data=None):
        if "exec" in path and "start" in path:
            self.start_response = _TimeoutMockResponse(
                read_bytes=self._read_bytes, read_delay=self._read_delay
            )
            return self.start_response
        if "exec" in path and "json" in path:
            return _TimeoutMockResponse(json_data={"ExitCode": self._exit_code})
        # create exec
        self.exec_options = json.loads(data) if isinstance(data, str) else data
        return _TimeoutMockResponse(json_data={"Id": "exec-xyz"})

    async def close(self) -> None:
        self.closed = True


class TestExecInContainerTimeout:
    """exec_in_container timeout enforcement + reaping control flow (offline)."""

    @pytest.mark.asyncio
    async def test_wraps_command_with_session_watchdog(self):
        docker = _TimeoutMockDocker(exit_code=0, read_bytes=b"ok")
        await exec_in_container(
            "c", ["bash", "-c", "echo ok"], client=docker, timeout=30.0
        )
        assert docker.exec_options is not None
        cmd = docker.exec_options["Cmd"]
        assert cmd[:2] == ["sh", "-c"]
        assert "setsid --wait" in cmd[2]
        assert any(arg.startswith(TIMEOUT_MARKER_PREFIX) for arg in cmd)
        assert cmd[-3:] == ["bash", "-c", "echo ok"]

    @pytest.mark.asyncio
    async def test_normal_exit_code_and_stdout_preserved(self):
        docker = _TimeoutMockDocker(exit_code=5, read_bytes=b"some output")
        result = await exec_in_container("c", ["false"], client=docker, timeout=120.0)
        assert result.exit_code == 5
        assert result.stdout == "some output"

    @pytest.mark.asyncio
    async def test_stdout_stderr_and_exit_code_are_preserved(self):
        def frame(stream: int, payload: bytes) -> bytes:
            return bytes([stream, 0, 0, 0]) + len(payload).to_bytes(4, "big") + payload

        docker = _TimeoutMockDocker(
            exit_code=7,
            read_bytes=frame(1, b"standard output") + frame(2, b"standard error"),
        )
        result = await exec_in_container("c", ["false"], client=docker, timeout=120.0)
        assert result.exit_code == 7
        assert result.stdout == "standard output"
        assert result.stderr == "standard error"

    @pytest.mark.parametrize("exit_code", [124, 137])
    @pytest.mark.asyncio
    async def test_delayed_target_timeout_exit_codes_are_ordinary_results(
        self, exit_code
    ):
        # Deliberately run for longer than the old elapsed-time heuristic floor.
        # Without the watchdog's marker, both statuses belong to the target.
        docker = _TimeoutMockDocker(
            exit_code=exit_code, read_bytes=b"target result", read_delay=0.06
        )
        result = await exec_in_container(
            "c", ["sh", "-c", f"exit {exit_code}"], client=docker, timeout=0.05
        )
        assert result.exit_code == exit_code
        assert result.stdout == "target result"

    @pytest.mark.asyncio
    async def test_watchdog_marker_surfaces_timeout_and_is_not_returned(
        self, monkeypatch
    ):
        marker = f"{TIMEOUT_MARKER_PREFIX}offline-watchdog"
        monkeypatch.setattr(commands, "new_timeout_marker", lambda: marker)
        docker = _TimeoutMockDocker(
            exit_code=0, read_bytes=f"before\n{marker}\n".encode()
        )
        with pytest.raises(asyncio.TimeoutError):
            await exec_in_container("c", ["sleep", "600"], client=docker, timeout=0.05)

    @pytest.mark.asyncio
    async def test_stuck_read_closes_response_and_awaits_reader(self, monkeypatch):
        # When the watchdog should already have fired but the HTTP read remains
        # blocked, the transport is closed before the reader is cancelled and
        # awaited. No detached task or response survives.
        monkeypatch.setattr(_timeout, "EXEC_TIMEOUT_GRACE_SECONDS", 0.05)
        docker = _TimeoutMockDocker(exit_code=0, read_delay=10.0)
        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await exec_in_container("c", ["sleep", "600"], client=docker, timeout=0.05)
        elapsed = time.monotonic() - start
        assert elapsed < 2.0
        assert docker.start_response is not None
        assert docker.start_response.closed is True

    @pytest.mark.asyncio
    async def test_outer_cancellation_closes_response_and_awaits_reader(self):
        docker = _TimeoutMockDocker(exit_code=0, read_delay=10.0)
        task = asyncio.create_task(
            exec_in_container("c", ["sleep", "600"], client=docker, timeout=30.0)
        )
        while docker.start_response is None:
            await asyncio.sleep(0)
        await docker.start_response.read_started.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert docker.start_response.closed is True
