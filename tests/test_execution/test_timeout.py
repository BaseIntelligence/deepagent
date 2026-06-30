"""Unit tests for in-container exec timeout enforcement + process-tree reaping.

These cover the offline control flow of the timeout/reap fix without a Docker
daemon: the in-container command is wrapped under coreutils ``timeout`` so the
daemon reaps the process tree, a coreutils-``timeout`` exit (124/137) at/after
the deadline is surfaced as a clean timeout, and a read that never returns after
the reap is force-abandoned (reader cancelled) instead of blocking forever.

The live-Docker proof that the process *tree* is actually reaped lives in
``test_commands_integration.py`` (``@pytest.mark.integration``).
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from swe_forge.execution import _timeout
from swe_forge.execution._timeout import (
    TIMEOUT_EXIT_CODES,
    is_timeout_exit,
    outer_read_deadline,
    wrap_command_with_timeout,
)
from swe_forge.execution.commands import exec_in_container


class TestTimeoutHelpers:
    """Pure-function tests for the shared timeout helpers."""

    def test_wrap_prefixes_coreutils_timeout_kill(self):
        wrapped = wrap_command_with_timeout(["bash", "-c", "echo hi"], 30.0)
        assert wrapped[0] == "timeout"
        assert "--signal" in wrapped
        assert "KILL" in wrapped
        # The original command is preserved verbatim as the wrapper's tail.
        assert wrapped[-3:] == ["bash", "-c", "echo hi"]

    def test_wrap_formats_fractional_and_integral_durations(self):
        assert wrap_command_with_timeout(["x"], 3.0)[3] == "3"
        assert wrap_command_with_timeout(["x"], 2.5)[3] == "2.5"
        assert wrap_command_with_timeout(["x"], 0)[3] == "0"

    def test_outer_deadline_adds_grace(self):
        assert outer_read_deadline(10.0) == 10.0 + _timeout.EXEC_TIMEOUT_GRACE_SECONDS

    def test_timeout_exit_codes_are_124_and_137(self):
        assert TIMEOUT_EXIT_CODES == frozenset({124, 137})

    @pytest.mark.parametrize("code", [124, 137])
    def test_is_timeout_when_killed_after_full_deadline(self, code):
        assert is_timeout_exit(code, duration=3.0, timeout=3.0) is True

    def test_not_timeout_for_normal_exit_codes(self):
        assert is_timeout_exit(0, duration=3.0, timeout=3.0) is False
        assert is_timeout_exit(1, duration=3.0, timeout=3.0) is False

    def test_not_timeout_for_early_kill(self):
        # An early OOM / external kill (137) well before the deadline must be
        # preserved as a normal failure, not misread as a timeout.
        assert is_timeout_exit(137, duration=0.2, timeout=3.0) is False


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

    async def __aenter__(self) -> "_TimeoutMockResponse":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    async def json(self) -> dict:
        return self._json or {}

    async def read(self) -> bytes:
        if self._read_delay:
            await asyncio.sleep(self._read_delay)
        return self._read_bytes


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

    def _query(self, path, method, headers=None, data=None):
        if "exec" in path and "start" in path:
            return _TimeoutMockResponse(
                read_bytes=self._read_bytes, read_delay=self._read_delay
            )
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
    async def test_wraps_command_with_coreutils_timeout(self):
        docker = _TimeoutMockDocker(exit_code=0, read_bytes=b"ok")
        await exec_in_container(
            "c", ["bash", "-c", "echo ok"], client=docker, timeout=30.0
        )
        assert docker.exec_options is not None
        cmd = docker.exec_options["Cmd"]
        assert cmd[0] == "timeout"
        assert "KILL" in cmd
        assert cmd[-3:] == ["bash", "-c", "echo ok"]

    @pytest.mark.asyncio
    async def test_normal_exit_code_and_stdout_preserved(self):
        docker = _TimeoutMockDocker(exit_code=5, read_bytes=b"some output")
        result = await exec_in_container("c", ["false"], client=docker, timeout=120.0)
        assert result.exit_code == 5
        assert result.stdout == "some output"

    @pytest.mark.asyncio
    async def test_early_kill_137_not_misread_as_timeout(self):
        # Exit 137 with near-zero duration (OOM/external kill) is a normal
        # failure, not the injected timeout: must return, not raise.
        docker = _TimeoutMockDocker(exit_code=137, read_bytes=b"")
        result = await exec_in_container(
            "c", ["bash", "-c", ":"], client=docker, timeout=120.0
        )
        assert result.exit_code == 137

    @pytest.mark.asyncio
    async def test_reaped_exit_surfaces_timeout(self):
        # coreutils ``timeout`` reaped the tree: read returns with exit 137 after
        # ~the full deadline -> surfaced as a clean asyncio.TimeoutError.
        docker = _TimeoutMockDocker(exit_code=137, read_bytes=b"", read_delay=0.06)
        with pytest.raises(asyncio.TimeoutError):
            await exec_in_container("c", ["sleep", "600"], client=docker, timeout=0.05)

    @pytest.mark.asyncio
    async def test_stuck_read_is_abandoned_and_raises(self, monkeypatch):
        # The HTTP read never returns after the reap; the outer net must cancel
        # the reader and raise within ~timeout+grace, never blocking on the
        # 10s read.
        monkeypatch.setattr(_timeout, "EXEC_TIMEOUT_GRACE_SECONDS", 0.05)
        docker = _TimeoutMockDocker(exit_code=0, read_delay=10.0)
        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await exec_in_container("c", ["sleep", "600"], client=docker, timeout=0.05)
        elapsed = time.monotonic() - start
        assert elapsed < 2.0
