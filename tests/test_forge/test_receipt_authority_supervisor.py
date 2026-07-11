"""Bounded cleanup tests for the production receipt supervisor."""

from __future__ import annotations

import signal
import subprocess
import sys

import pytest

from swe_forge.forge import receipt_authority_supervisor as supervisor


class _FakeProcess:
    def __init__(
        self,
        *,
        running: bool = True,
        wait_error: BaseException | None = None,
        timeout_once: bool = False,
    ) -> None:
        self.running = running
        self.wait_error = wait_error
        self.timeout_once = timeout_once
        self.calls: list[tuple[str, object]] = []

    def poll(self) -> int | None:
        self.calls.append(("poll", None))
        return None if self.running else 0

    def terminate(self) -> None:
        self.calls.append(("terminate", None))
        self.running = False

    def kill(self) -> None:
        self.calls.append(("kill", None))
        self.running = False

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append(("wait", timeout))
        if self.wait_error is not None:
            raise self.wait_error
        if self.timeout_once:
            self.timeout_once = False
            self.running = True
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        self.running = False
        return 0


def test_completed_child_is_reaped_without_signal() -> None:
    process = _FakeProcess(running=False)

    supervisor.terminate_kill_reap(process, timeout=0.25)  # type: ignore[arg-type]

    assert process.calls == [("poll", None), ("wait", None)]


def test_running_child_terminates_and_reaps_with_bounded_wait() -> None:
    process = _FakeProcess()

    supervisor.terminate_kill_reap(process, timeout=0.25)  # type: ignore[arg-type]

    assert process.calls == [
        ("poll", None),
        ("terminate", None),
        ("wait", 0.25),
    ]
    assert process.running is False


def test_sigterm_ignoring_child_is_killed_and_reaped() -> None:
    process = _FakeProcess(timeout_once=True)

    supervisor.terminate_kill_reap(process, timeout=0.25)  # type: ignore[arg-type]

    assert process.calls == [
        ("poll", None),
        ("terminate", None),
        ("wait", 0.25),
        ("kill", None),
        ("wait", None),
    ]
    assert process.running is False


def test_wait_exception_is_preserved_and_child_is_not_claimed_reaped() -> None:
    error = RuntimeError("wait failed")
    process = _FakeProcess(wait_error=error)

    with pytest.raises(RuntimeError, match="wait failed") as raised:
        supervisor.terminate_kill_reap(process, timeout=0.25)  # type: ignore[arg-type]

    assert raised.value is error
    assert process.calls == [
        ("poll", None),
        ("terminate", None),
        ("wait", 0.25),
    ]


def test_forge_cleanup_wait_exception_does_not_skip_authority_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forge = _FakeProcess(wait_error=RuntimeError("forge cleanup failed"))
    authority = _FakeProcess()

    with pytest.raises(RuntimeError, match="forge cleanup failed"):
        supervisor._cleanup_supervisor_children(  # noqa: SLF001
            forge,
            authority,
            timeout=0.25,  # type: ignore[arg-type]
        )

    assert forge.calls == [
        ("poll", None),
        ("terminate", None),
        ("wait", 0.25),
    ]
    assert authority.calls == [
        ("poll", None),
        ("terminate", None),
        ("wait", 0.25),
    ]


def test_authority_ignoring_sigterm_is_killed_and_reaped() -> None:
    forge = _FakeProcess(running=False)
    authority = _FakeProcess(timeout_once=True)

    supervisor._cleanup_supervisor_children(  # noqa: SLF001
        forge,
        authority,
        timeout=0.25,  # type: ignore[arg-type]
    )

    assert authority.calls == [
        ("poll", None),
        ("terminate", None),
        ("wait", 0.25),
        ("kill", None),
        ("wait", None),
    ]


def test_both_sigterm_ignoring_children_are_killed_and_reaped() -> None:
    forge = _FakeProcess(timeout_once=True)
    authority = _FakeProcess(timeout_once=True)

    supervisor._cleanup_supervisor_children(  # noqa: SLF001
        forge,
        authority,
        timeout=0.25,  # type: ignore[arg-type]
    )

    assert forge.calls[-2:] == [("kill", None), ("wait", None)]
    assert authority.calls[-2:] == [("kill", None), ("wait", None)]


@pytest.mark.integration
def test_real_sigterm_trapping_forge_is_killed_and_reaped() -> None:
    forge = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal, time; "
                "print('ready', flush=True); "
                "signal.signal(signal.SIGTERM, lambda *_: None); "
                "time.sleep(60)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    authority = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert forge.stdout is not None
        assert forge.stdout.readline() == b"ready\n"
        supervisor._cleanup_supervisor_children(  # noqa: SLF001
            forge, authority, timeout=0.05
        )
        assert forge.poll() is not None
        assert authority.poll() is not None
        assert forge.returncode == -signal.SIGKILL
        assert authority.returncode == -signal.SIGTERM
    finally:
        for process in (forge, authority):
            if process.poll() is None:
                process.kill()
            process.wait()
