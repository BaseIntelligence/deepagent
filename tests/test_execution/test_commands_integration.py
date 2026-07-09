"""Real-Docker integration tests for in-container exec timeout + tree reaping.

Marked ``@pytest.mark.integration`` (deselected from the milestone gate). Proves
that a deliberately-hung in-container command run through
``DockerSandbox.run_command`` / ``exec_in_container``:

* returns control (raises) within ~timeout+grace instead of blocking forever, and
* leaves NO surviving in-container process -- the whole process tree is reaped --
  while the normal (non-timeout) exit-code/stdout contract is unchanged.

Run manually:
    .venv/bin/python -m pytest tests/test_execution/test_commands_integration.py \
        -q -p no:cacheprovider -m integration
"""

from __future__ import annotations

import asyncio
import subprocess
import time

import pytest

from swe_forge.execution.docker_client import DockerClient, DockerError
from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

_IMAGE = "python:3.12-slim"


def _docker_top(container_id: str) -> str:
    proc = subprocess.run(
        ["docker", "top", container_id],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hung_command_is_bounded_and_process_tree_reaped():
    client = DockerClient()
    config = SandboxConfig(
        name="swe-forge-exec-timeout-it",
        image=_IMAGE,
        command_timeout=3.0,
    )
    async with DockerSandbox(client, config) as sandbox:
        container_id = sandbox.container_id
        assert container_id is not None

        # A deliberately-hung command that spawns a CHILD TREE (bash -> two
        # subshell loops) each appending a heartbeat. If the timeout reaps only
        # the direct child, the loops survive and keep growing the file.
        hung = (
            "rm -f /tmp/hb; "
            "(while true; do echo a >> /tmp/hb; sleep 0.25; done) & "
            "(while true; do echo b >> /tmp/hb; sleep 0.25; done) & "
            "wait"
        )

        start = time.monotonic()
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await sandbox.run_command(hung, timeout=3.0)
        elapsed = time.monotonic() - start
        # Bounded by ~timeout + grace, NOT the indefinite pre-fix hang.
        assert elapsed < 25.0, f"hung command took {elapsed:.1f}s to return"

        # Process tree reaped: the heartbeat file must stop growing.
        first = await sandbox.run_command(
            "wc -l < /tmp/hb 2>/dev/null || echo 0", timeout=15.0
        )
        await asyncio.sleep(2.0)
        second = await sandbox.run_command(
            "wc -l < /tmp/hb 2>/dev/null || echo 0", timeout=15.0
        )
        assert first.stdout.strip() == second.stdout.strip(), (
            "heartbeat file kept growing after timeout -> process tree NOT reaped"
        )

        # Authoritative cross-check via the host: no surviving loop/sleep
        # process from the reaped exec (only the container's keep-alive remains).
        top = _docker_top(container_id)
        assert "sleep 0.25" not in top, f"surviving in-container process:\n{top}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_normal_command_contract_unchanged_under_wrapper():
    client = DockerClient()
    config = SandboxConfig(
        name="swe-forge-exec-timeout-it-normal",
        image=_IMAGE,
        command_timeout=10.0,
    )
    async with DockerSandbox(client, config) as sandbox:
        # Exit code + stdout pass through the coreutils-timeout wrapper verbatim.
        ok = await sandbox.run_command("echo hello-forge && exit 0", timeout=10.0)
        assert ok.exit_code == 0
        assert "hello-forge" in ok.stdout

        failed = await sandbox.run_command("echo boom; exit 7", timeout=10.0)
        assert failed.exit_code == 7
        assert "boom" in failed.stdout

        # A command that finishes comfortably under its deadline is NOT flagged
        # as a timeout.
        slept = await sandbox.run_command("sleep 1; echo done", timeout=8.0)
        assert slept.exit_code == 0
        assert "done" in slept.stdout


@pytest.mark.integration
@pytest.mark.asyncio
async def test_target_exit_124_and_137_are_not_timeout_provenance():
    """Both APIs preserve target-owned conventional timeout exit statuses."""
    client = DockerClient()
    config = SandboxConfig(
        name="swe-forge-exec-timeout-it-target-exits",
        image=_IMAGE,
        command_timeout=10.0,
    )
    async with DockerSandbox(client, config) as sandbox:
        container_id = sandbox.container_id
        assert container_id is not None

        for exit_code in (124, 137):
            command = [
                "sh",
                "-c",
                f"sleep 2; printf target-{exit_code}; exit {exit_code}",
            ]

            via_commands = await sandbox.run_command(command, timeout=3.0)
            assert via_commands.exit_code == exit_code
            assert via_commands.stdout == f"target-{exit_code}"

            via_client = await client.exec(container_id, command, timeout=3.0)
            assert via_client.exit_code == exit_code
            assert via_client.stdout == f"target-{exit_code}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_docker_client_reaps_hung_process_tree():
    """DockerClient.exec applies the same watchdog/reaping contract directly."""
    client = DockerClient()
    config = SandboxConfig(
        name="swe-forge-docker-client-timeout-it",
        image=_IMAGE,
        command_timeout=3.0,
    )
    async with DockerSandbox(client, config) as sandbox:
        container_id = sandbox.container_id
        assert container_id is not None

        hung = (
            "rm -f /tmp/direct-hb; "
            "(while true; do echo a >> /tmp/direct-hb; sleep 0.2; done) & "
            "(while true; do echo b >> /tmp/direct-hb; sleep 0.2; done) & "
            "wait"
        )
        start = time.monotonic()
        with pytest.raises(DockerError, match="timed out"):
            await client.exec(container_id, ["sh", "-c", hung], timeout=2.0)
        assert time.monotonic() - start < 20.0

        first = await client.exec(
            container_id,
            ["sh", "-c", "wc -l < /tmp/direct-hb 2>/dev/null || echo 0"],
            timeout=10.0,
        )
        await asyncio.sleep(1.0)
        second = await client.exec(
            container_id,
            ["sh", "-c", "wc -l < /tmp/direct-hb 2>/dev/null || echo 0"],
            timeout=10.0,
        )
        assert first.stdout.strip() == second.stdout.strip()
        assert "sleep 0.2" not in _docker_top(container_id)
