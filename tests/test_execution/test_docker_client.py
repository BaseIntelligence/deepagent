"""Unit tests for DockerClient.

These tests mock the aiodocker library to test the client without requiring
a running Docker daemon.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_forge.execution import _timeout, docker_client
from swe_forge.execution import (
    ContainerConfig,
    ContainerStatus,
    DockerClient,
    DockerError,
    ExecResult,
)
from swe_forge.execution._timeout import TIMEOUT_MARKER_PREFIX


class MockContainer:
    """Mock container object from aiodocker."""

    def __init__(self, container_id: str = "test-container-id"):
        self.id = container_id
        self.start = AsyncMock()
        self.stop = AsyncMock()
        self.delete = AsyncMock()
        self.show = AsyncMock(return_value={"State": {"Status": "running"}})
        self.log = AsyncMock(return_value=["log line 1", "log line 2"])
        self.wait = AsyncMock(return_value={"StatusCode": 0})
        self.exec = AsyncMock(return_value="exec-id")


class MockStream:
    """Closable ``aiodocker.Stream`` stand-in."""

    def __init__(
        self,
        chunks: list[tuple[int, bytes]] | None = None,
        *,
        read_delay: float = 0.0,
    ) -> None:
        self._chunks = chunks or [(1, b"stdout output\n"), (1, b"more output\n")]
        self._read_delay = read_delay
        self._index = 0
        self.read_started = asyncio.Event()
        self.closed = False

    async def read_out(self):
        self.read_started.set()
        if self._read_delay:
            await asyncio.sleep(self._read_delay)
        if self._index >= len(self._chunks):
            return None
        stream, data = self._chunks[self._index]
        self._index += 1
        return SimpleNamespace(stream=stream, data=data)

    async def close(self) -> None:
        self.closed = True


class MockExec:
    """Mock exec object from aiodocker."""

    def __init__(
        self,
        exit_code: int = 0,
        *,
        chunks: list[tuple[int, bytes]] | None = None,
        read_delay: float = 0.0,
    ):
        self._exit_code = exit_code
        self.stream = MockStream(chunks, read_delay=read_delay)
        self.inspect = AsyncMock(return_value={"ExitCode": exit_code})

    def start(self, detach: bool = False):
        return self.stream


class MockDocker:
    """Mock Docker client from aiodocker."""

    def __init__(self, container_id: str = "test-container-id"):
        self._container_id = container_id
        self.containers = MagicMock()
        self.images = MagicMock()
        self.exec = MagicMock()
        self.close = AsyncMock()

        self._mock_container = MockContainer(container_id)

        self.containers.create_or_replace = AsyncMock(return_value=self._mock_container)
        self.containers.container = MagicMock(return_value=self._mock_container)
        self.containers.list = AsyncMock(
            return_value=[
                {"Id": container_id, "Names": ["/test-container"], "State": "running"}
            ]
        )

        self.images.list = AsyncMock(return_value=[])
        self.images.inspect = AsyncMock(return_value={"Id": "sha256:image-id"})
        self.images.pull = AsyncMock(
            return_value=[{"status": "Pulling"}, {"status": "Complete"}]
        )

        self.exec.return_value = MockExec()


class TestContainerConfig:
    """Tests for ContainerConfig."""

    def test_basic_config(self):
        config = ContainerConfig(name="test", image="python:3.11-slim")
        assert config.name == "test"
        assert config.image == "python:3.11-slim"
        assert config.cmd is None
        assert config.env == []

    def test_to_docker_config_minimal(self):
        config = ContainerConfig(name="test", image="python:3.11-slim")
        docker_config = config.to_docker_config()

        assert docker_config["Image"] == "python:3.11-slim"
        assert docker_config["Tty"] is True
        assert "Memory" in docker_config["HostConfig"]
        assert "NanoCpus" in docker_config["HostConfig"]

    def test_to_docker_config_full(self):
        config = ContainerConfig(
            name="test",
            image="python:3.11-slim",
            cmd=["python", "-c", "print(1)"],
            env=["FOO=bar", "BAZ=qux"],
            working_dir="/workspace",
            volumes=["/host:/container"],
            user="1000:1000",
            network_mode="none",
            memory_mb=1024,
            cpu_limit=2.0,
            pids_limit=200,
        )
        docker_config = config.to_docker_config()

        assert docker_config["Cmd"] == ["python", "-c", "print(1)"]
        assert docker_config["Env"] == ["FOO=bar", "BAZ=qux"]
        assert docker_config["WorkingDir"] == "/workspace"
        assert docker_config["User"] == "1000:1000"
        assert docker_config["HostConfig"]["Binds"] == ["/host:/container"]
        assert docker_config["HostConfig"]["NetworkMode"] == "none"
        assert docker_config["HostConfig"]["Memory"] == 1024 * 1024 * 1024
        assert docker_config["HostConfig"]["NanoCpus"] == 2_000_000_000
        assert docker_config["HostConfig"]["PidsLimit"] == 200


class TestExecResult:
    """Tests for ExecResult."""

    def test_success(self):
        result = ExecResult(exit_code=0, stdout="output", stderr="")
        assert result.success is True

    def test_failure(self):
        result = ExecResult(exit_code=1, stdout="", stderr="error")
        assert result.success is False

    def test_nonzero_exit(self):
        result = ExecResult(exit_code=127, stdout="", stderr="command not found")
        assert result.exit_code == 127
        assert result.success is False


class TestDockerClient:
    """Tests for DockerClient."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                assert client._docker is not None

            mock_docker.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker_cls.return_value = mock_docker

            try:
                async with DockerClient():
                    raise ValueError("test error")
            except ValueError:
                pass

            mock_docker.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_from_docker(self):
        mock_docker = MockDocker()
        client = DockerClient.from_docker(mock_docker)

        assert client._docker is mock_docker
        assert client._own_connection is False

        async def test_context():
            async with client as c:
                assert c._docker is mock_docker

        await test_context()
        mock_docker.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_ping_success(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                result = await client.ping()
                assert result is True

    @pytest.mark.asyncio
    async def test_ping_failure(self):
        from aiodocker.exceptions import DockerError as AioDockerError

        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker.images.list = AsyncMock(
                side_effect=AioDockerError(500, "daemon error")
            )
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                result = await client.ping()
                assert result is False

    @pytest.mark.asyncio
    async def test_create_container(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker(container_id="created-id")
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                config = ContainerConfig(name="test", image="python:3.11-slim")
                container_id = await client.create_container(config)
                assert container_id == "created-id"

    @pytest.mark.asyncio
    async def test_start_container(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                await client.start_container("test-id")
                mock_docker._mock_container.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_container(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                await client.stop_container("test-id", timeout=5)
                mock_docker._mock_container.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_container(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                await client.remove_container("test-id", force=True)
                mock_docker._mock_container.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_containers(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                containers = await client.list_containers(all=True)
                assert len(containers) == 1
                assert containers[0]["Id"] == "test-container-id"

    @pytest.mark.asyncio
    async def test_get_container_status(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                status = await client.get_container_status("test-id")
                assert status == ContainerStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_container_status_exited(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker._mock_container.show = AsyncMock(
                return_value={"State": {"Status": "exited"}}
            )
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                status = await client.get_container_status("test-id")
                assert status == ContainerStatus.EXITED

    @pytest.mark.asyncio
    async def test_get_container_status_not_found(self):
        from aiodocker.exceptions import DockerError as AioDockerError

        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker._mock_container.show = AsyncMock(
                side_effect=AioDockerError(404, "No such container: test-id")
            )
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                with pytest.raises(DockerError) as exc_info:
                    await client.get_container_status("test-id")
                assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_exec(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_exec = MockExec(
                exit_code=0,
                chunks=[(1, b"stdout output\n"), (2, b"stderr output\n")],
            )
            mock_docker.exec.return_value = mock_exec
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                result = await client.exec("test-id", ["ls", "-la"])
                assert result.exit_code == 0
                assert result.stdout == "stdout output\n"
                assert result.stderr == "stderr output\n"
                assert mock_exec.stream.closed is True

    @pytest.mark.asyncio
    async def test_exec_without_timeout_preserves_blank_output_lines(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker.exec.return_value = MockExec(
                exit_code=0, chunks=[(1, b"first\n\nsecond\n"), (2, b"\nwarning\n")]
            )
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                result = await client.exec("test-id", ["echo", "output"])

            assert result.stdout == "first\n\nsecond\n"
            assert result.stderr == "\nwarning\n"

    @pytest.mark.asyncio
    async def test_exec_with_stuck_read_closes_stream_and_raises(self, monkeypatch):
        # Shrink the post-deadline grace so the stuck-read safety net fires fast.
        monkeypatch.setattr(_timeout, "EXEC_TIMEOUT_GRACE_SECONDS", 0.05)

        class SlowMockExec(MockExec):
            def __init__(self):
                super().__init__(exit_code=0, read_delay=10.0)

        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_exec = SlowMockExec()
            mock_docker.exec.return_value = mock_exec
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                with pytest.raises(DockerError) as exc_info:
                    await client.exec("test-id", ["sleep", "10"], timeout=0.1)
                assert "timed out" in str(exc_info.value).lower()
                assert mock_exec.stream.closed is True

    @pytest.mark.asyncio
    async def test_exec_wraps_cmd_with_session_watchdog(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_exec = MockExec(exit_code=0)
            mock_docker.exec.return_value = mock_exec
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                await client.exec("test-id", ["ls", "-la"], timeout=30.0)

            sent = mock_docker._mock_container.exec.await_args
            wrapped = sent.kwargs["cmd"]
            assert wrapped[:2] == ["sh", "-c"]
            assert "setsid --wait" in wrapped[2]
            assert any(arg.startswith(TIMEOUT_MARKER_PREFIX) for arg in wrapped)
            assert wrapped[-2:] == ["ls", "-la"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exit_code", [124, 137])
    async def test_delayed_target_timeout_exit_codes_are_ordinary_results(
        self, exit_code
    ):
        class DelayedTargetExec(MockExec):
            def __init__(self):
                super().__init__(
                    exit_code=exit_code,
                    chunks=[(1, b"target result")],
                    read_delay=0.06,
                )

        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker.exec.return_value = DelayedTargetExec()
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                result = await client.exec(
                    "test-id", ["sh", "-c", f"exit {exit_code}"], timeout=0.05
                )
                assert result.exit_code == exit_code
                assert result.stdout == "target result"

    @pytest.mark.asyncio
    async def test_watchdog_marker_surfaces_timeout_and_strips_marker(
        self, monkeypatch
    ):
        marker = f"{TIMEOUT_MARKER_PREFIX}docker-client-watchdog"
        monkeypatch.setattr(docker_client, "new_timeout_marker", lambda: marker)
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker.exec.return_value = MockExec(
                exit_code=0, chunks=[(1, f"before\n{marker}\n".encode())]
            )
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                with pytest.raises(DockerError, match="timed out"):
                    await client.exec("test-id", ["sleep", "600"], timeout=0.05)
            assert mock_docker.exec.return_value.stream.closed is True

    @pytest.mark.asyncio
    async def test_outer_cancellation_closes_stream_and_preserves_cancellation(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_exec = MockExec(exit_code=0, read_delay=10.0)
            mock_docker.exec.return_value = mock_exec
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                task = asyncio.create_task(
                    client.exec("test-id", ["sleep", "600"], timeout=30.0)
                )
                await mock_exec.stream.read_started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            assert mock_exec.stream.closed is True

    @pytest.mark.asyncio
    async def test_get_logs(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                logs = await client.get_logs("test-id")
                assert "log line" in logs

    @pytest.mark.asyncio
    async def test_wait_container(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                exit_code = await client.wait_container("test-id")
                assert exit_code == 0

    @pytest.mark.asyncio
    async def test_wait_container_timeout(self):
        import asyncio

        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()

            async def slow_wait(*args, **kwargs):
                await asyncio.sleep(10)
                return {"StatusCode": 0}

            mock_docker._mock_container.wait = slow_wait
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                with pytest.raises(DockerError) as exc_info:
                    await client.wait_container("test-id", timeout=0.1)
                assert "timed out" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_image_exists_true(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                exists = await client.image_exists("python:3.11-slim")
                assert exists is True

    @pytest.mark.asyncio
    async def test_image_exists_false(self):
        from aiodocker.exceptions import DockerError as AioDockerError

        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker.images.inspect = AsyncMock(
                side_effect=AioDockerError(404, "No such image")
            )
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                exists = await client.image_exists("nonexistent:image")
                assert exists is False

    @pytest.mark.asyncio
    async def test_inspect_image_returns_daemon_metadata(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                metadata = await client.inspect_image("python:3.11-slim")

            assert metadata == {"Id": "sha256:image-id"}
            mock_docker.images.inspect.assert_awaited_once_with("python:3.11-slim")

    @pytest.mark.asyncio
    async def test_ensure_image_already_exists(self):
        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                await client.ensure_image("python:3.11-slim")
                mock_docker.images.pull.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_image_pulls(self):
        from aiodocker.exceptions import DockerError as AioDockerError

        with patch("swe_forge.execution.docker_client.Docker") as mock_docker_cls:
            mock_docker = MockDocker()
            call_count = 0

            async def inspect_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise AioDockerError(404, "No such image")
                return {"Id": "sha256:new-image-id"}

            mock_docker.images.inspect = inspect_side_effect
            mock_docker_cls.return_value = mock_docker

            async with DockerClient() as client:
                await client.ensure_image("nonexistent:image")

    @pytest.mark.asyncio
    async def test_client_not_initialized_error(self):
        client = DockerClient()
        with pytest.raises(DockerError) as exc_info:
            await client.ping()
        assert "not initialized" in str(exc_info.value).lower()


class TestDockerError:
    """Tests for DockerError."""

    def test_basic_error(self):
        error = DockerError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.container_id is None
        assert error.exit_code is None

    def test_error_with_container_info(self):
        error = DockerError(
            "Container failed",
            container_id="abc123",
            exit_code=137,
        )
        assert error.container_id == "abc123"
        assert error.exit_code == 137

    def test_error_inheritance(self):
        error = DockerError("test")
        assert isinstance(error, Exception)
