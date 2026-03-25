"""Unit tests for Container lifecycle management.

These tests mock the DockerClient to test container lifecycle
without requiring a running Docker daemon.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_forge.execution import (
    ContainerConfig,
    ContainerStatus,
    DockerClient,
    DockerError,
    ExecResult,
)
from swe_forge.execution.container import (
    ContainerManager,
    ContainerSpec,
    ManagedContainer,
    ManagedContainerStatus,
    VolumeMount,
)


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


class MockExec:
    """Mock exec object from aiodocker."""

    def __init__(self, exit_code: int = 0):
        self._exit_code = exit_code
        self.inspect = AsyncMock(return_value={"ExitCode": exit_code})

    def start(self, detach: bool = False):
        return self._stream_generator()

    async def _stream_generator(self):
        for chunk in [b"stdout output\n", b"more output\n"]:
            yield chunk


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
        self.images.pull = MagicMock(return_value=self._pull_generator())

        self.exec.return_value = MockExec()

    async def _pull_generator(self):
        yield {"status": "Pulling"}
        yield {"status": "Complete"}


def create_mock_client(container_id: str = "test-container-id") -> DockerClient:
    """Create a DockerClient with mocked Docker instance."""
    mock_docker = MockDocker(container_id)
    client = DockerClient.from_docker(mock_docker)
    return client


class TestVolumeMount:
    """Tests for VolumeMount."""

    def test_basic_mount(self):
        mount = VolumeMount(host_path="/host/path", container_path="/container/path")
        assert mount.host_path == "/host/path"
        assert mount.container_path == "/container/path"
        assert mount.read_only is False

    def test_read_only_mount(self):
        mount = VolumeMount(
            host_path="/host/path", container_path="/container/path", read_only=True
        )
        assert mount.read_only is True

    def test_to_docker_bind_read_write(self):
        mount = VolumeMount(host_path="/host/path", container_path="/container/path")
        bind = mount.to_docker_bind()
        assert bind == "/host/path:/container/path:rw"

    def test_to_docker_bind_read_only(self):
        mount = VolumeMount(
            host_path="/host/path", container_path="/container/path", read_only=True
        )
        bind = mount.to_docker_bind()
        assert bind == "/host/path:/container/path:ro"


class TestContainerSpec:
    """Tests for ContainerSpec."""

    def test_basic_spec(self):
        spec = ContainerSpec(name="test", image="python:3.11-slim")
        assert spec.name == "test"
        assert spec.image == "python:3.11-slim"
        assert spec.command is None
        assert spec.volumes == []
        assert spec.env == {}

    def test_spec_with_volumes(self):
        spec = ContainerSpec(
            name="test",
            image="python:3.11-slim",
            volumes=[
                VolumeMount("/host/path1", "/container/path1"),
                VolumeMount("/host/path2", "/container/path2", read_only=True),
            ],
        )
        assert len(spec.volumes) == 2
        assert spec.volumes[0].host_path == "/host/path1"
        assert spec.volumes[1].read_only is True

    def test_spec_with_env(self):
        spec = ContainerSpec(
            name="test",
            image="python:3.11-slim",
            env={"FOO": "bar", "BAZ": "qux"},
        )
        assert spec.env == {"FOO": "bar", "BAZ": "qux"}

    def test_to_container_config_minimal(self):
        spec = ContainerSpec(name="test", image="python:3.11-slim")
        config = spec.to_container_config()

        assert config.name == "test"
        assert config.image == "python:3.11-slim"
        assert config.cmd is None
        assert config.env == []
        assert config.volumes == []

    def test_to_container_config_full(self):
        spec = ContainerSpec(
            name="test",
            image="python:3.11-slim",
            command=["python", "-c", "print(1)"],
            volumes=[
                VolumeMount("/host/path", "/container/path"),
                VolumeMount("/host/readonly", "/container/readonly", read_only=True),
            ],
            env={"FOO": "bar", "BAZ": "qux"},
            working_dir="/workspace",
            user="1000:1000",
            network_mode="none",
            memory_mb=1024,
            cpu_limit=2.0,
            pids_limit=200,
            stop_timeout=30,
        )
        config = spec.to_container_config()

        assert config.name == "test"
        assert config.image == "python:3.11-slim"
        assert config.cmd == ["python", "-c", "print(1)"]
        assert set(config.env) == {"FOO=bar", "BAZ=qux"}
        assert config.volumes == [
            "/host/path:/container/path:rw",
            "/host/readonly:/container/readonly:ro",
        ]
        assert config.working_dir == "/workspace"
        assert config.user == "1000:1000"
        assert config.network_mode == "none"
        assert config.memory_mb == 1024
        assert config.cpu_limit == 2.0
        assert config.pids_limit == 200


class TestManagedContainerStatus:
    """Tests for ManagedContainerStatus."""

    def test_str_representation(self):
        assert str(ManagedContainerStatus.PENDING) == "pending"
        assert str(ManagedContainerStatus.RUNNING) == "running"
        assert str(ManagedContainerStatus.COMPLETED) == "completed"
        assert str(ManagedContainerStatus.FAILED) == "failed"
        assert str(ManagedContainerStatus.TIMEOUT) == "timeout"

    def test_equality(self):
        assert ManagedContainerStatus.RUNNING == ManagedContainerStatus.RUNNING
        assert ManagedContainerStatus.RUNNING != ManagedContainerStatus.COMPLETED


class TestManagedContainer:
    """Tests for ManagedContainer."""

    def test_basic_container(self):
        spec = ContainerSpec(name="test", image="python:3.11-slim")
        container = ManagedContainer(id="test-id", spec=spec)
        assert container.id == "test-id"
        assert container.status == ManagedContainerStatus.PENDING
        assert container.exit_code is None
        assert container.error_message is None

    def test_is_terminal(self):
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        container = ManagedContainer(
            id="test-id", spec=spec, status=ManagedContainerStatus.COMPLETED
        )
        assert container.is_terminal is True

        container = ManagedContainer(
            id="test-id", spec=spec, status=ManagedContainerStatus.FAILED
        )
        assert container.is_terminal is True

        container = ManagedContainer(
            id="test-id", spec=spec, status=ManagedContainerStatus.TIMEOUT
        )
        assert container.is_terminal is True

        container = ManagedContainer(
            id="test-id", spec=spec, status=ManagedContainerStatus.RUNNING
        )
        assert container.is_terminal is False

        container = ManagedContainer(
            id="test-id", spec=spec, status=ManagedContainerStatus.PENDING
        )
        assert container.is_terminal is False

    def test_is_running(self):
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        container = ManagedContainer(
            id="test-id", spec=spec, status=ManagedContainerStatus.RUNNING
        )
        assert container.is_running is True

        container = ManagedContainer(
            id="test-id", spec=spec, status=ManagedContainerStatus.PENDING
        )
        assert container.is_running is False


class TestContainerManager:
    """Tests for ContainerManager."""

    @pytest.mark.asyncio
    async def test_context_manager_creates_and_starts(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec) as manager:
            assert manager.container_id == "test-container-id"
            assert manager.container.status == ManagedContainerStatus.RUNNING

    @pytest.mark.asyncio
    async def test_context_manager_auto_start_disabled(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec, auto_start=False) as manager:
            assert manager.container_id == "test-container-id"
            assert manager.container.status == ManagedContainerStatus.PENDING

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_normal_exit(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec) as manager:
            pass

        client._docker._mock_container.stop.assert_called()
        client._docker._mock_container.delete.assert_called()

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        with pytest.raises(ValueError):
            async with ContainerManager(client, spec) as manager:
                raise ValueError("test error")

        client._docker._mock_container.stop.assert_called()
        client._docker._mock_container.delete.assert_called()

    @pytest.mark.asyncio
    async def test_context_manager_force_remove_on_stop_failure(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")
        client._docker._mock_container.stop = AsyncMock(
            side_effect=Exception("stop failed")
        )

        with pytest.raises(ValueError):
            async with ContainerManager(client, spec) as manager:
                raise ValueError("test error")

        client._docker._mock_container.delete.assert_called_once()
        call_kwargs = client._docker._mock_container.delete.call_args[1]
        assert call_kwargs.get("force", False) is True

    @pytest.mark.asyncio
    async def test_manual_start(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec, auto_start=False) as manager:
            assert manager.container.status == ManagedContainerStatus.PENDING
            await manager.start()
            assert manager.container.status == ManagedContainerStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_twice_fails(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec) as manager:
            with pytest.raises(DockerError) as exc_info:
                await manager.start()
            assert "running state" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_exec_in_running_container(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec) as manager:
            result = await manager.exec(["ls", "-la"])
            assert result.exit_code == 0
            assert "stdout output" in result.stdout

    @pytest.mark.asyncio
    async def test_exec_in_non_running_container_fails(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec, auto_start=False) as manager:
            with pytest.raises(DockerError) as exc_info:
                await manager.exec(["ls", "-la"])
            assert "cannot exec" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_get_logs(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec) as manager:
            logs = await manager.get_logs()
            assert "log line" in logs

    @pytest.mark.asyncio
    async def test_wait_for_container(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec) as manager:
            exit_code = await manager.wait()
            assert exit_code == 0

    @pytest.mark.asyncio
    async def test_wait_updates_status_on_success(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec) as manager:
            manager._container.status = ManagedContainerStatus.RUNNING
            exit_code = await manager.wait()
            assert exit_code == 0
            assert manager.container.status == ManagedContainerStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_wait_updates_status_on_failure(self):
        client = create_mock_client()
        client._docker._mock_container.wait = AsyncMock(return_value={"StatusCode": 1})
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec) as manager:
            manager._container.status = ManagedContainerStatus.RUNNING
            exit_code = await manager.wait()
            assert exit_code == 1
            assert manager.container.status == ManagedContainerStatus.FAILED
            assert "code 1" in manager.container.error_message

    @pytest.mark.asyncio
    async def test_sync_status(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec) as manager:
            status = await manager.sync_status()
            assert status == ManagedContainerStatus.RUNNING

    @pytest.mark.asyncio
    async def test_stop_container(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec) as manager:
            assert manager.container.status == ManagedContainerStatus.RUNNING
            await manager.stop()
            assert manager.container.status == ManagedContainerStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_stop_non_running_container_noop(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec, auto_start=False) as manager:
            await manager.stop()
            client._docker._mock_container.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_properties(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        manager = ContainerManager(client, spec)
        assert manager.container is None
        assert manager.container_id is None

    @pytest.mark.asyncio
    async def test_properties_after_enter(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager(client, spec) as manager:
            assert manager.container is not None
            assert manager.container_id == "test-container-id"

    @pytest.mark.asyncio
    async def test_container_not_created_errors(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        manager = ContainerManager(client, spec)

        with pytest.raises(DockerError) as exc_info:
            await manager.start()
        assert "not created" in str(exc_info.value).lower()

        with pytest.raises(DockerError) as exc_info:
            await manager.exec(["ls"])
        assert "not created" in str(exc_info.value).lower()

        with pytest.raises(DockerError) as exc_info:
            await manager.get_logs()
        assert "not created" in str(exc_info.value).lower()

        with pytest.raises(DockerError) as exc_info:
            await manager.wait()
        assert "not created" in str(exc_info.value).lower()

        with pytest.raises(DockerError) as exc_info:
            await manager.sync_status()
        assert "not created" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_stop_timeout_from_spec(self):
        client = create_mock_client()
        spec = ContainerSpec(name="test", image="python:3.11-slim", stop_timeout=30)

        async with ContainerManager(client, spec) as manager:
            await manager.stop()

            client._docker._mock_container.stop.assert_called_once()
            call_args = client._docker._mock_container.stop.call_args
            assert call_args[1]["t"] == 30

    @pytest.mark.asyncio
    async def test_volume_mounts_in_config(self):
        spec = ContainerSpec(
            name="test",
            image="python:3.11-slim",
            volumes=[
                VolumeMount("/host/workspace", "/workspace"),
                VolumeMount("/host/data", "/data", read_only=True),
            ],
        )

        config = spec.to_container_config()
        assert "/host/workspace:/workspace:rw" in config.volumes
        assert "/host/data:/data:ro" in config.volumes

    @pytest.mark.asyncio
    async def test_env_vars_in_config(self):
        spec = ContainerSpec(
            name="test",
            image="python:3.11-slim",
            env={"DEBUG": "true", "PATH": "/custom/bin"},
        )

        config = spec.to_container_config()
        assert "DEBUG=true" in config.env
        assert "PATH=/custom/bin" in config.env


class TestContainerManagerFromExisting:
    """Tests for ContainerManager.from_existing."""

    @pytest.mark.asyncio
    async def test_from_existing_container(self):
        client = create_mock_client(container_id="existing-id")
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager.from_existing(
            client, "existing-id", spec
        ) as manager:
            assert manager.container_id == "existing-id"
            assert manager.container.status == ManagedContainerStatus.PENDING

    @pytest.mark.asyncio
    async def test_from_existing_with_auto_remove(self):
        client = create_mock_client(container_id="existing-id")
        spec = ContainerSpec(name="test", image="python:3.11-slim")

        async with ContainerManager.from_existing(
            client, "existing-id", spec, auto_remove=True
        ) as manager:
            pass

        client._docker._mock_container.delete.assert_called()
