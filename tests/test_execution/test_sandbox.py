"""Unit tests for DockerSandbox.

These tests mock the DockerClient and ContainerManager to test
the sandbox functionality without requiring a running Docker daemon.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_forge.execution import DockerClient, DockerError
from swe_forge.execution.sandbox import (
    DockerSandbox,
    SandboxConfig,
    SandboxState,
)
from swe_forge.execution.container import (
    ContainerSpec,
    ManagedContainer,
    ManagedContainerStatus,
)
from swe_forge.execution.commands import ExecResult


class MockContainer:
    """Mock container object from aiodocker."""

    def __init__(self, container_id: str = "test-sandbox-id"):
        self.id = container_id
        self.start = AsyncMock()
        self.stop = AsyncMock()
        self.delete = AsyncMock()
        self.show = AsyncMock(return_value={"State": {"Status": "running"}})
        self.log = AsyncMock(return_value=["log line 1"])
        self.wait = AsyncMock(return_value={"StatusCode": 0})
        self.exec = AsyncMock(return_value="exec-id")


class MockDocker:
    """Mock Docker client from aiodocker."""

    def __init__(self, container_id: str = "test-sandbox-id"):
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
                {"Id": container_id, "Names": ["/test-sandbox"], "State": "running"}
            ]
        )

        self.images.list = AsyncMock(return_value=[])
        self.images.inspect = AsyncMock(return_value={"Id": "sha256:image-id"})
        self.images.pull = MagicMock(return_value=self._pull_generator())

        self.exec.return_value = MockExec()

    async def _pull_generator(self):
        yield {"status": "Pulling"}
        yield {"status": "Complete"}


class MockExec:
    """Mock exec object from aiodocker."""

    def __init__(self, exit_code: int = 0, stdout: str = "output\n"):
        self._exit_code = exit_code
        self._stdout = stdout
        self.inspect = AsyncMock(return_value={"ExitCode": exit_code})

    def start(self, detach: bool = False):
        return self._stream_generator()

    async def _stream_generator(self):
        yield self._stdout.encode()


def create_mock_client(container_id: str = "test-sandbox-id") -> DockerClient:
    """Create a DockerClient with mocked Docker instance."""
    mock_docker = MockDocker(container_id)
    client = DockerClient.from_docker(mock_docker)
    return client


def create_exec_result(
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration: float = 0.1,
) -> ExecResult:
    """Create an ExecResult for mocking."""
    return ExecResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration=duration,
    )


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_default_config(self):
        config = SandboxConfig()
        assert config.name == "swe-sandbox"
        assert config.image == "python:3.11-slim"
        assert config.workspace_dir == "/repo"
        assert config.memory_mb == 2048
        assert config.clone_timeout == 600.0
        assert config.command_timeout == 120.0

    def test_custom_config(self):
        config = SandboxConfig(
            name="custom-sandbox",
            image="node:20-slim",
            workspace_dir="/workspace",
            memory_mb=4096,
            clone_timeout=300.0,
        )
        assert config.name == "custom-sandbox"
        assert config.image == "node:20-slim"
        assert config.workspace_dir == "/workspace"
        assert config.memory_mb == 4096
        assert config.clone_timeout == 300.0


class TestSandboxState:
    """Tests for SandboxState."""

    def test_default_state(self):
        state = SandboxState()
        assert state.container_id is None
        assert state.repo_url is None
        assert state.commit is None
        assert state.workspace_ready is False

    def test_state_with_values(self):
        state = SandboxState(
            container_id="abc123",
            repo_url="https://github.com/owner/repo",
            commit="def456",
            workspace_ready=True,
        )
        assert state.container_id == "abc123"
        assert state.repo_url == "https://github.com/owner/repo"
        assert state.commit == "def456"
        assert state.workspace_ready is True


class TestDockerSandboxInit:
    """Tests for DockerSandbox initialization."""

    def test_init_with_defaults(self):
        client = create_mock_client()
        sandbox = DockerSandbox(client)

        assert sandbox._client == client
        assert sandbox._config.name == "swe-sandbox"
        assert sandbox._config.image == "python:3.11-slim"
        assert sandbox.container_id is None

    def test_init_with_config(self):
        client = create_mock_client()
        config = SandboxConfig(name="test", image="node:20-slim")
        sandbox = DockerSandbox(client, config)

        assert sandbox._config.name == "test"
        assert sandbox._config.image == "node:20-slim"

    def test_init_with_image_override(self):
        client = create_mock_client()
        sandbox = DockerSandbox(client, image="golang:1.21-slim")

        assert sandbox._config.image == "golang:1.21-slim"

    def test_init_with_name_override(self):
        client = create_mock_client()
        sandbox = DockerSandbox(client, name="my-sandbox")

        assert sandbox._config.name == "my-sandbox"

    def test_unique_container_names(self):
        client = create_mock_client()
        sandbox1 = DockerSandbox(client)
        sandbox2 = DockerSandbox(client)

        assert sandbox1.container_name != sandbox2.container_name
        assert sandbox1.container_name.startswith("swe-sandbox-")
        assert sandbox2.container_name.startswith("swe-sandbox-")


class TestDockerSandboxContextManager:
    """Tests for DockerSandbox async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_creates_container(self):
        client = create_mock_client()
        async with DockerSandbox(client) as sandbox:
            assert sandbox.container_id == "test-sandbox-id"

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_normal_exit(self):
        client = create_mock_client()
        async with DockerSandbox(client) as sandbox:
            pass

        client._docker._mock_container.stop.assert_called()
        client._docker._mock_container.delete.assert_called()

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self):
        client = create_mock_client()

        with pytest.raises(ValueError):
            async with DockerSandbox(client):
                raise ValueError("test error")

        client._docker._mock_container.stop.assert_called()
        client._docker._mock_container.delete.assert_called()

    @pytest.mark.asyncio
    async def test_properties_after_enter(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            assert sandbox.container_id is not None
            assert sandbox.container_name.startswith("swe-sandbox-")
            assert sandbox.workspace_dir == "/repo"
            assert sandbox.workspace_ready is False


class TestDockerSandboxUrlNormalization:
    """Tests for URL normalization."""

    def test_https_url_unchanged(self):
        client = create_mock_client()
        sandbox = DockerSandbox(client)

        url = sandbox._normalize_repo_url("https://github.com/owner/repo")
        assert url == "https://github.com/owner/repo"

    def test_ssh_url_converted(self):
        client = create_mock_client()
        sandbox = DockerSandbox(client)

        url = sandbox._normalize_repo_url("git@github.com:owner/repo")
        assert url == "https://github.com/owner/repo"

    def test_short_format_expanded(self):
        client = create_mock_client()
        sandbox = DockerSandbox(client)

        url = sandbox._normalize_repo_url("owner/repo")
        assert url == "https://github.com/owner/repo"

    def test_url_with_whitespace_stripped(self):
        client = create_mock_client()
        sandbox = DockerSandbox(client)

        url = sandbox._normalize_repo_url("  https://github.com/owner/repo  ")
        assert url == "https://github.com/owner/repo"


class TestDockerSandboxCommands:
    """Tests for command execution."""

    @pytest.mark.asyncio
    async def test_run_command_success(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(
                    exit_code=0, stdout="success output\n"
                )
                result = await sandbox.run_command("echo hello")

                assert result.exit_code == 0
                assert "success output" in result.stdout

    @pytest.mark.asyncio
    async def test_run_command_with_list_args(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(exit_code=0)
                result = await sandbox.run_command(["python", "-c", "print(1)"])
                assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_run_command_with_cwd(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(exit_code=0)
                result = await sandbox.run_command("ls", cwd="/custom/path")
                assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_run_command_with_env(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(exit_code=0)
                result = await sandbox.run_command("echo $FOO", env={"FOO": "bar"})
                assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_run_command_before_setup_fails(self):
        client = create_mock_client()
        sandbox = DockerSandbox(client)

        with pytest.raises(DockerError) as exc_info:
            await sandbox.run_command("ls")
        assert "not created" in str(exc_info.value).lower()


class TestDockerSandboxDependencies:
    """Tests for dependency installation."""

    @pytest.mark.asyncio
    async def test_install_pip_packages(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(exit_code=0)
                result = await sandbox.install_dependencies(["pytest", "black"])
                assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_install_from_requirements(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(exit_code=0)
                result = await sandbox.install_dependencies(
                    requirements_file="requirements.txt"
                )
                assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_install_apt_packages(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(exit_code=0)
                result = await sandbox.install_dependencies(
                    ["curl", "jq"], use_pip=False
                )
                assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_install_requires_packages_or_file(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with pytest.raises(ValueError) as exc_info:
                await sandbox.install_dependencies()
            assert "packages or requirements_file" in str(exc_info.value)


class TestDockerSandboxFileOperations:
    """Tests for file operations."""

    @pytest.mark.asyncio
    async def test_write_file_success(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(exit_code=0)
                await sandbox.write_file("test.txt", "hello world")

    @pytest.mark.asyncio
    async def test_write_file_in_subdirectory(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(exit_code=0)
                await sandbox.write_file("subdir/nested/file.txt", "nested content")

    @pytest.mark.asyncio
    async def test_write_file_rejects_path_traversal(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with pytest.raises(DockerError) as exc_info:
                await sandbox.write_file("../../../etc/passwd", "malicious")
            assert "invalid path" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_write_file_rejects_absolute_path(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with pytest.raises(DockerError) as exc_info:
                await sandbox.write_file("/etc/passwd", "malicious")
            assert "invalid path" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_read_file_success(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(
                    exit_code=0, stdout="file content\n"
                )
                content = await sandbox.read_file("test.txt")
                assert content == "file content\n"

    @pytest.mark.asyncio
    async def test_read_file_rejects_path_traversal(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with pytest.raises(DockerError) as exc_info:
                await sandbox.read_file("../../../etc/passwd")
            assert "invalid path" in str(exc_info.value).lower()


class TestDockerSandboxGit:
    """Tests for git operations."""

    @pytest.mark.asyncio
    async def test_get_current_commit(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(
                    exit_code=0, stdout="abc123def456\n"
                )
                commit = await sandbox.get_current_commit()
                assert commit == "abc123def456"

    @pytest.mark.asyncio
    async def test_get_git_status(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(
                    exit_code=0, stdout="M modified.txt\nA added.txt\n"
                )
                status = await sandbox.get_git_status()
                assert "modified.txt" in status


class TestDockerSandboxImageForLanguage:
    """Tests for image_for_language static method."""

    def test_python_image(self):
        assert DockerSandbox.image_for_language("python") == "python:3.11-slim"
        assert DockerSandbox.image_for_language("Python") == "python:3.11-slim"
        assert DockerSandbox.image_for_language("python3") == "python:3.11-slim"

    def test_javascript_image(self):
        assert DockerSandbox.image_for_language("javascript") == "node:20-slim"
        assert DockerSandbox.image_for_language("typescript") == "node:20-slim"
        assert DockerSandbox.image_for_language("node") == "node:20-slim"

    def test_go_image(self):
        assert DockerSandbox.image_for_language("go") == "golang:1.21-slim"
        assert DockerSandbox.image_for_language("golang") == "golang:1.21-slim"

    def test_rust_image(self):
        assert DockerSandbox.image_for_language("rust") == "rust:1.75-slim"

    def test_unknown_defaults_to_python(self):
        assert DockerSandbox.image_for_language("unknown") == "python:3.11-slim"
        assert DockerSandbox.image_for_language("") == "python:3.11-slim"


class TestDockerSandboxFromSpec:
    """Tests for DockerSandbox.from_spec."""

    def test_from_spec_creates_sandbox(self):
        client = create_mock_client()
        spec = ContainerSpec(
            name="custom-sandbox",
            image="node:20-slim",
            memory_mb=1024,
        )

        sandbox = DockerSandbox.from_spec(client, spec)
        assert sandbox._config.name == "custom-sandbox"
        assert sandbox._config.image == "node:20-slim"
        assert sandbox._config.memory_mb == 1024


class TestDockerSandboxTruncate:
    """Tests for string truncation."""

    def test_short_string_unchanged(self):
        client = create_mock_client()
        sandbox = DockerSandbox(client)

        result = sandbox._truncate("hello", 10)
        assert result == "hello"

    def test_long_string_truncated(self):
        client = create_mock_client()
        sandbox = DockerSandbox(client)

        result = sandbox._truncate("hello world this is long", 10)
        assert len(result) == 13
        assert result.endswith("...")

    def test_exact_length_unchanged(self):
        client = create_mock_client()
        sandbox = DockerSandbox(client)

        result = sandbox._truncate("1234567890", 10)
        assert result == "1234567890"

    def test_empty_string(self):
        client = create_mock_client()
        sandbox = DockerSandbox(client)

        result = sandbox._truncate("", 10)
        assert result == ""


class TestDockerSandboxSetupWorkspace:
    """Tests for setup_workspace."""

    @pytest.mark.asyncio
    async def test_setup_workspace_basic(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(exit_code=0)
                await sandbox.setup_workspace("https://github.com/owner/repo")

                assert sandbox.workspace_ready
                assert sandbox.repo_url == "https://github.com/owner/repo"

    @pytest.mark.asyncio
    async def test_setup_workspace_with_commit(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(exit_code=0)
                await sandbox.setup_workspace(
                    "https://github.com/owner/repo", commit="abc123"
                )

                assert sandbox.workspace_ready
                assert sandbox.commit == "abc123"

    @pytest.mark.asyncio
    async def test_setup_workspace_short_url(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(exit_code=0)
                await sandbox.setup_workspace("owner/repo")

                assert sandbox.repo_url == "https://github.com/owner/repo"

    @pytest.mark.asyncio
    async def test_setup_workspace_git_install_failure(self):
        client = create_mock_client()

        async with DockerSandbox(client) as sandbox:
            with patch(
                "swe_forge.execution.sandbox.exec_in_container",
                new_callable=AsyncMock,
            ) as mock_exec:
                mock_exec.return_value = create_exec_result(
                    exit_code=1, stderr="apt-get failed"
                )
                with pytest.raises(DockerError) as exc_info:
                    await sandbox.setup_workspace("owner/repo")
                assert "install git" in str(exc_info.value).lower()
