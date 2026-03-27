"""Resource limits and cleanup policies for Docker container execution.

This module provides execution resource limits based on difficulty levels
and cleanup policies for container lifecycle management.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# =============================================================================
# Constants
# =============================================================================

#: Exit code indicating OOM kill (SIGKILL, typically from OOM or manual kill)
OOM_EXIT_CODE = 137

#: Default memory limit in MB
DEFAULT_MEMORY_MB = 512

#: Default CPU limit (1.0 = 1 full CPU core)
DEFAULT_CPU_CORES = 1.0

#: Default disk space limit in GB
DEFAULT_DISK_GB = 5

#: Default maximum processes
DEFAULT_MAX_PROCESSES = 100

#: Default timeout in seconds (20 minutes)
DEFAULT_TIMEOUT_SECONDS = 1200

#: CPU period in microseconds (100ms is the standard period)
CPU_PERIOD_US = 100_000


# =============================================================================
# Enums
# =============================================================================


class CleanupPolicy(Enum):
    """Policy for container cleanup after execution.

    Attributes:
        ALWAYS: Always remove containers after execution (default).
        ON_FAILURE: Only remove containers when execution fails.
        NEVER: Never automatically remove containers.
    """

    ALWAYS = "always"
    ON_FAILURE = "on_failure"
    NEVER = "never"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class ResourceLimits:
    """Execution resource limits for a container.

    These limits control the resources available to a container
    during task execution.

    Attributes:
        memory_mb: Memory limit in megabytes.
        cpu_cores: CPU cores available (e.g., 0.5, 1.0, 2.0).
        disk_gb: Disk space limit in gigabytes.
        max_processes: Maximum number of processes allowed (PIDs limit).
        timeout_seconds: Timeout in seconds before the container is killed.
    """

    memory_mb: int = DEFAULT_MEMORY_MB
    cpu_cores: float = DEFAULT_CPU_CORES
    disk_gb: int = DEFAULT_DISK_GB
    max_processes: int = DEFAULT_MAX_PROCESSES
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    def memory_bytes(self) -> int:
        """Return memory limit in bytes.

        Returns:
            Memory limit as number of bytes.
        """
        return self.memory_mb * 1024 * 1024

    def cpu_period(self) -> int:
        """Return CPU period in microseconds.

        Uses the standard 100ms (100,000 microseconds) period.

        Returns:
            CPU scheduler period in microseconds.
        """
        return CPU_PERIOD_US

    def cpu_quota(self) -> int:
        """Return CPU quota in microseconds.

        The quota is calculated as: quota = period * cores
        For example, 1.0 core = 100,000 quota (100% of one CPU)

        Returns:
            CPU quota in microseconds.
        """
        return int(self.cpu_period() * self.cpu_cores)

    def disk_bytes(self) -> int:
        """Return disk space limit in bytes.

        Returns:
            Disk limit as number of bytes.
        """
        return self.disk_gb * 1024 * 1024 * 1024

    def nano_cpus(self) -> int:
        """Return CPU limit in nanoseconds (NanoCpus format).

        Docker's NanoCpus is 1 billion nanoseconds per CPU.
        For example, 1.0 CPU = 1,000,000,000 nanoseconds.

        Returns:
            CPU limit in nanoseconds.
        """
        return int(self.cpu_cores * 1_000_000_000)


# =============================================================================
# Helper Functions
# =============================================================================


def is_oom_killed(exit_code: int | None) -> bool:
    """Check if the exit code indicates an OOM kill.

    Exit code 137 indicates the container was killed with SIGKILL,
    which typically happens due to OOM or a manual kill signal.

    Args:
        exit_code: The exit code from container execution.

    Returns:
        True if the exit code indicates OOM kill.

    Example:
        >>> is_oom_killed(137)
        True
        >>> is_oom_killed(0)
        False
    """
    return exit_code == OOM_EXIT_CODE


def apply_resource_limits(
    container_config: dict[str, Any],
    limits: ResourceLimits,
) -> dict[str, Any]:
    """Apply resource limits to a Docker container configuration.

    Modifies the container config's HostConfig to include resource limits.
    Creates HostConfig if it doesn't exist.

    Args:
        container_config: Docker container configuration dict to modify.
        limits: ResourceLimits instance with the limits to apply.

    Returns:
        The modified container configuration dict.

    Example:
        >>> config = {"Image": "ubuntu:24.04"}
        >>> limits = ResourceLimits(memory_mb=1024, cpu_cores=2.0)
        >>> apply_resource_limits(config, limits)
        {'Image': 'ubuntu:24.04', 'HostConfig': {'Memory': 1073741824, ...}}
    """
    if "HostConfig" not in container_config:
        container_config["HostConfig"] = {}

    host_config = container_config["HostConfig"]

    # Memory limit in bytes
    host_config["Memory"] = limits.memory_bytes()

    # Disable swap limit (set to -1 to use same as memory)
    host_config["MemorySwap"] = -1

    # CPU limits
    # Using NanoCpus for Docker API compatibility
    host_config["NanoCpus"] = limits.nano_cpus()

    # CPU quota/period (alternative method, also supported)
    host_config["CpuPeriod"] = limits.cpu_period()
    host_config["CpuQuota"] = limits.cpu_quota()

    # Process limit (PIDs)
    host_config["PidsLimit"] = limits.max_processes

    return container_config


def get_resource_limits(difficulty: str) -> ResourceLimits:
    """Get execution limits based on difficulty level.

    Supported difficulty levels (case-insensitive):
        - "easy": Light resources for simple tasks
        - "medium": Moderate resources for standard tasks (default)
        - "hard": Heavy resources for complex tasks
        - "expert": Very high resources for expert-level tasks
        - "nightmare": Maximum resources for extreme tasks

    Unknown difficulty levels default to "medium".

    Args:
        difficulty: The difficulty level as a string (case-insensitive).

    Returns:
        ResourceLimits appropriate for the difficulty level.

    Example:
        >>> limits = get_resource_limits("hard")
        >>> limits.memory_mb
        2048
        >>> limits.cpu_cores
        2.0
    """
    difficulty_lower = difficulty.lower()

    if difficulty_lower == "easy":
        return ResourceLimits(
            memory_mb=512,
            cpu_cores=0.5,
            disk_gb=2,
            max_processes=50,
            timeout_seconds=600,  # 10 minutes
        )

    if difficulty_lower == "medium":
        return ResourceLimits(
            memory_mb=1024,
            cpu_cores=1.0,
            disk_gb=5,
            max_processes=100,
            timeout_seconds=1200,  # 20 minutes
        )

    if difficulty_lower == "hard":
        return ResourceLimits(
            memory_mb=2048,
            cpu_cores=2.0,
            disk_gb=10,
            max_processes=200,
            timeout_seconds=2400,  # 40 minutes
        )

    if difficulty_lower == "expert":
        return ResourceLimits(
            memory_mb=4096,
            cpu_cores=4.0,
            disk_gb=20,
            max_processes=500,
            timeout_seconds=4800,  # 80 minutes
        )

    if difficulty_lower == "nightmare":
        return ResourceLimits(
            memory_mb=8192,
            cpu_cores=8.0,
            disk_gb=50,
            max_processes=1000,
            timeout_seconds=9000,  # 150 minutes
        )

    # Unknown difficulty defaults to medium
    return ResourceLimits(
        memory_mb=1024,
        cpu_cores=1.0,
        disk_gb=5,
        max_processes=100,
        timeout_seconds=1200,
    )


def should_cleanup(policy: CleanupPolicy, success: bool) -> bool:
    """Determine if cleanup should occur based on policy and execution result.

    Args:
        policy: The cleanup policy to apply.
        success: Whether the execution was successful.

    Returns:
        True if cleanup should be performed.

    Example:
        >>> should_cleanup(CleanupPolicy.ALWAYS, True)
        True
        >>> should_cleanup(CleanupPolicy.ON_FAILURE, True)
        False
        >>> should_cleanup(CleanupPolicy.NEVER, False)
        False
    """
    if policy == CleanupPolicy.ALWAYS:
        return True

    if policy == CleanupPolicy.ON_FAILURE:
        return not success

    # CleanupPolicy.NEVER
    return False
