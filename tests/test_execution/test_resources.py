"""Unit tests for Resource limits and cleanup policies.

These tests verify resource limit application to Docker configs
and cleanup policy behavior.
"""

from __future__ import annotations


from swe_forge.execution.resources import (
    CPU_PERIOD_US,
    DEFAULT_CPU_CORES,
    DEFAULT_DISK_GB,
    DEFAULT_MAX_PROCESSES,
    DEFAULT_MEMORY_MB,
    DEFAULT_TIMEOUT_SECONDS,
    OOM_EXIT_CODE,
    CleanupPolicy,
    ResourceLimits,
    apply_resource_limits,
    get_resource_limits,
    is_oom_killed,
    should_cleanup,
)


class TestConstants:
    """Tests for module constants."""

    def test_oom_exit_code(self):
        """OOM exit code should be 137 (SIGKILL)."""
        assert OOM_EXIT_CODE == 137

    def test_default_memory(self):
        """Default memory should be 512 MB."""
        assert DEFAULT_MEMORY_MB == 512

    def test_default_cpu_cores(self):
        """Default CPU should be 1.0 core."""
        assert DEFAULT_CPU_CORES == 1.0

    def test_default_disk(self):
        """Default disk should be 5 GB."""
        assert DEFAULT_DISK_GB == 5

    def test_default_max_processes(self):
        """Default max processes should be 100."""
        assert DEFAULT_MAX_PROCESSES == 100

    def test_default_timeout(self):
        """Default timeout should be 20 minutes."""
        assert DEFAULT_TIMEOUT_SECONDS == 1200

    def test_cpu_period(self):
        """CPU period should be 100ms (100000 microseconds)."""
        assert CPU_PERIOD_US == 100_000


class TestCleanupPolicy:
    """Tests for CleanupPolicy enum."""

    def test_always_value(self):
        """ALWAYS should have value 'always'."""
        assert CleanupPolicy.ALWAYS.value == "always"

    def test_on_failure_value(self):
        """ON_FAILURE should have value 'on_failure'."""
        assert CleanupPolicy.ON_FAILURE.value == "on_failure"

    def test_never_value(self):
        """NEVER should have value 'never'."""
        assert CleanupPolicy.NEVER.value == "never"


class TestShouldCleanup:
    """Tests for should_cleanup function."""

    def test_always_policy_success(self):
        """ALWAYS policy should cleanup on success."""
        assert should_cleanup(CleanupPolicy.ALWAYS, success=True) is True

    def test_always_policy_failure(self):
        """ALWAYS policy should cleanup on failure."""
        assert should_cleanup(CleanupPolicy.ALWAYS, success=False) is True

    def test_on_failure_policy_success(self):
        """ON_FAILURE policy should NOT cleanup on success."""
        assert should_cleanup(CleanupPolicy.ON_FAILURE, success=True) is False

    def test_on_failure_policy_failure(self):
        """ON_FAILURE policy should cleanup on failure."""
        assert should_cleanup(CleanupPolicy.ON_FAILURE, success=False) is True

    def test_never_policy_success(self):
        """NEVER policy should NOT cleanup on success."""
        assert should_cleanup(CleanupPolicy.NEVER, success=True) is False

    def test_never_policy_failure(self):
        """NEVER policy should NOT cleanup on failure."""
        assert should_cleanup(CleanupPolicy.NEVER, success=False) is False


class TestIsOomKilled:
    """Tests for is_oom_killed function."""

    def test_oom_exit_code_returns_true(self):
        """Exit code 137 should indicate OOM kill."""
        assert is_oom_killed(137) is True

    def test_zero_exit_code_returns_false(self):
        """Exit code 0 should not indicate OOM kill."""
        assert is_oom_killed(0) is False

    def test_exit_code_1_returns_false(self):
        """Exit code 1 should not indicate OOM kill."""
        assert is_oom_killed(1) is False

    def test_none_exit_code_returns_false(self):
        """None exit code should not indicate OOM kill."""
        assert is_oom_killed(None) is False

    def test_other_exit_codes_return_false(self):
        """Other exit codes should not indicate OOM kill."""
        assert is_oom_killed(127) is False  # Command not found
        assert is_oom_killed(126) is False  # Permission denied
        assert is_oom_killed(139) is False  # Segfault (just above 137)


class TestResourceLimits:
    """Tests for ResourceLimits dataclass."""

    def test_default_values(self):
        """Default ResourceLimits should use module defaults."""
        limits = ResourceLimits()
        assert limits.memory_mb == DEFAULT_MEMORY_MB
        assert limits.cpu_cores == DEFAULT_CPU_CORES
        assert limits.disk_gb == DEFAULT_DISK_GB
        assert limits.max_processes == DEFAULT_MAX_PROCESSES
        assert limits.timeout_seconds == DEFAULT_TIMEOUT_SECONDS

    def test_custom_values(self):
        """ResourceLimits should accept custom values."""
        limits = ResourceLimits(
            memory_mb=2048,
            cpu_cores=2.0,
            disk_gb=10,
            max_processes=200,
            timeout_seconds=2400,
        )
        assert limits.memory_mb == 2048
        assert limits.cpu_cores == 2.0
        assert limits.disk_gb == 10
        assert limits.max_processes == 200
        assert limits.timeout_seconds == 2400

    def test_memory_bytes_conversion(self):
        """memory_bytes should convert MB to bytes correctly."""
        limits = ResourceLimits(memory_mb=512)
        assert limits.memory_bytes() == 512 * 1024 * 1024

    def test_memory_bytes_large(self):
        """memory_bytes should handle large values."""
        limits = ResourceLimits(memory_mb=8192)
        assert limits.memory_bytes() == 8_589_934_592

    def test_cpu_period(self):
        """cpu_period should return standard 100ms period."""
        limits = ResourceLimits()
        assert limits.cpu_period() == CPU_PERIOD_US

    def test_cpu_quota_single_core(self):
        """cpu_quota for 1.0 core should equal cpu_period."""
        limits = ResourceLimits(cpu_cores=1.0)
        assert limits.cpu_quota() == CPU_PERIOD_US

    def test_cpu_quota_half_core(self):
        """cpu_quota for 0.5 core should be half the period."""
        limits = ResourceLimits(cpu_cores=0.5)
        assert limits.cpu_quota() == CPU_PERIOD_US // 2

    def test_cpu_quota_double_core(self):
        """cpu_quota for 2.0 cores should be double the period."""
        limits = ResourceLimits(cpu_cores=2.0)
        assert limits.cpu_quota() == CPU_PERIOD_US * 2

    def test_disk_bytes_conversion(self):
        """disk_bytes should convert GB to bytes correctly."""
        limits = ResourceLimits(disk_gb=10)
        assert limits.disk_bytes() == 10 * 1024 * 1024 * 1024

    def test_nano_cpus_single(self):
        """nano_cpus for 1.0 core should be 1 billion."""
        limits = ResourceLimits(cpu_cores=1.0)
        assert limits.nano_cpus() == 1_000_000_000

    def test_nano_cpus_fractional(self):
        """nano_cpus for fractional cores should be correct."""
        limits = ResourceLimits(cpu_cores=0.5)
        assert limits.nano_cpus() == 500_000_000

    def test_nano_cpus_multiple(self):
        """nano_cpus for multiple cores should be correct."""
        limits = ResourceLimits(cpu_cores=4.0)
        assert limits.nano_cpus() == 4_000_000_000


class TestApplyResourceLimits:
    """Tests for apply_resource_limits function."""

    def test_creates_host_config_if_missing(self):
        """Should create HostConfig if not present."""
        config = {"Image": "python:3.11"}
        limits = ResourceLimits(memory_mb=1024)

        result = apply_resource_limits(config, limits)

        assert "HostConfig" in result
        assert result["HostConfig"]["Memory"] == 1024 * 1024 * 1024

    def test_preserves_existing_host_config(self):
        """Should preserve existing HostConfig values."""
        config = {
            "Image": "python:3.11",
            "HostConfig": {"NetworkMode": "none"},
        }
        limits = ResourceLimits(memory_mb=512)

        result = apply_resource_limits(config, limits)

        assert result["HostConfig"]["NetworkMode"] == "none"

    def test_sets_memory_limit(self):
        """Should set Memory limit correctly."""
        config = {}
        limits = ResourceLimits(memory_mb=2048)

        apply_resource_limits(config, limits)

        assert config["HostConfig"]["Memory"] == 2048 * 1024 * 1024

    def test_sets_memory_swap(self):
        """Should set MemorySwap to -1 (disable swap limit)."""
        config = {}
        limits = ResourceLimits()

        apply_resource_limits(config, limits)

        assert config["HostConfig"]["MemorySwap"] == -1

    def test_sets_nano_cpus(self):
        """Should set NanoCpus correctly."""
        config = {}
        limits = ResourceLimits(cpu_cores=2.0)

        apply_resource_limits(config, limits)

        assert config["HostConfig"]["NanoCpus"] == 2_000_000_000

    def test_sets_cpu_period(self):
        """Should set CpuPeriod to standard 100ms."""
        config = {}
        limits = ResourceLimits()

        apply_resource_limits(config, limits)

        assert config["HostConfig"]["CpuPeriod"] == CPU_PERIOD_US

    def test_sets_cpu_quota(self):
        """Should set CpuQuota based on cores."""
        config = {}
        limits = ResourceLimits(cpu_cores=2.0)

        apply_resource_limits(config, limits)

        assert config["HostConfig"]["CpuQuota"] == CPU_PERIOD_US * 2

    def test_sets_pids_limit(self):
        """Should set PidsLimit for max processes."""
        config = {}
        limits = ResourceLimits(max_processes=200)

        apply_resource_limits(config, limits)

        assert config["HostConfig"]["PidsLimit"] == 200

    def test_returns_modified_config(self):
        """Should return the modified config."""
        config = {"Image": "python:3.11"}
        limits = ResourceLimits()

        result = apply_resource_limits(config, limits)

        assert result is config
        assert "HostConfig" in result


class TestGetResourceLimits:
    """Tests for get_resource_limits function."""

    def test_easy_limits(self):
        """Easy difficulty should have light resources."""
        limits = get_resource_limits("easy")

        assert limits.memory_mb == 512
        assert limits.cpu_cores == 0.5
        assert limits.disk_gb == 2
        assert limits.max_processes == 50
        assert limits.timeout_seconds == 600

    def test_medium_limits(self):
        """Medium difficulty should have moderate resources."""
        limits = get_resource_limits("medium")

        assert limits.memory_mb == 1024
        assert limits.cpu_cores == 1.0
        assert limits.disk_gb == 5
        assert limits.max_processes == 100
        assert limits.timeout_seconds == 1200

    def test_hard_limits(self):
        """Hard difficulty should have heavy resources."""
        limits = get_resource_limits("hard")

        assert limits.memory_mb == 2048
        assert limits.cpu_cores == 2.0
        assert limits.disk_gb == 10
        assert limits.max_processes == 200
        assert limits.timeout_seconds == 2400

    def test_expert_limits(self):
        """Expert difficulty should have very high resources."""
        limits = get_resource_limits("expert")

        assert limits.memory_mb == 4096
        assert limits.cpu_cores == 4.0
        assert limits.disk_gb == 20
        assert limits.max_processes == 500
        assert limits.timeout_seconds == 4800

    def test_nightmare_limits(self):
        """Nightmare difficulty should have maximum resources."""
        limits = get_resource_limits("nightmare")

        assert limits.memory_mb == 8192
        assert limits.cpu_cores == 8.0
        assert limits.disk_gb == 50
        assert limits.max_processes == 1000
        assert limits.timeout_seconds == 9000

    def test_case_insensitive_lowercase(self):
        """Should handle lowercase difficulty."""
        limits = get_resource_limits("hard")
        assert limits.memory_mb == 2048

    def test_case_insensitive_uppercase(self):
        """Should handle uppercase difficulty."""
        limits = get_resource_limits("HARD")
        assert limits.memory_mb == 2048

    def test_case_insensitive_mixed_case(self):
        """Should handle mixed case difficulty."""
        limits = get_resource_limits("Hard")
        assert limits.memory_mb == 2048

    def test_unknown_difficulty_defaults_to_medium(self):
        """Unknown difficulty should default to medium."""
        limits = get_resource_limits("unknown")
        medium = get_resource_limits("medium")

        assert limits.memory_mb == medium.memory_mb
        assert limits.cpu_cores == medium.cpu_cores
        assert limits.disk_gb == medium.disk_gb
        assert limits.max_processes == medium.max_processes
        assert limits.timeout_seconds == medium.timeout_seconds

    def test_empty_string_defaults_to_medium(self):
        """Empty string should default to medium."""
        limits = get_resource_limits("")
        medium = get_resource_limits("medium")

        assert limits.memory_mb == medium.memory_mb


class TestResourceLimitsIntegration:
    """Integration tests for resource limits."""

    def test_apply_easy_limits_to_config(self):
        """Applying easy limits to config should set correct values."""
        config = {"Image": "python:3.11"}
        limits = get_resource_limits("easy")

        result = apply_resource_limits(config, limits)

        # Memory: 512 MB in bytes
        assert result["HostConfig"]["Memory"] == 512 * 1024 * 1024
        # CPU: 0.5 cores
        assert result["HostConfig"]["NanoCpus"] == 500_000_000
        # PIDs: 50
        assert result["HostConfig"]["PidsLimit"] == 50

    def test_apply_nightmare_limits_to_config(self):
        """Applying nightmare limits to config should set correct values."""
        config = {"Image": "python:3.11"}
        limits = get_resource_limits("nightmare")

        result = apply_resource_limits(config, limits)

        # Memory: 8192 MB in bytes
        assert result["HostConfig"]["Memory"] == 8192 * 1024 * 1024
        # CPU: 8 cores
        assert result["HostConfig"]["NanoCpus"] == 8_000_000_000
        # PIDs: 1000
        assert result["HostConfig"]["PidsLimit"] == 1000

    def test_full_config_transformation(self):
        """Test full config transformation with all limits."""
        config = {
            "Image": "python:3.11",
            "Cmd": ["python", "-c", "print('hello')"],
            "Env": ["FOO=bar"],
            "WorkingDir": "/workspace",
        }
        limits = ResourceLimits(
            memory_mb=4096,
            cpu_cores=4.0,
            disk_gb=20,
            max_processes=500,
            timeout_seconds=4800,
        )

        result = apply_resource_limits(config, limits)

        # Original config should be preserved
        assert result["Image"] == "python:3.11"
        assert result["Cmd"] == ["python", "-c", "print('hello')"]
        assert result["Env"] == ["FOO=bar"]
        assert result["WorkingDir"] == "/workspace"

        # HostConfig should be added
        assert result["HostConfig"]["Memory"] == 4096 * 1024 * 1024
        assert result["HostConfig"]["NanoCpus"] == 4_000_000_000
        assert result["HostConfig"]["PidsLimit"] == 500
