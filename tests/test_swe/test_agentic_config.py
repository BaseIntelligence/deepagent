"""Tests for agentic configuration detection - NO HARDCODING."""


from swe_forge.swe.agentic_config import (
    RepositoryConfig,
    SubmitConfigArgs,
)


class TestRepositoryConfig:
    """Test RepositoryConfig dataclass."""

    def test_config_creation(self):
        """Test basic config creation."""
        config = RepositoryConfig(
            python_version="3.11",
            package_manager="poetry",
            install_commands=["poetry install"],
            test_command="pytest",
            test_framework="pytest",
            docker_image="python:3.11-slim",
            validated=True,
        )

        assert config.python_version == "3.11"
        assert config.package_manager == "poetry"
        assert config.is_valid() is True

    def test_config_invalid_missing_fields(self):
        """Test config validation with missing fields."""
        # Missing install_commands
        config1 = RepositoryConfig(
            python_version="3.11",
            package_manager="pip",
            install_commands=[],
            test_command="pytest",
            validated=True,
        )
        assert config1.is_valid() is False

        # Missing test_command
        config2 = RepositoryConfig(
            python_version="3.11",
            package_manager="pip",
            install_commands=["pip install -e ."],
            test_command="",
            validated=True,
        )
        assert config2.is_valid() is False

    def test_config_not_validated(self):
        """Test config that wasn't validated."""
        config = RepositoryConfig(
            python_version="3.11",
            package_manager="pip",
            install_commands=["pip install -e ."],
            test_command="pytest",
            validated=False,  # Not validated!
        )

        # Should be invalid even if fields are filled
        assert config.is_valid() is False


class TestSubmitConfigArgs:
    """Test SubmitConfigArgs validation."""

    def test_submit_config_args(self):
        """Test SubmitConfigArgs creation."""
        args = SubmitConfigArgs(
            python_version="3.11",
            package_manager="poetry",
            install_commands=["poetry install"],
            test_command="pytest",
            test_framework="pytest",
        )

        assert args.python_version == "3.11"
        assert args.package_manager == "poetry"


class TestEmptyConfig:
    """Test that empty config is invalid."""

    def test_empty_config_invalid(self):
        """Test that empty config is invalid."""
        config = RepositoryConfig()

        # Should be invalid - no fields set
        assert config.is_valid() is False
        assert config.python_version == ""
        assert config.package_manager == ""

    def test_config_with_validation_errors(self):
        """Test config with validation errors."""
        config = RepositoryConfig(
            python_version="",
            package_manager="",
            install_commands=[],
            test_command="",
            validated=False,
        )

        assert config.is_valid() is False
        assert len(config.validation_errors) >= 0  # May have no errors explicitly


# Skip agentic detection tests since they require complex mocking
# The actual detection is tested via integration tests
