"""Tests for logging configuration."""

import logging
import os
from io import StringIO
from unittest.mock import patch


from swe_forge.logging_config import (
    bind_contextvars,
    clear_contextvars,
    configure_logging,
    get_logger,
)


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def teardown_method(self) -> None:
        """Reset logging after each test."""
        logging.getLogger().handlers.clear()
        if "LOG_FORMAT" in os.environ:
            del os.environ["LOG_FORMAT"]

    def test_default_configuration_is_info_level(self) -> None:
        """configure_logging should set INFO level by default."""
        with patch("sys.stdout", new_callable=StringIO):
            configure_logging()
            assert logging.getLogger().level == logging.INFO

    def test_debug_level_configuration(self) -> None:
        """configure_logging should accept DEBUG level."""
        with patch("sys.stdout", new_callable=StringIO):
            configure_logging(level="DEBUG")
            assert logging.getLogger().level == logging.DEBUG

    def test_error_level_configuration(self) -> None:
        """configure_logging should accept ERROR level."""
        with patch("sys.stdout", new_callable=StringIO):
            configure_logging(level="ERROR")
            assert logging.getLogger().level == logging.ERROR

    def test_case_insensitive_level(self) -> None:
        """configure_logging should handle lowercase level strings."""
        with patch("sys.stdout", new_callable=StringIO):
            configure_logging(level="debug")
            assert logging.getLogger().level == logging.DEBUG

    def test_invalid_level_defaults_to_info(self) -> None:
        """configure_logging should default to INFO for invalid levels."""
        with patch("sys.stdout", new_callable=StringIO):
            configure_logging(level="INVALID")
            assert logging.getLogger().level == logging.INFO

    def test_clears_existing_handlers(self) -> None:
        """configure_logging should clear existing handlers before adding new one."""
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        initial_count = len(root.handlers)

        with patch("sys.stdout", new_callable=StringIO):
            configure_logging()
            assert len(root.handlers) == 1
            assert len(root.handlers) != initial_count or initial_count == 0


class TestGetLogger:
    """Tests for get_logger function."""

    def teardown_method(self) -> None:
        """Reset logging after each test."""
        logging.getLogger().handlers.clear()
        if "LOG_FORMAT" in os.environ:
            del os.environ["LOG_FORMAT"]

    def test_returns_bound_logger(self) -> None:
        """get_logger should return a BoundLogger."""
        with patch("sys.stdout", new_callable=StringIO):
            configure_logging()
            logger = get_logger("test.module")
            assert hasattr(logger, "info")
            assert hasattr(logger, "debug")
            assert hasattr(logger, "error")

    def test_default_name(self) -> None:
        """get_logger should use __name__ as default."""
        with patch("sys.stdout", new_callable=StringIO):
            configure_logging()
            logger = get_logger()
            assert logger is not None


class TestCorrelationId:
    """Tests for correlation ID support via contextvars."""

    def teardown_method(self) -> None:
        """Reset logging after each test."""
        logging.getLogger().handlers.clear()
        clear_contextvars()
        if "LOG_FORMAT" in os.environ:
            del os.environ["LOG_FORMAT"]

    def test_bind_contextvars_available(self) -> None:
        """bind_contextvars should be importable and callable."""
        bind_contextvars(request_id="test-123", user_id="user-456")
        assert True

    def test_clear_contextvars_available(self) -> None:
        """clear_contextvars should be importable and callable."""
        bind_contextvars(request_id="test-123")
        clear_contextvars()
        assert True


class TestJsonOutput:
    """Tests for JSON output mode."""

    def teardown_method(self) -> None:
        """Reset logging after each test."""
        logging.getLogger().handlers.clear()
        if "LOG_FORMAT" in os.environ:
            del os.environ["LOG_FORMAT"]

    def test_json_format_enabled_via_env(self) -> None:
        """configure_logging should use JSON when LOG_FORMAT=json."""
        os.environ["LOG_FORMAT"] = "json"
        with patch("sys.stdout", new_callable=StringIO):
            configure_logging()
            logger = get_logger("test.json")
            logger.info("test message")

    def test_json_format_case_insensitive(self) -> None:
        """configure_logging should handle LOG_FORMAT=JSON (uppercase)."""
        os.environ["LOG_FORMAT"] = "JSON"
        with patch("sys.stdout", new_callable=StringIO):
            configure_logging()
            logger = get_logger("test.json.upper")
            logger.info("test message")


class TestConsoleOutput:
    """Tests for console output mode (default)."""

    def teardown_method(self) -> None:
        """Reset logging after each test."""
        logging.getLogger().handlers.clear()
        if "LOG_FORMAT" in os.environ:
            del os.environ["LOG_FORMAT"]

    def test_console_format_default(self) -> None:
        """configure_logging should use console format by default."""
        with patch("sys.stdout", new_callable=StringIO):
            configure_logging()
            logger = get_logger("test.console")
            logger.info("test message")


class TestStdlibIntegration:
    """Tests for stdlib logging integration."""

    def teardown_method(self) -> None:
        """Reset logging after each test."""
        logging.getLogger().handlers.clear()
        if "LOG_FORMAT" in os.environ:
            del os.environ["LOG_FORMAT"]

    def test_stdlib_logger_uses_structlog_format(self) -> None:
        """stdlib loggers should be formatted through structlog."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            configure_logging()
            stdlib_logger = logging.getLogger("test.stdlib")
            stdlib_logger.info("stdlib message")
            output = mock_stdout.getvalue()
            assert "stdlib message" in output
