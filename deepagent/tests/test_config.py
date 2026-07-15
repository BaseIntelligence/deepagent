"""Config loads OpenRouter model ids from env without printing secrets."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.config import (
    DEFAULT_BUDGET_USD,
    DEFAULT_PANEL_MODELS,
    DEFAULT_TEACHER_MODEL,
    FactorySettings,
    load_settings,
)


def test_defaults_match_mission_models() -> None:
    settings = load_settings(
        env={
            "OPENROUTER_API_KEY": "sk-or-v1-test-secret-key-do-not-leak",
        }
    )
    assert settings.teacher_model == DEFAULT_TEACHER_MODEL
    assert settings.teacher_model == "anthropic/claude-opus-4.8"
    assert settings.panel_models == list(DEFAULT_PANEL_MODELS)
    # VAL-RPANEL-001: Real-PR pair only (Grok + Kimi); Opus not required.
    assert settings.panel_models == [
        "x-ai/grok-4.5",
        "moonshotai/kimi-k2.6",
    ]
    assert settings.budget_usd == DEFAULT_BUDGET_USD
    assert settings.budget_usd == 600.0
    assert settings.has_api_key() is True


def test_env_overrides_models_and_budget() -> None:
    settings = load_settings(
        env={
            "OPENROUTER_API_KEY": "************************",
            "FACTORY_TEACHER_MODEL": "anthropic/claude-opus-4.8",
            "FACTORY_PANEL_MODELS": "x-ai/grok-4.5,moonshotai/kimi-k2.6",
            "FACTORY_BUDGET_USD": "600",
            "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        }
    )
    assert settings.teacher_model == "anthropic/claude-opus-4.8"
    assert settings.panel_models == [
        "x-ai/grok-4.5",
        "moonshotai/kimi-k2.6",
    ]
    assert settings.budget_usd == 600.0
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_masked_summary_hides_api_key() -> None:
    secret = "sk-or-v1-super-secret-never-print-me-9876543210"
    settings = load_settings(env={"OPENROUTER_API_KEY": secret})
    summary = settings.masked_summary()
    blob = json.dumps(summary)
    assert secret not in blob
    assert summary["openrouter_api_key"] == "***"
    assert summary["teacher_model"] == "anthropic/claude-opus-4.8"
    assert "x-ai/grok-4.5" in summary["panel_models"]  # type: ignore[operator]
    assert secret not in repr(settings)
    assert "***" in repr(settings)


def test_cli_config_json_masks_secret() -> None:
    secret = "sk-or-v1-cli-config-secret-xyz"
    runner = CliRunner()
    env = {
        **os.environ,
        "OPENROUTER_API_KEY": secret,
        "FACTORY_TEACHER_MODEL": "anthropic/claude-opus-4.8",
        "FACTORY_PANEL_MODELS": "x-ai/grok-4.5,moonshotai/kimi-k2.6",
        "FACTORY_BUDGET_USD": "600",
    }
    result = runner.invoke(app, ["config", "--json"], env=env)
    assert result.exit_code == 0, result.output
    assert secret not in result.output
    payload = json.loads(result.output)
    assert payload["teacher_model"] == "anthropic/claude-opus-4.8"
    assert payload["panel_models"] == [
        "x-ai/grok-4.5",
        "moonshotai/kimi-k2.6",
    ]
    assert payload["budget_usd"] == 600.0
    assert payload["openrouter_api_key"] == "***"


def test_settings_from_env_file_without_leaking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-or-v1-dotenv-secret-abc"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"OPENROUTER_API_KEY={secret}",
                "FACTORY_TEACHER_MODEL=anthropic/claude-opus-4.8",
                "FACTORY_PANEL_MODELS=x-ai/grok-4.5,moonshotai/kimi-k2.6",
                "FACTORY_BUDGET_USD=600",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    # Clear process keys so file is the source.
    for key in (
        "OPENROUTER_API_KEY",
        "FACTORY_TEACHER_MODEL",
        "FACTORY_PANEL_MODELS",
        "FACTORY_BUDGET_USD",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = FactorySettings()  # type: ignore[call-arg]
    assert settings.teacher_model == "anthropic/claude-opus-4.8"
    assert settings.has_api_key()
    assert secret not in settings.masked_summary().__repr__()
    assert secret not in repr(settings)
