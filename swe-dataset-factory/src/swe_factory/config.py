"""Environment-driven factory settings (OpenRouter only for V1)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_TEACHER_MODEL = "anthropic/claude-opus-4.8"
# Real-PR wave hardness panel (VAL-RPANEL-001): Grok 4.5 + Kimi K2.6 only.
# Opus is not required for this wave (may be re-enabled via FACTORY_PANEL_MODELS).
DEFAULT_PANEL_MODELS: tuple[str, ...] = (
    "x-ai/grok-4.5",
    "moonshotai/kimi-k2.6",
)
# Historical M7 triad (sealed offline fixtures may still reference Opus as optional).
HISTORICAL_TRIAD_PANEL_MODELS: tuple[str, ...] = (
    "x-ai/grok-4.5",
    "moonshotai/kimi-k2.6",
    "anthropic/claude-opus-4.8",
)
DEFAULT_BUDGET_USD = 600.0
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class FactorySettings(BaseSettings):
    """Load model ids and budget from environment without exposing secrets."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # pydantic-settings reads .env for model fields only (does not mutate os.environ).
        populate_by_name=True,
    )

    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )
    openrouter_base_url: str = Field(
        default=DEFAULT_OPENROUTER_BASE_URL,
        validation_alias="OPENROUTER_BASE_URL",
    )
    teacher_model: str = Field(
        default=DEFAULT_TEACHER_MODEL,
        validation_alias="FACTORY_TEACHER_MODEL",
    )
    panel_models_raw: str = Field(
        default=",".join(DEFAULT_PANEL_MODELS),
        validation_alias="FACTORY_PANEL_MODELS",
    )
    budget_usd: float = Field(
        default=DEFAULT_BUDGET_USD,
        validation_alias="FACTORY_BUDGET_USD",
    )
    github_token: SecretStr | None = Field(default=None, validation_alias="GITHUB_TOKEN")

    @field_validator("teacher_model")
    @classmethod
    def _teacher_nonempty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("FACTORY_TEACHER_MODEL must be non-empty")
        return cleaned

    @field_validator("budget_usd")
    @classmethod
    def _budget_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("FACTORY_BUDGET_USD must be > 0")
        return value

    @property
    def panel_models(self) -> list[str]:
        models = [part.strip() for part in self.panel_models_raw.split(",") if part.strip()]
        if not models:
            raise ValueError("FACTORY_PANEL_MODELS must list at least one model id")
        return models

    def has_api_key(self) -> bool:
        key = self.openrouter_api_key
        return key is not None and bool(key.get_secret_value().strip())

    def masked_summary(self) -> dict[str, object]:
        """Public config view safe for logs and CLI (never includes raw secrets)."""
        key_present = self.has_api_key()
        return {
            "openrouter_base_url": self.openrouter_base_url,
            "openrouter_api_key": "***" if key_present else None,
            "teacher_model": self.teacher_model,
            "panel_models": self.panel_models,
            "budget_usd": self.budget_usd,
            "github_token": "***" if self.github_token else None,
        }

    def __repr__(self) -> str:
        return (
            "FactorySettings("
            f"teacher_model={self.teacher_model!r}, "
            f"panel_models={self.panel_models!r}, "
            f"budget_usd={self.budget_usd!r}, "
            f"openrouter_api_key={'***' if self.has_api_key() else None}, "
            f"openrouter_base_url={self.openrouter_base_url!r})"
        )


def _settings_from_values(
    *,
    openrouter_api_key: str | None,
    openrouter_base_url: str,
    teacher_model: str,
    panel_models_raw: str,
    budget_usd: float,
    github_token: str | None,
) -> FactorySettings:
    """Construct settings via field names (mypy-friendly, no provider calls)."""
    data: dict[str, Any] = {
        "openrouter_base_url": openrouter_base_url,
        "teacher_model": teacher_model,
        "panel_models_raw": panel_models_raw,
        "budget_usd": budget_usd,
    }
    if openrouter_api_key is not None:
        data["openrouter_api_key"] = SecretStr(openrouter_api_key)
    if github_token is not None:
        data["github_token"] = SecretStr(github_token)
    return FactorySettings.model_validate(data)


@lru_cache(maxsize=1)
def get_settings() -> FactorySettings:
    """Cached settings load from process environment / optional .env file."""
    return FactorySettings()


def load_settings(*, env: dict[str, str] | None = None, dotenv: bool = True) -> FactorySettings:
    """Load settings, optionally from an explicit env mapping (no provider calls).

    When ``env`` is provided, values are taken from that mapping only (dot-env file ignored).
    """
    if env is not None:
        budget_raw = env.get("FACTORY_BUDGET_USD", str(DEFAULT_BUDGET_USD))
        return _settings_from_values(
            openrouter_api_key=env.get("OPENROUTER_API_KEY"),
            openrouter_base_url=env.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            teacher_model=env.get("FACTORY_TEACHER_MODEL", DEFAULT_TEACHER_MODEL),
            panel_models_raw=env.get("FACTORY_PANEL_MODELS", ",".join(DEFAULT_PANEL_MODELS)),
            budget_usd=float(budget_raw),
            github_token=env.get("GITHUB_TOKEN"),
        )
    if not dotenv:
        return FactorySettings(_env_file=None)  # type: ignore[call-arg]
    return FactorySettings()
