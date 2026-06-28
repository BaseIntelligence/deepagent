"""Env-driven settings for the forge LLM layer.

All endpoint configuration is read from environment variables (and the gitignored
``.env``). No provider hostname or brand string is hardcoded here; defaults are
empty so a set environment variable is never overridden. Extra/unknown env vars
are ignored so the live environment (which carries many unrelated vars) never
breaks settings construction.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ForgeSettings(BaseSettings):
    """Endpoint configuration for the teacher (and optional panel) LLM layer."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    teacher_llm_base_url: str = Field(
        default="",
        description="Teacher endpoint base URL (host-only for anthropic, <base>/v1 for openai)",
    )
    teacher_llm_api_key: str = Field(default="", description="Teacher endpoint API key")
    teacher_llm_model: str = Field(
        default="",
        description="Provider-prefixed teacher model id, e.g. 'anthropic/<id>' or 'openai/<id>'",
    )
    teacher_llm_provider: str = Field(
        default="",
        description="Teacher provider routing hint, e.g. 'anthropic' or 'openai'",
    )

    panel_llm_base_url: str = Field(
        default="",
        description="Optional panel endpoint base URL; falls back to the teacher base URL when unset",
    )
    panel_llm_api_key: str = Field(
        default="",
        description="Optional panel endpoint API key; falls back to the teacher key when unset",
    )

    @property
    def effective_panel_base_url(self) -> str:
        """Panel base URL, defaulting to the teacher base URL when no override is set."""
        return self.panel_llm_base_url or self.teacher_llm_base_url

    @property
    def effective_panel_api_key(self) -> str:
        """Panel API key, defaulting to the teacher key when no override is set."""
        return self.panel_llm_api_key or self.teacher_llm_api_key
