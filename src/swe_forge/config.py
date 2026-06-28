"""Configuration module using pydantic-settings for environment-based configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    github_token: str = Field(..., description="GitHub personal access token")
    openai_api_key: str = Field(default="", description="OpenAI API key (optional)")
    anthropic_api_key: str = Field(
        default="", description="Anthropic API key (optional)"
    )
    oxylabs_username: str = Field(default="", description="Oxylabs proxy username")
    oxylabs_password: str = Field(default="", description="Oxylabs proxy password")
    oxylabs_rps: int = Field(default=40, description="Oxylabs max requests per second")
    log_level: str = Field(default="INFO", description="Logging level")
    model_name: str = Field(
        default="openai/gpt-4o-mini", description="Default model name for LLM calls"
    )


class PipelineConfig:
    """Configuration for pipeline concurrency limits."""

    max_concurrent_gharchive: int = 8
    max_concurrent_enrichment: int = 20
    max_concurrent_llm: int = 25


class HarnessConfig:
    """Configuration for test harness execution."""

    # DEPRECATED: Use agentic_config for image detection.
    docker_image: str | None = None  # Agent detects, no default
    agent_timeout: int = 600
    test_timeout: int = 120
