"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "staging", "production"]


class Settings(BaseSettings):
    """Runtime settings for the API service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SRE_RAG_",
        extra="ignore",
    )

    app_name: str = "SRE Kubernetes Hybrid RAG"
    app_version: str = "0.1.0"
    environment: Environment = "development"
    log_level: str = "INFO"
    openai_api_key: SecretStr | None = None
    openai_generation_model: str = "gpt-5.6-luna"
    openai_evaluation_model: str = "gpt-5.6-terra"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    generation_max_output_tokens: int = Field(default=800, gt=0)
    generation_timeout_seconds: float = Field(default=30.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    """Return one validated settings object per process."""

    return Settings()
