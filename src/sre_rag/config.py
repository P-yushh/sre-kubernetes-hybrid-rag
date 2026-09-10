"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

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


@lru_cache
def get_settings() -> Settings:
    """Return one validated settings object per process."""

    return Settings()
