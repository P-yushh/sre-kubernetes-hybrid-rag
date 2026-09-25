"""Tests for environment-backed application configuration."""

import pytest

from sre_rag.config import Settings


def test_generation_settings_use_service_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SRE_RAG_OPENAI_GENERATION_MODEL", "test-model")
    monkeypatch.setenv("SRE_RAG_OPENAI_EVALUATION_MODEL", "evaluation-model")
    monkeypatch.setenv("SRE_RAG_OLLAMA_MODEL", "local-model")
    monkeypatch.setenv("SRE_RAG_GENERATION_MAX_OUTPUT_TOKENS", "321")
    monkeypatch.setenv("SRE_RAG_API_BASE_URL", "http://localhost:9000")
    monkeypatch.setenv("SRE_RAG_QDRANT_URL", "http://localhost:7333")
    monkeypatch.setenv("SRE_RAG_QDRANT_COLLECTION_NAME", "test_chunks")

    settings = Settings()

    assert settings.openai_generation_model == "test-model"
    assert settings.openai_evaluation_model == "evaluation-model"
    assert settings.ollama_model == "local-model"
    assert settings.generation_max_output_tokens == 321
    assert settings.api_base_url == "http://localhost:9000"
    assert settings.qdrant_url == "http://localhost:7333"
    assert settings.qdrant_collection_name == "test_chunks"


def test_qdrant_collection_name_cannot_be_empty() -> None:
    with pytest.raises(ValueError, match="at least 1 character"):
        Settings(qdrant_collection_name="")
