"""Tests for environment-backed application configuration."""

import pytest

from sre_rag.config import Settings


def test_generation_settings_use_service_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SRE_RAG_OPENAI_GENERATION_MODEL", "test-model")
    monkeypatch.setenv("SRE_RAG_OPENAI_EVALUATION_MODEL", "evaluation-model")
    monkeypatch.setenv("SRE_RAG_OLLAMA_MODEL", "local-model")
    monkeypatch.setenv("SRE_RAG_GENERATION_MAX_OUTPUT_TOKENS", "321")

    settings = Settings()

    assert settings.openai_generation_model == "test-model"
    assert settings.openai_evaluation_model == "evaluation-model"
    assert settings.ollama_model == "local-model"
    assert settings.generation_max_output_tokens == 321
