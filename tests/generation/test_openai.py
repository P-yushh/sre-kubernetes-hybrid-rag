"""Tests for the OpenAI Responses API adapter without network calls."""

from types import SimpleNamespace
from typing import Any

import pytest

from sre_rag.domain.generation import GenerationProvider
from sre_rag.generation.base import (
    GenerationRequest,
    InvalidGenerationOutputError,
    ProviderUnavailableError,
)
from sre_rag.generation.openai import OpenAIConfig, OpenAIGenerationBackend


class FakeResponses:
    def __init__(self, response: object | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeOpenAIClient:
    def __init__(self, response: object | Exception) -> None:
        self.responses = FakeResponses(response)


def _request() -> GenerationRequest:
    return GenerationRequest(
        query_id="query-1",
        instructions="ground only",
        prompt="question and context",
        output_schema={"type": "object"},
        max_output_tokens=500,
    )


def _response(**overrides: Any) -> SimpleNamespace:
    values = {
        "id": "resp-1",
        "model": "gpt-5.6-luna",
        "status": "completed",
        "output_text": (
            '{"answer":"The pod exceeded its memory limit [1].","citation_numbers":[1]}'
        ),
        "usage": SimpleNamespace(input_tokens=40, output_tokens=12),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_openai_backend_uses_responses_api_and_parses_usage() -> None:
    client = FakeOpenAIClient(_response())

    result = OpenAIGenerationBackend(client).generate(_request())

    assert result.provider is GenerationProvider.OPENAI
    assert result.response_id == "resp-1"
    assert result.usage.input_tokens == 40
    assert result.draft.citation_numbers == (1,)
    call = client.responses.calls[0]
    assert call["store"] is False
    assert call["temperature"] == 0.0
    assert call["max_output_tokens"] == 500
    assert call["metadata"] == {"query_id": "query-1"}
    assert call["text"] == {
        "format": {
            "type": "json_schema",
            "name": "grounded_answer_draft",
            "schema": {"type": "object"},
            "strict": True,
        }
    }


@pytest.mark.parametrize("output", [None, "", "not-json", '{"answer":"missing citations"}'])
def test_openai_backend_rejects_invalid_output(output: object) -> None:
    with pytest.raises(InvalidGenerationOutputError):
        OpenAIGenerationBackend(FakeOpenAIClient(_response(output_text=output))).generate(
            _request()
        )


def test_openai_provider_failure_is_translated_for_fallback() -> None:
    backend = OpenAIGenerationBackend(
        FakeOpenAIClient(RuntimeError("offline")),
        provider_errors=(RuntimeError,),
    )

    with pytest.raises(ProviderUnavailableError, match="OpenAI"):
        backend.generate(_request())


def test_failed_and_incomplete_responses_are_classified() -> None:
    with pytest.raises(ProviderUnavailableError):
        OpenAIGenerationBackend(FakeOpenAIClient(_response(status="failed"))).generate(_request())
    with pytest.raises(InvalidGenerationOutputError, match="in_progress"):
        OpenAIGenerationBackend(FakeOpenAIClient(_response(status="in_progress"))).generate(
            _request()
        )


def test_invalid_usage_metadata_defaults_to_zero() -> None:
    result = OpenAIGenerationBackend(
        FakeOpenAIClient(
            _response(
                id="",
                model=None,
                usage=SimpleNamespace(input_tokens=True, output_tokens=-1),
            )
        )
    ).generate(_request())

    assert result.response_id is None
    assert result.model == "gpt-5.6-luna"
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


def test_openai_configuration_and_key_are_validated() -> None:
    with pytest.raises(ValueError, match="model"):
        OpenAIConfig(model=" ")
    with pytest.raises(ValueError, match="timeout"):
        OpenAIConfig(timeout_seconds=0)
    with pytest.raises(ValueError, match="retries"):
        OpenAIConfig(max_retries=-1)
    with pytest.raises(ValueError, match="temperature"):
        OpenAIConfig(temperature=2.1)
    with pytest.raises(ValueError, match="API key"):
        OpenAIGenerationBackend.from_api_key(" ")
