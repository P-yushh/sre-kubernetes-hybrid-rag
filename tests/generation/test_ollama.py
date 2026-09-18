"""Tests for the native Ollama generation adapter."""

import pytest

from sre_rag.domain.generation import GenerationProvider
from sre_rag.generation.base import (
    GenerationRequest,
    InvalidGenerationOutputError,
    ProviderUnavailableError,
)
from sre_rag.generation.ollama import OllamaConfig, OllamaGenerationBackend


class FakeHttpResponse:
    def __init__(self, payload: object, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    def json(self) -> object:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeHttpClient:
    def __init__(self, response: FakeHttpResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, object, float]] = []

    def post(self, url: str, *, json: object, timeout: float) -> FakeHttpResponse:
        self.calls.append((url, json, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _request() -> GenerationRequest:
    return GenerationRequest(
        query_id="query-1",
        instructions="ground only",
        prompt="question and context",
        output_schema={"type": "object"},
        max_output_tokens=400,
    )


def _payload() -> dict[str, object]:
    return {
        "model": "qwen3:8b",
        "response": '{"answer":"Use maxSurge [1].","citation_numbers":[1]}',
        "done": True,
        "prompt_eval_count": 30,
        "eval_count": 9,
    }


def test_ollama_backend_uses_native_non_streaming_structured_api() -> None:
    client = FakeHttpClient(FakeHttpResponse(_payload()))

    result = OllamaGenerationBackend(client).generate(_request())

    assert result.provider is GenerationProvider.OLLAMA
    assert result.model == "qwen3:8b"
    assert result.usage.input_tokens == 30
    url, body, timeout = client.calls[0]
    assert url == "http://localhost:11434/api/generate"
    assert timeout == 120.0
    assert isinstance(body, dict)
    assert body["stream"] is False
    assert body["think"] is False
    assert body["format"] == {"type": "object"}
    assert body["options"] == {"temperature": 0, "num_predict": 400}


def test_ollama_transport_failure_is_translated_for_fallback() -> None:
    backend = OllamaGenerationBackend(
        FakeHttpClient(RuntimeError("offline")),
        provider_errors=(RuntimeError,),
    )

    with pytest.raises(ProviderUnavailableError, match="Ollama"):
        backend.generate(_request())


@pytest.mark.parametrize(
    "payload",
    [
        ValueError("invalid JSON"),
        [],
        {},
        {"done": True},
        {"response": "not-json", "done": True},
        {"response": '{"answer":"missing citations"}', "done": True},
        {"response": '{"answer":"answer [1]","citation_numbers":[1]}', "done": False},
    ],
)
def test_ollama_rejects_invalid_responses(payload: object) -> None:
    with pytest.raises(InvalidGenerationOutputError):
        OllamaGenerationBackend(FakeHttpClient(FakeHttpResponse(payload))).generate(_request())


def test_ollama_http_status_failure_is_translated() -> None:
    response = FakeHttpResponse(_payload(), RuntimeError("500"))
    backend = OllamaGenerationBackend(
        FakeHttpClient(response),
        provider_errors=(RuntimeError,),
    )

    with pytest.raises(ProviderUnavailableError):
        backend.generate(_request())


def test_ollama_error_payload_is_operational_failure() -> None:
    with pytest.raises(ProviderUnavailableError):
        OllamaGenerationBackend(
            FakeHttpClient(FakeHttpResponse({"error": "model unavailable", "done": True}))
        ).generate(_request())


def test_invalid_usage_and_model_metadata_use_safe_defaults() -> None:
    payload = _payload()
    payload.update({"model": "", "prompt_eval_count": True, "eval_count": -1})

    result = OllamaGenerationBackend(FakeHttpClient(FakeHttpResponse(payload))).generate(_request())

    assert result.model == "qwen3:8b"
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_url": "localhost:11434"},
        {"model": " "},
        {"timeout_seconds": 0},
        {"keep_alive": " "},
    ],
)
def test_ollama_configuration_is_validated(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        OllamaConfig(**kwargs)  # type: ignore[arg-type]
