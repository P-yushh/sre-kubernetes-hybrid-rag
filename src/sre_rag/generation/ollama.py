"""Native Ollama API adapter for local structured generation."""

from dataclasses import dataclass
from typing import Protocol, cast

import httpx

from sre_rag.domain.generation import (
    GeneratedAnswerDraft,
    GenerationProvider,
    GenerationResult,
    GenerationUsage,
)
from sre_rag.generation.base import (
    GenerationRequest,
    InvalidGenerationOutputError,
    ProviderUnavailableError,
)


class _HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> object: ...


class _HttpClient(Protocol):
    def post(
        self,
        url: str,
        *,
        json: object,
        timeout: float,
    ) -> _HttpResponse: ...


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    """Operational settings for the native local Ollama API."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    timeout_seconds: float = 120.0
    keep_alive: str = "5m"

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama base URL must use HTTP or HTTPS")
        if not self.model.strip():
            raise ValueError("Ollama model must not be blank")
        if self.timeout_seconds <= 0:
            raise ValueError("Ollama timeout must be positive")
        if not self.keep_alive.strip():
            raise ValueError("Ollama keep-alive must not be blank")


class OllamaGenerationBackend:
    """Call ``/api/generate`` without relying on compatibility shims."""

    def __init__(
        self,
        client: _HttpClient,
        config: OllamaConfig | None = None,
        *,
        provider_errors: tuple[type[Exception], ...] = (httpx.HTTPError,),
    ) -> None:
        self._client = client
        self._config = config or OllamaConfig()
        self._provider_errors = provider_errors

    @classmethod
    def with_httpx(cls, config: OllamaConfig | None = None) -> "OllamaGenerationBackend":
        return cls(cast(_HttpClient, httpx.Client()), config)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            response = self._client.post(
                f"{self._config.base_url.rstrip('/')}/api/generate",
                json={
                    "model": self._config.model,
                    "system": request.instructions,
                    "prompt": request.prompt,
                    "format": request.output_schema,
                    "stream": False,
                    "think": False,
                    "keep_alive": self._config.keep_alive,
                    "options": {
                        "temperature": 0,
                        "num_predict": request.max_output_tokens,
                    },
                },
                timeout=self._config.timeout_seconds,
            )
            response.raise_for_status()
        except self._provider_errors as error:
            raise ProviderUnavailableError("Ollama generation request failed") from error

        try:
            payload = response.json()
        except ValueError as error:
            raise InvalidGenerationOutputError("Ollama returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise InvalidGenerationOutputError("Ollama response must be a JSON object")
        if payload.get("error"):
            raise ProviderUnavailableError("Ollama generation response failed")
        if payload.get("done") is not True:
            raise InvalidGenerationOutputError("Ollama generation did not complete")
        output_text = payload.get("response")
        if not isinstance(output_text, str) or not output_text.strip():
            raise InvalidGenerationOutputError("Ollama response contained no generated text")
        try:
            draft = GeneratedAnswerDraft.model_validate_json(output_text)
        except ValueError as error:
            raise InvalidGenerationOutputError(
                "Ollama returned invalid structured output"
            ) from error

        model = payload.get("model", self._config.model)
        return GenerationResult(
            draft=draft,
            provider=GenerationProvider.OLLAMA,
            model=model if isinstance(model, str) and model else self._config.model,
            usage=GenerationUsage(
                input_tokens=_non_negative_int(payload.get("prompt_eval_count")),
                output_tokens=_non_negative_int(payload.get("eval_count")),
            ),
        )


def _non_negative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
