"""OpenAI Responses API adapter for structured grounded generation."""

from dataclasses import dataclass
from typing import Protocol, cast

from openai import OpenAI, OpenAIError

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


class _ResponsesResource(Protocol):
    def create(self, **kwargs: object) -> object: ...


class _OpenAIClient(Protocol):
    @property
    def responses(self) -> _ResponsesResource: ...


@dataclass(frozen=True, slots=True)
class OpenAIConfig:
    """Operational settings for hosted generation."""

    model: str = "gpt-5.6-luna"
    timeout_seconds: float = 30.0
    max_retries: int = 2
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("OpenAI model must not be blank")
        if self.timeout_seconds <= 0:
            raise ValueError("OpenAI timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("OpenAI retries must be non-negative")
        if not 0 <= self.temperature <= 2:
            raise ValueError("OpenAI temperature must be between zero and two")


class OpenAIGenerationBackend:
    """Call the Responses API and validate its JSON-schema output."""

    def __init__(
        self,
        client: _OpenAIClient,
        config: OpenAIConfig | None = None,
        *,
        provider_errors: tuple[type[Exception], ...] = (OpenAIError,),
    ) -> None:
        self._client = client
        self._config = config or OpenAIConfig()
        self._provider_errors = provider_errors

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        config: OpenAIConfig | None = None,
    ) -> "OpenAIGenerationBackend":
        """Create a bounded synchronous SDK client from an explicit secret."""

        if not api_key.strip():
            raise ValueError("OpenAI API key must not be blank")
        resolved = config or OpenAIConfig()
        client = OpenAI(
            api_key=api_key,
            timeout=resolved.timeout_seconds,
            max_retries=resolved.max_retries,
        )
        return cls(cast(_OpenAIClient, client), resolved)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            response = self._client.responses.create(
                model=self._config.model,
                instructions=request.instructions,
                input=request.prompt,
                max_output_tokens=request.max_output_tokens,
                metadata={"query_id": request.query_id},
                store=False,
                temperature=self._config.temperature,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "grounded_answer_draft",
                        "schema": request.output_schema,
                        "strict": True,
                    }
                },
            )
        except self._provider_errors as error:
            raise ProviderUnavailableError("OpenAI generation request failed") from error

        status = getattr(response, "status", "completed")
        if status == "failed":
            raise ProviderUnavailableError("OpenAI generation response failed")
        if status != "completed":
            raise InvalidGenerationOutputError(
                f"OpenAI generation did not complete successfully: {status}"
            )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise InvalidGenerationOutputError("OpenAI response contained no output text")
        try:
            draft = GeneratedAnswerDraft.model_validate_json(output_text)
        except ValueError as error:
            raise InvalidGenerationOutputError(
                "OpenAI returned invalid structured output"
            ) from error

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        return GenerationResult(
            draft=draft,
            provider=GenerationProvider.OPENAI,
            model=_non_empty_string(getattr(response, "model", None), self._config.model),
            response_id=_optional_string(getattr(response, "id", None)),
            usage=GenerationUsage(
                input_tokens=_non_negative_int(input_tokens),
                output_tokens=_non_negative_int(output_tokens),
            ),
        )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _non_empty_string(value: object, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _non_negative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
