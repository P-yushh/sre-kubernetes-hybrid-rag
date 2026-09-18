"""Provider-independent generation interfaces and failure taxonomy."""

from dataclasses import dataclass
from typing import Protocol

from sre_rag.domain.generation import GenerationResult


class GenerationError(RuntimeError):
    """Base failure raised by the generation layer."""


class ProviderUnavailableError(GenerationError):
    """A provider could not complete a request due to an operational failure."""


class AllProvidersUnavailableError(ProviderUnavailableError):
    """Neither the primary provider nor its fallback was available."""


class InvalidGenerationOutputError(GenerationError):
    """A provider completed but returned invalid structured output."""


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Provider-neutral prompt and schema sent to a model backend."""

    query_id: str
    instructions: str
    prompt: str
    output_schema: dict[str, object]
    max_output_tokens: int


class GenerationBackend(Protocol):
    """Generate one structured answer draft."""

    def generate(self, request: GenerationRequest) -> GenerationResult: ...
