"""Provider-neutral contracts for structured grounded-generation drafts."""

from enum import StrEnum

from pydantic import Field, field_validator

from sre_rag.domain.base import DomainModel, NonEmptyText


class GenerationProvider(StrEnum):
    """Generation backends supported by the service."""

    OPENAI = "openai"
    OLLAMA = "ollama"


class GeneratedAnswerDraft(DomainModel):
    """Structured model output awaiting independent citation validation."""

    answer: NonEmptyText
    citation_numbers: tuple[int, ...] = Field(min_length=1)

    @field_validator("citation_numbers")
    @classmethod
    def citations_must_be_positive_and_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(number <= 0 for number in value):
            raise ValueError("draft citation numbers must be positive")
        if len(value) != len(set(value)):
            raise ValueError("draft citation numbers must be unique")
        return value


class GenerationUsage(DomainModel):
    """Provider-reported tokens used by one generation request."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class GenerationResult(DomainModel):
    """A structured draft plus metadata needed for telemetry and cost accounting."""

    draft: GeneratedAnswerDraft
    provider: GenerationProvider
    model: NonEmptyText
    response_id: str | None = None
    usage: GenerationUsage
