"""Provider-neutral contracts for structured grounded-generation drafts."""

from enum import StrEnum

from pydantic import Field, model_validator

from sre_rag.domain.base import DomainModel, NonEmptyText


class GenerationProvider(StrEnum):
    """Generation backends supported by the service."""

    OPENAI = "openai"
    OLLAMA = "ollama"


class GeneratedAnswerDraft(DomainModel):
    """Structured model output awaiting independent citation validation."""

    answerable: bool
    answer: NonEmptyText
    citation_numbers: tuple[int, ...] = ()

    @model_validator(mode="after")
    def citations_must_match_answerability(self) -> "GeneratedAnswerDraft":
        if any(number <= 0 for number in self.citation_numbers):
            raise ValueError("draft citation numbers must be positive")
        if len(self.citation_numbers) != len(set(self.citation_numbers)):
            raise ValueError("draft citation numbers must be unique")
        if self.answerable and not self.citation_numbers:
            raise ValueError("answerable drafts must cite at least one context")
        if not self.answerable and self.citation_numbers:
            raise ValueError("unanswerable drafts cannot cite contexts")
        return self


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
