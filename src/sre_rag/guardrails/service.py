"""Safe orchestration from retrieved evidence to a public query response."""

from collections.abc import Sequence
from dataclasses import dataclass

from sre_rag.domain.answers import (
    GroundedAnswer,
    QueryResponse,
    RefusalCode,
    RefusalResponse,
)
from sre_rag.domain.generation import GenerationProvider, GenerationResult, GenerationUsage
from sre_rag.domain.retrieval import RetrievalCandidate, RetrievalQuery
from sre_rag.generation.base import InvalidGenerationOutputError
from sre_rag.generation.prompting import GroundedDraftGenerator
from sre_rag.guardrails.citations import CitationValidationError, CitationValidator

_REFUSAL_MESSAGES = {
    RefusalCode.INSUFFICIENT_CONTEXT: "The retrieved sources do not contain enough evidence.",
    RefusalCode.CITATION_VALIDATION_FAILED: (
        "The generated answer could not be verified against the retrieved sources."
    ),
    RefusalCode.MODEL_OUTPUT_INVALID: "The generation provider returned an invalid response.",
}


@dataclass(frozen=True, slots=True)
class GenerationTelemetry:
    """Safe provider metadata that excludes an unvalidated model draft."""

    provider: GenerationProvider
    model: str
    response_id: str | None
    usage: GenerationUsage

    @classmethod
    def from_result(cls, result: GenerationResult) -> "GenerationTelemetry":
        return cls(
            provider=result.provider,
            model=result.model,
            response_id=result.response_id,
            usage=result.usage,
        )


@dataclass(frozen=True, slots=True)
class GuardedAnswerResult:
    """A public response plus telemetry that cannot leak an unsafe draft."""

    response: QueryResponse
    telemetry: GenerationTelemetry | None


class GroundedAnswerService:
    """Expose generated text only after evidence and citation checks pass."""

    def __init__(
        self,
        generator: GroundedDraftGenerator,
        citation_validator: CitationValidator | None = None,
    ) -> None:
        self._generator = generator
        self._citation_validator = citation_validator or CitationValidator()

    def answer(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalCandidate],
    ) -> GuardedAnswerResult:
        """Generate a grounded answer or return a stable machine-readable refusal."""

        if not candidates:
            return GuardedAnswerResult(
                response=_refusal(RefusalCode.INSUFFICIENT_CONTEXT),
                telemetry=None,
            )

        contexts = self._generator.select_contexts(candidates)
        try:
            generation = self._generator.generate(query, contexts)
        except InvalidGenerationOutputError:
            return GuardedAnswerResult(
                response=_refusal(RefusalCode.MODEL_OUTPUT_INVALID),
                telemetry=None,
            )

        telemetry = GenerationTelemetry.from_result(generation)
        if not generation.draft.answerable:
            return GuardedAnswerResult(
                response=_refusal(RefusalCode.INSUFFICIENT_CONTEXT),
                telemetry=telemetry,
            )

        try:
            answer: GroundedAnswer = self._citation_validator.validate(generation.draft, contexts)
        except CitationValidationError:
            return GuardedAnswerResult(
                response=_refusal(RefusalCode.CITATION_VALIDATION_FAILED),
                telemetry=telemetry,
            )
        return GuardedAnswerResult(response=answer, telemetry=telemetry)


def _refusal(code: RefusalCode) -> RefusalResponse:
    return RefusalResponse(code=code, message=_REFUSAL_MESSAGES[code])
