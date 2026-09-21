"""Factories and fakes shared by generation tests."""

from collections.abc import Sequence

from sre_rag.domain.generation import (
    GeneratedAnswerDraft,
    GenerationProvider,
    GenerationResult,
    GenerationUsage,
)
from sre_rag.domain.retrieval import RetrievalCandidate, RetrievalMethod, RetrievalStage
from sre_rag.generation.base import GenerationRequest
from tests.domain.factories import make_chunk


def make_reranked_candidate(
    chunk_id: str,
    text: str,
    rank: int,
) -> RetrievalCandidate:
    chunk = make_chunk().model_copy(update={"chunk_id": chunk_id, "text": text})
    return RetrievalCandidate(
        chunk=chunk,
        stages=(
            RetrievalStage(method=RetrievalMethod.RRF, score=0.02, rank=rank),
            RetrievalStage(method=RetrievalMethod.RERANKER, score=5.0, rank=rank),
        ),
    )


def make_generation_result(
    provider: GenerationProvider = GenerationProvider.OPENAI,
) -> GenerationResult:
    return GenerationResult(
        draft=GeneratedAnswerDraft(
            answerable=True,
            answer="The container exceeded its memory limit [1].",
            citation_numbers=(1,),
        ),
        provider=provider,
        model="test-model",
        response_id="response-1" if provider is GenerationProvider.OPENAI else None,
        usage=GenerationUsage(input_tokens=20, output_tokens=10),
    )


class RecordingBackend:
    def __init__(self, result: GenerationResult | Exception) -> None:
        self.result = result
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class SequenceBackend:
    """Backend that exposes whether it was invoked without provider details."""

    def __init__(self, outcomes: Sequence[GenerationResult | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        del request
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
