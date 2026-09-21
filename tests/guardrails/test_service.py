"""Tests for safe answer assembly and deterministic refusal codes."""

import pytest

from sre_rag.domain.answers import GroundedAnswer, RefusalCode, RefusalResponse
from sre_rag.domain.generation import GeneratedAnswerDraft
from sre_rag.domain.retrieval import RetrievalCandidate, RetrievalQuery
from sre_rag.generation.base import InvalidGenerationOutputError, ProviderUnavailableError
from sre_rag.generation.prompting import GroundedDraftGenerator, GroundedGenerationConfig
from sre_rag.guardrails.service import GroundedAnswerService
from tests.generation.helpers import (
    RecordingBackend,
    make_generation_result,
    make_reranked_candidate,
)


def _candidate(rank: int = 1) -> RetrievalCandidate:
    return make_reranked_candidate(f"chunk-{rank}", f"evidence {rank}", rank)


def _service(backend: RecordingBackend, *, max_contexts: int = 5) -> GroundedAnswerService:
    generator = GroundedDraftGenerator(
        backend,
        GroundedGenerationConfig(max_contexts=max_contexts),
    )
    return GroundedAnswerService(generator)


def test_service_returns_validated_answer_with_generation_telemetry() -> None:
    generation = make_generation_result()
    backend = RecordingBackend(generation)

    result = _service(backend).answer(RetrievalQuery(text="Why OOMKilled?"), (_candidate(),))

    assert isinstance(result.response, GroundedAnswer)
    assert result.response.citations[0].chunk_id == "chunk-1"
    assert result.telemetry is not None
    assert result.telemetry.provider is generation.provider
    assert result.telemetry.usage == generation.usage


def test_empty_evidence_returns_insufficient_context_without_generation() -> None:
    backend = RecordingBackend(make_generation_result())

    result = _service(backend).answer(RetrievalQuery(text="question"), ())

    assert isinstance(result.response, RefusalResponse)
    assert result.response.code is RefusalCode.INSUFFICIENT_CONTEXT
    assert result.telemetry is None
    assert backend.requests == []


def test_model_can_declare_context_insufficient() -> None:
    generation = make_generation_result().model_copy(
        update={
            "draft": GeneratedAnswerDraft(
                answerable=False,
                answer="The supplied evidence does not answer the question.",
            )
        }
    )

    result = _service(RecordingBackend(generation)).answer(
        RetrievalQuery(text="question"), (_candidate(),)
    )

    assert isinstance(result.response, RefusalResponse)
    assert result.response.code is RefusalCode.INSUFFICIENT_CONTEXT
    assert result.telemetry is not None
    assert result.telemetry.provider is generation.provider


def test_invalid_provider_output_returns_model_output_refusal() -> None:
    result = _service(RecordingBackend(InvalidGenerationOutputError("invalid"))).answer(
        RetrievalQuery(text="question"), (_candidate(),)
    )

    assert isinstance(result.response, RefusalResponse)
    assert result.response.code is RefusalCode.MODEL_OUTPUT_INVALID
    assert result.telemetry is None


@pytest.mark.parametrize(
    ("answer", "citations"),
    [
        ("Marker and declaration disagree [2].", (1,)),
        ("Citation exceeds the bounded evidence [3].", (3,)),
    ],
)
def test_invalid_citations_return_citation_refusal(answer: str, citations: tuple[int, ...]) -> None:
    generation = make_generation_result().model_copy(
        update={
            "draft": GeneratedAnswerDraft(
                answerable=True,
                answer=answer,
                citation_numbers=citations,
            )
        }
    )

    result = _service(RecordingBackend(generation), max_contexts=2).answer(
        RetrievalQuery(text="question"), (_candidate(1), _candidate(2), _candidate(3))
    )

    assert isinstance(result.response, RefusalResponse)
    assert result.response.code is RefusalCode.CITATION_VALIDATION_FAILED
    assert result.telemetry is not None
    assert not hasattr(result.telemetry, "draft")


def test_operational_provider_failure_remains_an_availability_error() -> None:
    service = _service(RecordingBackend(ProviderUnavailableError("offline")))

    with pytest.raises(ProviderUnavailableError, match="offline"):
        service.answer(RetrievalQuery(text="question"), (_candidate(),))
