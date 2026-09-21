"""Tests for independent citation validation and source resolution."""

import pytest

from sre_rag.domain.generation import GeneratedAnswerDraft
from sre_rag.domain.retrieval import RetrievalCandidate
from sre_rag.guardrails.citations import CitationValidationError, CitationValidator
from tests.generation.helpers import make_reranked_candidate


def _contexts() -> tuple[RetrievalCandidate, ...]:
    return (
        make_reranked_candidate("first", "Probe configuration.", 1),
        make_reranked_candidate("second", "OOMKilled maps to exit code 137.", 2),
    )


def test_validator_resolves_context_number_to_source_metadata() -> None:
    draft = GeneratedAnswerDraft(
        answerable=True,
        answer="Exit code 137 can accompany OOMKilled [2].",
        citation_numbers=(2,),
    )

    answer = CitationValidator().validate(draft, _contexts())

    assert answer.answer == draft.answer
    assert len(answer.citations) == 1
    assert answer.citations[0].number == 2
    assert answer.citations[0].chunk_id == "second"
    assert answer.citations[0].source_title == "Debug Pods > Exit codes"


@pytest.mark.parametrize(
    "draft",
    [
        GeneratedAnswerDraft(
            answerable=True,
            answer="The answer cites a marker not declared by the model [2].",
            citation_numbers=(1,),
        ),
        GeneratedAnswerDraft(
            answerable=True,
            answer="The answer cites a context that was never supplied [3].",
            citation_numbers=(3,),
        ),
    ],
)
def test_validator_rejects_unverifiable_citations(draft: GeneratedAnswerDraft) -> None:
    with pytest.raises(CitationValidationError):
        CitationValidator().validate(draft, _contexts())


def test_validator_rejects_unanswerable_draft() -> None:
    draft = GeneratedAnswerDraft(
        answerable=False,
        answer="The evidence is insufficient.",
    )

    with pytest.raises(CitationValidationError, match="unanswerable"):
        CitationValidator().validate(draft, _contexts())


def test_citation_title_falls_back_to_source_filename() -> None:
    context = _contexts()[0]
    root_chunk = context.chunk.model_copy(update={"heading_path": ()})
    root_context = context.model_copy(update={"chunk": root_chunk})
    draft = GeneratedAnswerDraft(
        answerable=True,
        answer="The root context supports this answer [1].",
        citation_numbers=(1,),
    )

    answer = CitationValidator().validate(draft, (root_context,))

    assert answer.citations[0].source_title == "debug-application.md"
