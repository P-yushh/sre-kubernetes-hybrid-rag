"""Tests for provider-neutral generation contracts."""

import pytest
from pydantic import ValidationError

from sre_rag.domain.generation import GeneratedAnswerDraft, GenerationUsage


def test_generation_draft_preserves_answer_and_citations() -> None:
    draft = GeneratedAnswerDraft(
        answerable=True,
        answer="Use the referenced setting [2].",
        citation_numbers=(2,),
    )

    assert draft.citation_numbers == (2,)


@pytest.mark.parametrize("citations", [(), (0,), (1, 1)])
def test_generation_draft_rejects_invalid_citations(citations: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError):
        GeneratedAnswerDraft(answerable=True, answer="answer", citation_numbers=citations)


def test_unanswerable_draft_cannot_claim_citations() -> None:
    with pytest.raises(ValidationError, match="cannot cite"):
        GeneratedAnswerDraft(
            answerable=False,
            answer="The supplied context is insufficient.",
            citation_numbers=(1,),
        )


def test_generation_usage_rejects_negative_tokens() -> None:
    with pytest.raises(ValidationError):
        GenerationUsage(input_tokens=-1, output_tokens=0)
