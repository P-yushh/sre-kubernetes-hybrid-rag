"""Tests for grounded-answer and refusal contracts."""

import pytest
from pydantic import TypeAdapter, ValidationError

from sre_rag.domain.answers import (
    Citation,
    GroundedAnswer,
    QueryResponse,
    RefusalCode,
    RefusalResponse,
)
from tests.domain.factories import make_chunk


def make_citation(number: int = 1) -> Citation:
    chunk = make_chunk()
    return Citation(
        number=number,
        chunk_id=chunk.chunk_id,
        source_title="Debug Applications",
        source_path=chunk.provenance.relative_path,
        source_url=chunk.provenance.source_url,
    )


def test_grounded_answer_maps_inline_markers_to_citations() -> None:
    response = GroundedAnswer(
        answer="Exit code 137 commonly indicates a forced termination [1].",
        citations=(make_citation(),),
    )

    assert response.outcome == "answer"
    assert response.citations[0].chunk_id == make_chunk().chunk_id


@pytest.mark.parametrize(
    ("answer", "citations", "message"),
    [
        ("There is no inline marker.", (make_citation(),), "markers must match"),
        ("Only the first source is used [1].", (make_citation(), make_citation()), "unique"),
        ("Citation record does not match marker [2].", (make_citation(1),), "markers must match"),
    ],
)
def test_grounded_answer_rejects_invalid_citation_mappings(
    answer: str, citations: tuple[Citation, ...], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        GroundedAnswer(answer=answer, citations=citations)


def test_grounded_answer_allows_citing_only_the_second_context() -> None:
    response = GroundedAnswer(
        answer="The second context contains the relevant evidence [2].",
        citations=(make_citation(2),),
    )

    assert response.citations[0].number == 2


def test_query_response_is_a_discriminated_union() -> None:
    adapter: TypeAdapter[QueryResponse] = TypeAdapter(QueryResponse)
    refusal = adapter.validate_python(
        {
            "outcome": "refusal",
            "code": "INSUFFICIENT_CONTEXT",
            "message": "The indexed sources do not contain enough evidence.",
        }
    )
    schema = adapter.json_schema()

    assert isinstance(refusal, RefusalResponse)
    assert refusal.code is RefusalCode.INSUFFICIENT_CONTEXT
    assert schema["discriminator"]["propertyName"] == "outcome"


def test_response_cannot_mix_answer_and_refusal_fields() -> None:
    adapter: TypeAdapter[QueryResponse] = TypeAdapter(QueryResponse)

    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "outcome": "answer",
                "answer": "Supported statement [1].",
                "citations": [make_citation().model_dump()],
                "code": "INSUFFICIENT_CONTEXT",
                "message": "Conflicting refusal",
            }
        )
