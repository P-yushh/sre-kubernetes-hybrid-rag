"""Tests for retrieval query and ranking contracts."""

from math import nan

import pytest
from pydantic import ValidationError

from sre_rag.domain.retrieval import (
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalStage,
)
from tests.domain.factories import make_chunk


def test_query_preserves_exact_search_tokens() -> None:
    text = "Why does --enable-admission-plugins affect OOMKilled pods?"

    query = RetrievalQuery(text=text)

    assert query.text == text
    assert query.query_id is not None


def test_query_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(text=" \n")


def test_candidate_keeps_stage_scores_separate() -> None:
    candidate = RetrievalCandidate(
        chunk=make_chunk(),
        stages=(
            RetrievalStage(method=RetrievalMethod.BM25, score=8.4, rank=1),
            RetrievalStage(method=RetrievalMethod.DENSE, score=0.82, rank=3),
            RetrievalStage(method=RetrievalMethod.RRF, score=0.031, rank=1),
        ),
    )

    assert [stage.method for stage in candidate.stages] == [
        RetrievalMethod.BM25,
        RetrievalMethod.DENSE,
        RetrievalMethod.RRF,
    ]


def test_candidate_rejects_duplicate_stage_methods() -> None:
    stage = RetrievalStage(method=RetrievalMethod.BM25, score=1.0, rank=1)

    with pytest.raises(ValidationError, match="methods must be unique"):
        RetrievalCandidate(chunk=make_chunk(), stages=(stage, stage))


@pytest.mark.parametrize(
    ("method", "score"),
    [(RetrievalMethod.BM25, nan), (RetrievalMethod.RRF, 0.0)],
)
def test_stage_rejects_invalid_scores(method: RetrievalMethod, score: float) -> None:
    with pytest.raises(ValidationError):
        RetrievalStage(method=method, score=score, rank=1)
