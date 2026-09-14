"""Tests for deterministic Reciprocal Rank Fusion."""

import pytest

from sre_rag.domain.documents import Chunk
from sre_rag.domain.retrieval import (
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalStage,
)
from sre_rag.retrieval.fusion import ReciprocalRankFusion, RRFConfig
from tests.domain.factories import make_chunk


def _chunk(chunk_id: str, text: str | None = None) -> Chunk:
    updates: dict[str, object] = {"chunk_id": chunk_id}
    if text is not None:
        updates["text"] = text
    return make_chunk().model_copy(update=updates)


def _candidate(
    chunk: Chunk,
    method: RetrievalMethod,
    rank: int,
    *,
    score: float = 1.0,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=chunk,
        stages=(RetrievalStage(method=method, score=score, rank=rank),),
    )


def _stage(candidate: RetrievalCandidate, method: RetrievalMethod) -> RetrievalStage:
    return next(stage for stage in candidate.stages if stage.method is method)


def test_overlapping_candidate_receives_both_rank_contributions() -> None:
    shared = _chunk("shared")
    sparse_only = _chunk("sparse-only")
    dense_only = _chunk("dense-only")
    rankings = {
        RetrievalMethod.BM25: (
            _candidate(sparse_only, RetrievalMethod.BM25, 1, score=100.0),
            _candidate(shared, RetrievalMethod.BM25, 2, score=2.0),
        ),
        RetrievalMethod.DENSE: (
            _candidate(shared, RetrievalMethod.DENSE, 1, score=0.82),
            _candidate(dense_only, RetrievalMethod.DENSE, 2, score=0.81),
        ),
    }

    results = ReciprocalRankFusion().fuse(rankings)

    assert [candidate.chunk.chunk_id for candidate in results] == [
        "shared",
        "sparse-only",
        "dense-only",
    ]
    assert _stage(results[0], RetrievalMethod.RRF).score == pytest.approx(1 / 62 + 1 / 61)
    assert [stage.method for stage in results[0].stages] == [
        RetrievalMethod.BM25,
        RetrievalMethod.DENSE,
        RetrievalMethod.RRF,
    ]


def test_raw_retrieval_scores_do_not_affect_fusion() -> None:
    first = _candidate(_chunk("first"), RetrievalMethod.BM25, 1, score=-1000.0)
    second = _candidate(_chunk("second"), RetrievalMethod.BM25, 2, score=1_000_000.0)

    results = ReciprocalRankFusion(RRFConfig(rank_constant=10)).fuse(
        {RetrievalMethod.BM25: (first, second)}
    )

    assert [candidate.chunk.chunk_id for candidate in results] == ["first", "second"]
    assert _stage(results[0], RetrievalMethod.RRF).score == pytest.approx(1 / 11)
    assert _stage(results[1], RetrievalMethod.RRF).score == pytest.approx(1 / 12)


def test_single_source_and_empty_rankings_are_supported() -> None:
    candidate = _candidate(_chunk("sparse"), RetrievalMethod.BM25, 1)
    fusion = ReciprocalRankFusion()

    assert fusion.fuse({}) == ()
    assert fusion.fuse({RetrievalMethod.BM25: ()}) == ()
    assert fusion.fuse({RetrievalMethod.BM25: (candidate,)})[0].chunk.chunk_id == "sparse"


def test_result_limit_is_applied_after_global_fusion() -> None:
    rankings = {
        RetrievalMethod.BM25: tuple(
            _candidate(_chunk(f"chunk-{rank}"), RetrievalMethod.BM25, rank) for rank in range(1, 4)
        )
    }

    results = ReciprocalRankFusion().fuse(rankings, limit=2)

    assert [candidate.chunk.chunk_id for candidate in results] == ["chunk-1", "chunk-2"]
    assert [_stage(candidate, RetrievalMethod.RRF).rank for candidate in results] == [1, 2]


def test_equal_fusion_scores_use_chunk_id_as_tie_breaker() -> None:
    rankings = {
        RetrievalMethod.BM25: (_candidate(_chunk("z-chunk"), RetrievalMethod.BM25, 1),),
        RetrievalMethod.DENSE: (_candidate(_chunk("a-chunk"), RetrievalMethod.DENSE, 1),),
    }

    results = ReciprocalRankFusion().fuse(rankings)

    assert [candidate.chunk.chunk_id for candidate in results] == ["a-chunk", "z-chunk"]


@pytest.mark.parametrize(
    "config",
    [RRFConfig(rank_constant=1, default_limit=1)],
)
def test_configuration_is_exposed(config: RRFConfig) -> None:
    candidate = _candidate(_chunk("chunk"), RetrievalMethod.BM25, 1)

    result = ReciprocalRankFusion(config).fuse({RetrievalMethod.BM25: (candidate,)})[0]

    assert _stage(result, RetrievalMethod.RRF).score == pytest.approx(0.5)


def test_invalid_configuration_and_limit_are_rejected() -> None:
    with pytest.raises(ValueError, match="rank constant"):
        RRFConfig(rank_constant=0)
    with pytest.raises(ValueError, match="default limit"):
        RRFConfig(default_limit=0)
    with pytest.raises(ValueError, match="fusion limit"):
        ReciprocalRankFusion().fuse({}, limit=0)


def test_unsupported_source_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported.*reranker"):
        ReciprocalRankFusion().fuse({RetrievalMethod.RERANKER: ()})


def test_candidate_must_have_exactly_its_declared_source_stage() -> None:
    chunk = _chunk("chunk")
    wrong_method = _candidate(chunk, RetrievalMethod.DENSE, 1)
    extra_stage = RetrievalCandidate(
        chunk=chunk,
        stages=(
            RetrievalStage(method=RetrievalMethod.BM25, score=1.0, rank=1),
            RetrievalStage(method=RetrievalMethod.DENSE, score=0.8, rank=1),
        ),
    )

    with pytest.raises(ValueError, match="only its source stage"):
        ReciprocalRankFusion().fuse({RetrievalMethod.BM25: (wrong_method,)})
    with pytest.raises(ValueError, match="only its source stage"):
        ReciprocalRankFusion().fuse({RetrievalMethod.BM25: (extra_stage,)})


def test_duplicate_chunk_or_rank_within_source_is_rejected() -> None:
    chunk = _chunk("chunk")
    other = _chunk("other")
    fusion = ReciprocalRankFusion()

    with pytest.raises(ValueError, match="duplicate chunk ID"):
        fusion.fuse(
            {
                RetrievalMethod.BM25: (
                    _candidate(chunk, RetrievalMethod.BM25, 1),
                    _candidate(chunk, RetrievalMethod.BM25, 2),
                )
            }
        )
    with pytest.raises(ValueError, match="duplicate rank"):
        fusion.fuse(
            {
                RetrievalMethod.DENSE: (
                    _candidate(chunk, RetrievalMethod.DENSE, 1),
                    _candidate(other, RetrievalMethod.DENSE, 1),
                )
            }
        )


def test_same_chunk_id_with_inconsistent_payload_is_rejected() -> None:
    sparse = _candidate(_chunk("shared", "sparse payload"), RetrievalMethod.BM25, 1)
    dense = _candidate(_chunk("shared", "dense payload"), RetrievalMethod.DENSE, 1)

    with pytest.raises(ValueError, match="inconsistent payloads"):
        ReciprocalRankFusion().fuse(
            {
                RetrievalMethod.BM25: (sparse,),
                RetrievalMethod.DENSE: (dense,),
            }
        )
