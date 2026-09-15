"""Tests for cross-encoder reranking without downloading model weights."""

from typing import Any

import numpy as np
import pytest

from sre_rag.domain.documents import Chunk
from sre_rag.domain.retrieval import (
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalStage,
)
from sre_rag.retrieval.reranking import (
    BGE_RERANKER_MODEL,
    BGE_RERANKER_REVISION,
    MINILM_RERANKER_MODEL,
    MINILM_RERANKER_REVISION,
    CrossEncoderConfig,
    CrossEncoderReranker,
)
from tests.domain.factories import make_chunk


class FakeCrossEncoder:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

    def predict(self, inputs: list[tuple[str, str]], **kwargs: Any) -> object:
        self.calls.append((inputs, kwargs))
        return self.output


def _chunk(chunk_id: str, text: str) -> Chunk:
    return make_chunk().model_copy(update={"chunk_id": chunk_id, "text": text})


def _fused_candidate(chunk: Chunk, rank: int, *, rrf_score: float = 0.02) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=chunk,
        stages=(
            RetrievalStage(method=RetrievalMethod.BM25, score=2.0, rank=rank),
            RetrievalStage(method=RetrievalMethod.RRF, score=rrf_score, rank=rank),
        ),
    )


def _reranker_stage(candidate: RetrievalCandidate) -> RetrievalStage:
    return next(stage for stage in candidate.stages if stage.method is RetrievalMethod.RERANKER)


def test_reranker_jointly_scores_query_chunk_pairs_and_reorders_candidates() -> None:
    first = _fused_candidate(_chunk("first", "A generic container restart."), 1)
    second = _fused_candidate(
        _chunk("second", "OOMKilled commonly corresponds to exit code 137."),
        2,
    )
    backend = FakeCrossEncoder(np.array([-2.0, 4.5]))
    query = RetrievalQuery(text="Why was my pod terminated with code 137?")

    results = CrossEncoderReranker(backend).rerank(query, (first, second))

    assert [candidate.chunk.chunk_id for candidate in results] == ["second", "first"]
    assert [stage.method for stage in results[0].stages] == [
        RetrievalMethod.BM25,
        RetrievalMethod.RRF,
        RetrievalMethod.RERANKER,
    ]
    assert _reranker_stage(results[0]).score == pytest.approx(4.5)
    assert [_reranker_stage(candidate).rank for candidate in results] == [1, 2]
    pairs, options = backend.calls[0]
    assert pairs == [(query.text, first.chunk.text), (query.text, second.chunk.text)]
    assert options == {
        "batch_size": 8,
        "show_progress_bar": False,
        "convert_to_numpy": True,
        "apply_softmax": False,
    }


def test_limit_is_applied_after_all_candidates_are_scored() -> None:
    candidates = tuple(
        _fused_candidate(_chunk(f"chunk-{index}", f"passage {index}"), index + 1)
        for index in range(3)
    )
    backend = FakeCrossEncoder(np.array([0.1, 0.2, 0.9]))

    results = CrossEncoderReranker(backend).rerank(
        RetrievalQuery(text="query"),
        candidates,
        limit=1,
    )

    assert [candidate.chunk.chunk_id for candidate in results] == ["chunk-2"]
    assert len(backend.calls[0][0]) == 3


def test_equal_scores_use_chunk_id_as_deterministic_tie_breaker() -> None:
    candidates = (
        _fused_candidate(_chunk("z-chunk", "z"), 1),
        _fused_candidate(_chunk("a-chunk", "a"), 2),
    )

    results = CrossEncoderReranker(FakeCrossEncoder(np.array([0.5, 0.5]))).rerank(
        RetrievalQuery(text="query"), candidates
    )

    assert [candidate.chunk.chunk_id for candidate in results] == ["a-chunk", "z-chunk"]


def test_column_vector_output_is_flattened() -> None:
    candidate = _fused_candidate(_chunk("chunk", "passage"), 1)

    result = CrossEncoderReranker(FakeCrossEncoder(np.array([[3.2]]))).rerank(
        RetrievalQuery(text="query"), (candidate,)
    )[0]

    assert _reranker_stage(result).score == pytest.approx(3.2)


def test_empty_candidates_do_not_invoke_model() -> None:
    backend = FakeCrossEncoder(np.empty(0))

    assert CrossEncoderReranker(backend).rerank(RetrievalQuery(text="query"), ()) == ()
    assert backend.calls == []


@pytest.mark.parametrize(
    "output",
    [np.array([1.0]), np.array([1.0, float("nan")]), np.array([[1.0, 2.0]])],
)
def test_invalid_model_output_is_rejected(output: object) -> None:
    candidates = (
        _fused_candidate(_chunk("first", "first"), 1),
        _fused_candidate(_chunk("second", "second"), 2),
    )

    with pytest.raises(ValueError, match="output shape|non-finite"):
        CrossEncoderReranker(FakeCrossEncoder(output)).rerank(
            RetrievalQuery(text="query"), candidates
        )


def test_candidates_require_rrf_and_must_not_be_reranked_twice() -> None:
    chunk = _chunk("chunk", "passage")
    missing_rrf = RetrievalCandidate(
        chunk=chunk,
        stages=(RetrievalStage(method=RetrievalMethod.BM25, score=1.0, rank=1),),
    )
    already_reranked = RetrievalCandidate(
        chunk=chunk,
        stages=(
            RetrievalStage(method=RetrievalMethod.RRF, score=0.1, rank=1),
            RetrievalStage(method=RetrievalMethod.RERANKER, score=1.0, rank=1),
        ),
    )
    reranker = CrossEncoderReranker(FakeCrossEncoder(np.array([1.0])))

    with pytest.raises(ValueError, match="must contain an RRF"):
        reranker.rerank(RetrievalQuery(text="query"), (missing_rrf,))
    with pytest.raises(ValueError, match="already been reranked"):
        reranker.rerank(RetrievalQuery(text="query"), (already_reranked,))


def test_duplicate_candidate_ids_are_rejected() -> None:
    candidate = _fused_candidate(_chunk("chunk", "passage"), 1)

    with pytest.raises(ValueError, match="duplicate reranker chunk ID"):
        CrossEncoderReranker(FakeCrossEncoder(np.array([1.0, 1.0]))).rerank(
            RetrievalQuery(text="query"), (candidate, candidate)
        )


def test_duplicate_rrf_ranks_are_rejected() -> None:
    candidates = (
        _fused_candidate(_chunk("first", "first"), 1),
        _fused_candidate(_chunk("second", "second"), 1),
    )

    with pytest.raises(ValueError, match="duplicate RRF rank"):
        CrossEncoderReranker(FakeCrossEncoder(np.array([1.0, 0.0]))).rerank(
            RetrievalQuery(text="query"), candidates
        )


def test_candidate_and_result_limits_are_enforced() -> None:
    config = CrossEncoderConfig(max_candidates=1, default_limit=1)
    reranker = CrossEncoderReranker(FakeCrossEncoder(np.array([1.0, 0.0])), config)
    candidates = (
        _fused_candidate(_chunk("first", "first"), 1),
        _fused_candidate(_chunk("second", "second"), 2),
    )

    with pytest.raises(ValueError, match="received 2 candidates"):
        reranker.rerank(RetrievalQuery(text="query"), candidates)
    with pytest.raises(ValueError, match="within candidate limit"):
        reranker.rerank(RetrievalQuery(text="query"), (), limit=2)


def test_model_presets_are_pinned_and_use_conservative_batches() -> None:
    minilm = CrossEncoderConfig.minilm()
    bge = CrossEncoderConfig.bge()

    assert (minilm.model_name, minilm.revision, minilm.batch_size) == (
        MINILM_RERANKER_MODEL,
        MINILM_RERANKER_REVISION,
        8,
    )
    assert (bge.model_name, bge.revision, bge.batch_size) == (
        BGE_RERANKER_MODEL,
        BGE_RERANKER_REVISION,
        4,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name": " "},
        {"revision": " "},
        {"batch_size": 0},
        {"max_length": 0},
        {"max_candidates": 0},
        {"default_limit": 0},
        {"max_candidates": 2, "default_limit": 3},
        {"device": "cuda"},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CrossEncoderConfig(**kwargs)  # type: ignore[arg-type]
