"""Tests for provider-independent dense retrieval orchestration."""

import pytest

from sre_rag.domain.retrieval import RetrievalMethod, RetrievalQuery
from sre_rag.retrieval.dense import DenseRetriever
from sre_rag.retrieval.vector_store import VectorHit
from tests.domain.factories import make_chunk
from tests.retrieval.fakes import FakeEmbedder, FakeVectorStore


def test_index_embeds_chunk_text_and_upserts_vectors() -> None:
    chunk = make_chunk()
    embedder = FakeEmbedder()
    embedder.document_vectors = ((0.1, 0.2, 0.3),)
    store = FakeVectorStore()

    DenseRetriever(embedder, store).index((chunk,))

    assert embedder.document_inputs == [chunk.text]
    assert store.ensured_sizes == [3]
    assert store.upserts == [((chunk,), ((0.1, 0.2, 0.3),))]


def test_retrieve_returns_dense_stage_with_deterministic_ranks() -> None:
    first = make_chunk().model_copy(update={"chunk_id": "z-chunk"})
    second = make_chunk().model_copy(update={"chunk_id": "a-chunk"})
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    store.hits = (VectorHit(first, 0.8), VectorHit(second, 0.8))

    results = DenseRetriever(embedder, store).retrieve(
        RetrievalQuery(text="Why does the container keep restarting?"),
        limit=2,
    )

    assert embedder.query_inputs == ["Why does the container keep restarting?"]
    assert store.searches == [(embedder.query_vector, 2)]
    assert [result.chunk.chunk_id for result in results] == ["a-chunk", "z-chunk"]
    assert [result.stages[0].rank for result in results] == [1, 2]
    assert all(result.stages[0].method is RetrievalMethod.DENSE for result in results)


def test_empty_index_is_a_noop() -> None:
    embedder = FakeEmbedder()
    store = FakeVectorStore()

    DenseRetriever(embedder, store).index(())

    assert store.ensured_sizes == []
    assert embedder.document_inputs == []


def test_duplicate_chunk_ids_are_rejected() -> None:
    chunk = make_chunk()

    with pytest.raises(ValueError, match="unique chunk IDs"):
        DenseRetriever(FakeEmbedder(), FakeVectorStore()).index((chunk, chunk))


def test_embedding_count_must_match_chunk_count() -> None:
    with pytest.raises(ValueError, match="different number"):
        DenseRetriever(FakeEmbedder(), FakeVectorStore()).index((make_chunk(),))


def test_invalid_result_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="default limit"):
        DenseRetriever(FakeEmbedder(), FakeVectorStore(), default_limit=0)

    with pytest.raises(ValueError, match="limit"):
        DenseRetriever(FakeEmbedder(), FakeVectorStore()).retrieve(
            RetrievalQuery(text="OOMKilled"),
            limit=0,
        )
