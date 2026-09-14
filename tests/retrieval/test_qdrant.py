"""Integration tests for Qdrant storage using its in-memory client."""

from qdrant_client import QdrantClient, models

from sre_rag.retrieval.qdrant import QdrantVectorStore, _point_id
from tests.domain.factories import make_chunk


def test_chunk_round_trip_and_cosine_search() -> None:
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client, collection_name="chunks", upload_batch_size=1)
    relevant = make_chunk().model_copy(update={"chunk_id": "relevant"})
    unrelated = make_chunk().model_copy(update={"chunk_id": "unrelated"})

    store.ensure_collection(3)
    store.upsert((relevant, unrelated), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    results = store.search((1.0, 0.0, 0.0), limit=2)

    assert [hit.chunk.chunk_id for hit in results] == ["relevant", "unrelated"]
    assert results[0].chunk.provenance == relevant.provenance
    assert results[0].score > results[1].score


def test_reindexing_same_chunk_is_idempotent() -> None:
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client, collection_name="chunks")
    chunk = make_chunk()
    store.ensure_collection(3)

    store.upsert((chunk,), ((1.0, 0.0, 0.0),))
    store.upsert((chunk,), ((0.0, 1.0, 0.0),))

    assert client.count("chunks", exact=True).count == 1
    assert _point_id(chunk.chunk_id) == _point_id(chunk.chunk_id)


def test_existing_collection_schema_is_validated() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="wrong-size",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )

    store = QdrantVectorStore(client, collection_name="wrong-size")

    try:
        store.ensure_collection(3)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("mismatched vector schema should be rejected")


def test_store_validates_arguments_and_operation_order() -> None:
    client = QdrantClient(":memory:")

    for kwargs in ({"collection_name": " "}, {"upload_batch_size": 0}):
        try:
            QdrantVectorStore(client, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid store configuration should be rejected")

    store = QdrantVectorStore(client, collection_name="chunks")
    try:
        store.ensure_collection(0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid vector size should be rejected")

    try:
        store.upsert((make_chunk(),), ((1.0, 0.0, 0.0),))
    except RuntimeError as error:
        assert "ensure_collection" in str(error)
    else:
        raise AssertionError("upsert before collection creation should be rejected")


def test_store_rejects_count_and_dimension_mismatches() -> None:
    store = QdrantVectorStore(QdrantClient(":memory:"), collection_name="chunks")
    store.ensure_collection(3)

    try:
        store.upsert((make_chunk(),), ())
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("chunk-vector count mismatch should be rejected")

    try:
        store.upsert((make_chunk(),), ((1.0, 0.0),))
    except ValueError as error:
        assert "dimension" in str(error)
    else:
        raise AssertionError("vector dimension mismatch should be rejected")

    try:
        store.search((1.0, 0.0), limit=1)
    except ValueError as error:
        assert "dimension" in str(error)
    else:
        raise AssertionError("query dimension mismatch should be rejected")
