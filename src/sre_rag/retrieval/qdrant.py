"""Qdrant implementation of the dense vector-store contract."""

from collections.abc import Sequence
from typing import cast
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from sre_rag.domain.documents import Chunk
from sre_rag.retrieval.embeddings import Vector
from sre_rag.retrieval.vector_store import VectorHit


class QdrantVectorStore:
    """Store normalized chunk vectors with reconstructable provenance payloads."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection_name: str = "sre_kubernetes_chunks",
        upload_batch_size: int = 64,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("Qdrant collection name must not be blank")
        if upload_batch_size <= 0:
            raise ValueError("Qdrant upload batch size must be positive")
        self._client = client
        self._collection_name = collection_name
        self._upload_batch_size = upload_batch_size
        self._vector_size: int | None = None

    def ensure_collection(self, vector_size: int) -> None:
        """Create a cosine collection or validate the existing vector schema."""

        if vector_size <= 0:
            raise ValueError("vector size must be positive")
        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        else:
            collection = self._client.get_collection(self._collection_name)
            vectors = collection.config.params.vectors
            if not isinstance(vectors, models.VectorParams):
                raise ValueError("Qdrant collection must use one unnamed dense vector")
            if vectors.size != vector_size or vectors.distance != models.Distance.COSINE:
                raise ValueError(
                    "Qdrant collection vector configuration does not match the embedder"
                )
        self._vector_size = vector_size

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        """Idempotently upload vectors using stable UUIDs derived from chunk IDs."""

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        if self._vector_size is None:
            raise RuntimeError("ensure_collection must be called before upsert")
        if any(len(vector) != self._vector_size for vector in vectors):
            raise ValueError("vector dimension does not match the Qdrant collection")

        points = (
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=list(vector),
                payload={"chunk": chunk.model_dump(mode="json")},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        )
        self._client.upload_points(
            collection_name=self._collection_name,
            points=points,
            batch_size=self._upload_batch_size,
            wait=True,
        )

    def search(self, query_vector: Vector, *, limit: int) -> tuple[VectorHit, ...]:
        """Search by cosine similarity and validate every stored chunk payload."""

        if limit <= 0:
            raise ValueError("vector search limit must be positive")
        if self._vector_size is not None and len(query_vector) != self._vector_size:
            raise ValueError("query vector dimension does not match the Qdrant collection")

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=list(query_vector),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        hits: list[VectorHit] = []
        for point in response.points:
            payload = point.payload or {}
            chunk_payload = payload.get("chunk")
            if not isinstance(chunk_payload, dict):
                raise ValueError(f"Qdrant point {point.id} has no valid chunk payload")
            chunk = Chunk.model_validate(cast(dict[str, object], chunk_payload))
            hits.append(VectorHit(chunk=chunk, score=float(point.score)))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return tuple(hits)


def _point_id(chunk_id: str) -> str:
    """Map an arbitrary stable chunk ID to a Qdrant-compatible UUID."""

    return str(uuid5(NAMESPACE_URL, f"sre-rag:{chunk_id}"))
