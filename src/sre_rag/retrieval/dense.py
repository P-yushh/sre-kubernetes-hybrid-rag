"""Dense retrieval composed from embedding and vector-store interfaces."""

from collections.abc import Sequence

from sre_rag.domain.documents import Chunk
from sre_rag.domain.retrieval import (
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalStage,
)
from sre_rag.retrieval.embeddings import Embedder
from sre_rag.retrieval.vector_store import VectorStore


class DenseRetriever:
    """Index chunks and retrieve semantic candidates without provider coupling."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        *,
        default_limit: int = 10,
    ) -> None:
        if default_limit <= 0:
            raise ValueError("default limit must be positive")
        self._embedder = embedder
        self._vector_store = vector_store
        self._default_limit = default_limit

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Embed and idempotently store a unique collection of chunks."""

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("dense index requires unique chunk IDs")
        if not chunks:
            return

        self._vector_store.ensure_collection(self._embedder.dimension)
        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedder returned a different number of vectors than chunks")
        self._vector_store.upsert(chunks, vectors)

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        limit: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        """Embed a query and preserve Qdrant similarity as the dense stage score."""

        result_limit = self._default_limit if limit is None else limit
        if result_limit <= 0:
            raise ValueError("limit must be positive")

        query_vector = self._embedder.embed_query(query.text)
        hits = self._vector_store.search(query_vector, limit=result_limit)
        ordered_hits = sorted(hits, key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return tuple(
            RetrievalCandidate(
                chunk=hit.chunk,
                stages=(
                    RetrievalStage(
                        method=RetrievalMethod.DENSE,
                        score=hit.score,
                        rank=rank,
                    ),
                ),
            )
            for rank, hit in enumerate(ordered_hits[:result_limit], start=1)
        )
