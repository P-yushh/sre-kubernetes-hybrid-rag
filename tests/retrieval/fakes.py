"""Small deterministic fakes shared by dense-retrieval tests."""

from collections.abc import Sequence

from sre_rag.domain.documents import Chunk
from sre_rag.retrieval.embeddings import Vector
from sre_rag.retrieval.vector_store import VectorHit


class FakeEmbedder:
    """Return configured vectors while recording embedding inputs."""

    def __init__(self, *, dimension: int = 3) -> None:
        self.dimension = dimension
        self.document_inputs: list[str] = []
        self.query_inputs: list[str] = []
        self.document_vectors: tuple[Vector, ...] = ()
        self.query_vector: Vector = tuple(1.0 if index == 0 else 0.0 for index in range(dimension))

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        self.document_inputs.extend(texts)
        return self.document_vectors

    def embed_query(self, text: str) -> Vector:
        self.query_inputs.append(text)
        return self.query_vector


class FakeVectorStore:
    """Record index/search operations and return configured vector hits."""

    def __init__(self) -> None:
        self.ensured_sizes: list[int] = []
        self.upserts: list[tuple[tuple[Chunk, ...], tuple[Vector, ...]]] = []
        self.searches: list[tuple[Vector, int]] = []
        self.hits: tuple[VectorHit, ...] = ()

    def ensure_collection(self, vector_size: int) -> None:
        self.ensured_sizes.append(vector_size)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        self.upserts.append((tuple(chunks), tuple(vectors)))

    def search(self, query_vector: Vector, *, limit: int) -> tuple[VectorHit, ...]:
        self.searches.append((query_vector, limit))
        return self.hits
