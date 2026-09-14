"""Provider-independent vector-store contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sre_rag.domain.documents import Chunk
from sre_rag.retrieval.embeddings import Vector


@dataclass(frozen=True, slots=True)
class VectorHit:
    """One vector-store result before conversion into pipeline history."""

    chunk: Chunk
    score: float


class VectorStore(Protocol):
    """Persist chunk vectors and execute nearest-neighbor search."""

    def ensure_collection(self, vector_size: int) -> None: ...

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None: ...

    def search(self, query_vector: Vector, *, limit: int) -> tuple[VectorHit, ...]: ...
