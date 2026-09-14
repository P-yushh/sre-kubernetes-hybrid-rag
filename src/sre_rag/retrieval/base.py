"""Provider-independent retrieval interfaces."""

from typing import Protocol

from sre_rag.domain.retrieval import RetrievalCandidate, RetrievalQuery


class Retriever(Protocol):
    """Rank chunks for a validated query without exposing backend details."""

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        limit: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]: ...
