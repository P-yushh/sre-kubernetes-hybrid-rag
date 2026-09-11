"""Provider-independent contracts shared across the RAG pipeline."""

from sre_rag.domain.answers import (
    Citation,
    GroundedAnswer,
    QueryResponse,
    RefusalCode,
    RefusalResponse,
)
from sre_rag.domain.documents import Chunk, Corpus, SourceDocument, SourceProvenance
from sre_rag.domain.retrieval import (
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalStage,
)

__all__ = [
    "Chunk",
    "Citation",
    "Corpus",
    "GroundedAnswer",
    "QueryResponse",
    "RefusalCode",
    "RefusalResponse",
    "RetrievalCandidate",
    "RetrievalMethod",
    "RetrievalQuery",
    "RetrievalStage",
    "SourceDocument",
    "SourceProvenance",
]
