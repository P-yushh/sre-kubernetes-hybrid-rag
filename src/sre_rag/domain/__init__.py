"""Provider-independent contracts shared across the RAG pipeline."""

from sre_rag.domain.answers import (
    Citation,
    GroundedAnswer,
    QueryResponse,
    RefusalCode,
    RefusalResponse,
)
from sre_rag.domain.documents import Chunk, Corpus, SourceDocument, SourceProvenance
from sre_rag.domain.evaluation import (
    EvaluationReport,
    EvaluationScores,
    EvaluationThresholds,
    GoldenDataset,
    GoldenSample,
    SampleEvaluation,
)
from sre_rag.domain.generation import (
    GeneratedAnswerDraft,
    GenerationProvider,
    GenerationResult,
    GenerationUsage,
)
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
    "EvaluationReport",
    "EvaluationScores",
    "EvaluationThresholds",
    "GroundedAnswer",
    "GeneratedAnswerDraft",
    "GenerationProvider",
    "GenerationResult",
    "GenerationUsage",
    "GoldenDataset",
    "GoldenSample",
    "QueryResponse",
    "RefusalCode",
    "RefusalResponse",
    "RetrievalCandidate",
    "RetrievalMethod",
    "RetrievalQuery",
    "RetrievalStage",
    "SampleEvaluation",
    "SourceDocument",
    "SourceProvenance",
]
