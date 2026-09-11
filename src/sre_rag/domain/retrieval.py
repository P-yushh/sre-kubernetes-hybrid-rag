"""Contracts for queries and ranked retrieval candidates."""

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from sre_rag.domain.base import DomainModel
from sre_rag.domain.documents import Chunk


class RetrievalMethod(StrEnum):
    """Named retrieval stages whose scores must remain distinct."""

    BM25 = "bm25"
    DENSE = "dense"
    RRF = "rrf"
    RERANKER = "reranker"


class RetrievalQuery(DomainModel):
    """A traceable user query passed to retrieval components."""

    query_id: UUID = Field(default_factory=uuid4)
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject empty queries without normalizing exact search tokens."""

        if not value.strip():
            raise ValueError("query text must not be blank")
        return value


class RetrievalStage(DomainModel):
    """The score and one-based rank assigned by one retrieval stage."""

    method: RetrievalMethod
    score: float
    rank: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_rrf_score(self) -> "RetrievalStage":
        """RRF scores are sums of positive reciprocal-rank contributions."""

        if self.method is RetrievalMethod.RRF and self.score <= 0:
            raise ValueError("RRF score must be positive")
        return self


class RetrievalCandidate(DomainModel):
    """A chunk and its independent ranking history across pipeline stages."""

    chunk: Chunk
    stages: tuple[RetrievalStage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def methods_must_be_unique(self) -> "RetrievalCandidate":
        """Prevent ambiguous duplicate scores for the same stage."""

        methods = [stage.method for stage in self.stages]
        if len(methods) != len(set(methods)):
            raise ValueError("retrieval stage methods must be unique per candidate")
        return self
