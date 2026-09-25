"""Public query endpoint for guarded answers and inspectable retrieval evidence."""

from time import perf_counter
from typing import Annotated, Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import Field, field_validator

from sre_rag.domain.answers import QueryResponse
from sre_rag.domain.base import DomainModel, NonEmptyText
from sre_rag.domain.generation import GenerationProvider
from sre_rag.domain.retrieval import RetrievalMethod, RetrievalQuery
from sre_rag.workflow.graph import RAGExecution

router = APIRouter(prefix="/v1", tags=["rag"])


class QueryWorkflow(Protocol):
    """Narrow workflow surface required by the HTTP adapter."""

    def invoke_with_evidence(self, query: RetrievalQuery) -> RAGExecution: ...


class QueryRequest(DomainModel):
    """One exact user question submitted to hybrid retrieval."""

    question: str = Field(max_length=2_000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class RetrievalStageView(DomainModel):
    """One rank and score assigned to evidence by a retrieval stage."""

    method: RetrievalMethod
    rank: int = Field(gt=0)
    score: float


class EvidenceView(DomainModel):
    """Public source excerpt and its complete ranking history."""

    chunk_id: NonEmptyText
    title: NonEmptyText
    source_path: NonEmptyText
    source_url: str
    text: NonEmptyText
    stages: tuple[RetrievalStageView, ...] = Field(min_length=1)


class GenerationView(DomainModel):
    """Safe generation metadata that excludes prompts and raw model output."""

    provider: GenerationProvider
    model: NonEmptyText
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class QueryResult(DomainModel):
    """Validated response consumed by the product dashboard."""

    query_id: UUID
    latency_ms: float = Field(ge=0)
    response: QueryResponse
    contexts: tuple[EvidenceView, ...]
    generation: GenerationView | None = None


def get_query_workflow(request: Request) -> QueryWorkflow:
    """Resolve the configured workflow without constructing models per request."""

    workflow: QueryWorkflow | None = request.app.state.query_workflow
    if workflow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "WORKFLOW_UNAVAILABLE",
                "message": "The RAG workflow has not been configured for this API process.",
            },
        )
    return workflow


@router.post("/query", response_model=QueryResult)
def query(
    payload: QueryRequest,
    workflow: Annotated[QueryWorkflow, Depends(get_query_workflow)],
) -> QueryResult:
    """Run hybrid retrieval and return only guarded output plus final evidence."""

    retrieval_query = RetrievalQuery(text=payload.question)
    started_at = perf_counter()
    execution = workflow.invoke_with_evidence(retrieval_query)
    latency_ms = (perf_counter() - started_at) * 1_000
    telemetry = execution.result.telemetry

    return QueryResult(
        query_id=retrieval_query.query_id,
        latency_ms=latency_ms,
        response=execution.result.response,
        contexts=tuple(
            EvidenceView(
                chunk_id=candidate.chunk.chunk_id,
                title=" > ".join(candidate.chunk.heading_path) or candidate.chunk.document_id,
                source_path=candidate.chunk.provenance.relative_path,
                source_url=str(candidate.chunk.provenance.source_url),
                text=candidate.chunk.text,
                stages=tuple(
                    RetrievalStageView(
                        method=stage.method,
                        rank=stage.rank,
                        score=stage.score,
                    )
                    for stage in candidate.stages
                ),
            )
            for candidate in execution.contexts
        ),
        generation=(
            GenerationView(
                provider=telemetry.provider,
                model=telemetry.model,
                input_tokens=telemetry.usage.input_tokens,
                output_tokens=telemetry.usage.output_tokens,
            )
            if telemetry is not None
            else None
        ),
    )
