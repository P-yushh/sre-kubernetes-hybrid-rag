"""Typed LangGraph composition of retrieval, ranking, and grounded generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from sre_rag.domain.retrieval import RetrievalCandidate, RetrievalMethod, RetrievalQuery
from sre_rag.guardrails.service import GuardedAnswerResult
from sre_rag.observability.base import ActiveObservation, WorkflowTelemetry
from sre_rag.observability.telemetry import NullWorkflowTelemetry, PipelineStage
from sre_rag.retrieval.base import Retriever
from sre_rag.retrieval.fusion import RankFusion
from sre_rag.retrieval.reranking import Reranker


class AnswerService(Protocol):
    """Convert reranked evidence into a guarded public response."""

    def answer(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalCandidate],
    ) -> GuardedAnswerResult: ...


class RAGInput(TypedDict):
    """Validated input accepted by the compiled workflow."""

    query: RetrievalQuery


class RAGOutput(TypedDict):
    """Public result emitted by the compiled workflow."""

    result: GuardedAnswerResult
    reranked_candidates: tuple[RetrievalCandidate, ...]


class RAGState(RAGInput, total=False):
    """Private state passed between explicit workflow nodes."""

    sparse_candidates: tuple[RetrievalCandidate, ...]
    dense_candidates: tuple[RetrievalCandidate, ...]
    fused_candidates: tuple[RetrievalCandidate, ...]
    reranked_candidates: tuple[RetrievalCandidate, ...]
    result: GuardedAnswerResult


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """Per-stage bounds that keep query work and latency predictable."""

    sparse_limit: int = 20
    dense_limit: int = 20
    fusion_limit: int = 20
    reranker_limit: int = 5

    def __post_init__(self) -> None:
        limits = {
            "sparse": self.sparse_limit,
            "dense": self.dense_limit,
            "fusion": self.fusion_limit,
            "reranker": self.reranker_limit,
        }
        invalid = [name for name, value in limits.items() if value <= 0]
        if invalid:
            names = ", ".join(invalid)
            raise ValueError(f"workflow limits must be positive: {names}")


@dataclass(frozen=True, slots=True)
class RAGExecution:
    """Guarded response plus the exact evidence retained for evaluation."""

    result: GuardedAnswerResult
    contexts: tuple[RetrievalCandidate, ...]


class RAGWorkflow:
    """Small application facade over a compiled deterministic state graph."""

    def __init__(
        self,
        sparse_retriever: Retriever,
        dense_retriever: Retriever,
        rank_fusion: RankFusion,
        reranker: Reranker,
        answer_service: AnswerService,
        config: WorkflowConfig | None = None,
        telemetry: WorkflowTelemetry | None = None,
    ) -> None:
        self._telemetry = telemetry or NullWorkflowTelemetry()
        self._graph = build_rag_graph(
            sparse_retriever=sparse_retriever,
            dense_retriever=dense_retriever,
            rank_fusion=rank_fusion,
            reranker=reranker,
            answer_service=answer_service,
            config=config,
            telemetry=self._telemetry,
        )

    def invoke(self, query: RetrievalQuery) -> GuardedAnswerResult:
        """Run one query through the complete graph and return its guarded result."""

        return self.invoke_with_evidence(query).result

    def invoke_with_evidence(self, query: RetrievalQuery) -> RAGExecution:
        """Run one query and retain the final evidence for offline evaluation."""

        query_id = str(query.query_id)
        with self._telemetry.observe(
            stage=PipelineStage.WORKFLOW,
            query_id=query_id,
            observation_type="chain",
        ) as observation:
            output = self._graph.invoke({"query": query})
            result = cast(GuardedAnswerResult, output["result"])
            contexts = cast(tuple[RetrievalCandidate, ...], output["reranked_candidates"])
            observation.update(output=_result_summary(result))
        self._telemetry.record_result(result)
        return RAGExecution(result=result, contexts=contexts)


def build_rag_graph(
    *,
    sparse_retriever: Retriever,
    dense_retriever: Retriever,
    rank_fusion: RankFusion,
    reranker: Reranker,
    answer_service: AnswerService,
    config: WorkflowConfig | None = None,
    telemetry: WorkflowTelemetry | None = None,
) -> CompiledStateGraph[RAGState, None, RAGInput, RAGOutput]:
    """Compile the fixed hybrid-RAG topology with parallel retrieval branches."""

    resolved = config or WorkflowConfig()
    observed = telemetry or NullWorkflowTelemetry()

    def retrieve_sparse(state: RAGState) -> dict[str, object]:
        with observed.observe(
            stage=PipelineStage.RETRIEVE_SPARSE,
            query_id=str(state["query"].query_id),
            observation_type="retriever",
            attributes={"limit": resolved.sparse_limit},
        ) as observation:
            candidates = sparse_retriever.retrieve(state["query"], limit=resolved.sparse_limit)
            observation.update(output={"candidate_count": len(candidates)})
        return {"sparse_candidates": candidates}

    def retrieve_dense(state: RAGState) -> dict[str, object]:
        with observed.observe(
            stage=PipelineStage.RETRIEVE_DENSE,
            query_id=str(state["query"].query_id),
            observation_type="retriever",
            attributes={"limit": resolved.dense_limit},
        ) as observation:
            candidates = dense_retriever.retrieve(state["query"], limit=resolved.dense_limit)
            observation.update(output={"candidate_count": len(candidates)})
        return {"dense_candidates": candidates}

    def fuse(state: RAGState) -> dict[str, object]:
        with observed.observe(
            stage=PipelineStage.FUSE,
            query_id=str(state["query"].query_id),
            observation_type="span",
            attributes={"limit": resolved.fusion_limit},
        ) as observation:
            candidates = rank_fusion.fuse(
                {
                    RetrievalMethod.BM25: state["sparse_candidates"],
                    RetrievalMethod.DENSE: state["dense_candidates"],
                },
                limit=resolved.fusion_limit,
            )
            observation.update(output={"candidate_count": len(candidates)})
        return {"fused_candidates": candidates}

    def rerank(state: RAGState) -> dict[str, object]:
        with observed.observe(
            stage=PipelineStage.RERANK,
            query_id=str(state["query"].query_id),
            observation_type="span",
            attributes={"limit": resolved.reranker_limit},
        ) as observation:
            candidates = reranker.rerank(
                state["query"],
                state["fused_candidates"],
                limit=resolved.reranker_limit,
            )
            observation.update(output={"candidate_count": len(candidates)})
        return {"reranked_candidates": candidates}

    def answer(state: RAGState) -> dict[str, object]:
        contexts = state["reranked_candidates"]
        with observed.observe(
            stage=PipelineStage.ANSWER,
            query_id=str(state["query"].query_id),
            observation_type="generation" if contexts else "guardrail",
            attributes={"context_count": len(contexts)},
        ) as observation:
            result = answer_service.answer(
                state["query"],
                contexts,
            )
            _update_answer_observation(observation, result)
        return {"result": result}

    builder = StateGraph(
        RAGState,
        input_schema=RAGInput,
        output_schema=RAGOutput,
    )
    builder.add_node("retrieve_sparse", retrieve_sparse)
    builder.add_node("retrieve_dense", retrieve_dense)
    builder.add_node("fuse", fuse)
    builder.add_node("rerank", rerank)
    builder.add_node("answer", answer)

    builder.add_edge(START, "retrieve_sparse")
    builder.add_edge(START, "retrieve_dense")
    builder.add_edge(["retrieve_sparse", "retrieve_dense"], "fuse")
    builder.add_edge("fuse", "rerank")
    builder.add_edge("rerank", "answer")
    builder.add_edge("answer", END)
    return builder.compile(name="sre-kubernetes-hybrid-rag")


def _update_answer_observation(
    observation: ActiveObservation,
    result: GuardedAnswerResult,
) -> None:
    telemetry = result.telemetry
    observation.update(
        output=_result_summary(result),
        model=telemetry.model if telemetry else None,
        usage_details=(
            {
                "input": telemetry.usage.input_tokens,
                "output": telemetry.usage.output_tokens,
            }
            if telemetry
            else None
        ),
    )


def _result_summary(result: GuardedAnswerResult) -> dict[str, str]:
    code = getattr(result.response, "code", None)
    return {
        "outcome": result.response.outcome,
        "refusal_code": code.value if code is not None else "none",
        "provider": result.telemetry.provider.value if result.telemetry else "none",
    }
