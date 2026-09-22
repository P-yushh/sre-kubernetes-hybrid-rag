"""Tests for the compiled hybrid retrieval and answer graph."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

from sre_rag.domain.answers import GroundedAnswer, RefusalCode, RefusalResponse
from sre_rag.domain.retrieval import (
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalStage,
)
from sre_rag.generation.prompting import GroundedDraftGenerator
from sre_rag.guardrails.service import GroundedAnswerService, GuardedAnswerResult
from sre_rag.observability.base import (
    ActiveObservation,
    AttributeValue,
    ObservationType,
    WorkflowTelemetry,
)
from sre_rag.retrieval.fusion import ReciprocalRankFusion
from sre_rag.workflow.graph import RAGWorkflow, WorkflowConfig, build_rag_graph
from tests.domain.factories import make_chunk
from tests.generation.helpers import RecordingBackend, make_generation_result


@dataclass
class RecordingRetriever:
    candidates: tuple[RetrievalCandidate, ...]
    calls: list[tuple[RetrievalQuery, int | None]] = field(default_factory=list)

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        limit: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        self.calls.append((query, limit))
        return self.candidates[:limit]


@dataclass
class RecordingReranker:
    limits: list[int | None] = field(default_factory=list)

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalCandidate],
        *,
        limit: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        del query
        self.limits.append(limit)
        ordered = tuple(candidates[:limit])
        return tuple(
            candidate.model_copy(
                update={
                    "stages": (
                        *candidate.stages,
                        RetrievalStage(
                            method=RetrievalMethod.RERANKER,
                            score=float(len(ordered) - index),
                            rank=index + 1,
                        ),
                    )
                }
            )
            for index, candidate in enumerate(ordered)
        )


class RecordingObservation:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []

    def update(
        self,
        *,
        output: Mapping[str, AttributeValue] | None = None,
        model: str | None = None,
        usage_details: Mapping[str, int] | None = None,
    ) -> None:
        self.updates.append({"output": output, "model": model, "usage_details": usage_details})


class RecordingWorkflowTelemetry:
    def __init__(self) -> None:
        self.stages: list[tuple[str, dict[str, AttributeValue], RecordingObservation]] = []
        self.results: list[GuardedAnswerResult] = []

    @contextmanager
    def observe(
        self,
        *,
        stage: str,
        query_id: str,
        observation_type: ObservationType,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> Iterator[ActiveObservation]:
        observation = RecordingObservation()
        metadata: dict[str, AttributeValue] = {
            "query_id": query_id,
            "observation_type": observation_type,
        }
        metadata.update(attributes or {})
        self.stages.append((stage, metadata, observation))
        yield observation

    def record_result(self, result: GuardedAnswerResult) -> None:
        self.results.append(result)


def _candidate(chunk_id: str, method: RetrievalMethod, rank: int) -> RetrievalCandidate:
    chunk = make_chunk().model_copy(update={"chunk_id": chunk_id})
    return RetrievalCandidate(
        chunk=chunk,
        stages=(RetrievalStage(method=method, score=float(10 - rank), rank=rank),),
    )


def _workflow(
    sparse: RecordingRetriever,
    dense: RecordingRetriever,
    *,
    config: WorkflowConfig | None = None,
    telemetry: WorkflowTelemetry | None = None,
) -> tuple[RAGWorkflow, RecordingReranker, RecordingBackend]:
    reranker = RecordingReranker()
    backend = RecordingBackend(make_generation_result())
    workflow = RAGWorkflow(
        sparse_retriever=sparse,
        dense_retriever=dense,
        rank_fusion=ReciprocalRankFusion(),
        reranker=reranker,
        answer_service=GroundedAnswerService(GroundedDraftGenerator(backend)),
        config=config,
        telemetry=telemetry,
    )
    return workflow, reranker, backend


def test_workflow_runs_hybrid_retrieval_to_guarded_answer() -> None:
    shared_sparse = _candidate("shared", RetrievalMethod.BM25, 1)
    shared_dense = shared_sparse.model_copy(
        update={"stages": (RetrievalStage(method=RetrievalMethod.DENSE, score=0.95, rank=1),)}
    )
    sparse = RecordingRetriever((shared_sparse, _candidate("exact", RetrievalMethod.BM25, 2)))
    dense = RecordingRetriever((shared_dense, _candidate("semantic", RetrievalMethod.DENSE, 2)))
    workflow, reranker, backend = _workflow(sparse, dense)
    query = RetrievalQuery(text="Why was the pod OOMKilled with exit code 137?")

    result = workflow.invoke(query)

    assert isinstance(result.response, GroundedAnswer)
    assert result.response.citations[0].chunk_id == "shared"
    assert sparse.calls == [(query, 20)]
    assert dense.calls == [(query, 20)]
    assert reranker.limits == [5]
    assert len(backend.requests) == 1
    assert "exit code 137" in backend.requests[0].prompt


def test_workflow_limits_are_forwarded_to_every_ranking_stage() -> None:
    sparse = RecordingRetriever((_candidate("sparse", RetrievalMethod.BM25, 1),))
    dense = RecordingRetriever((_candidate("dense", RetrievalMethod.DENSE, 1),))
    config = WorkflowConfig(sparse_limit=3, dense_limit=4, fusion_limit=6, reranker_limit=2)
    workflow, reranker, _ = _workflow(sparse, dense, config=config)

    workflow.invoke(RetrievalQuery(text="question"))

    assert sparse.calls[0][1] == 3
    assert dense.calls[0][1] == 4
    assert reranker.limits == [2]


def test_workflow_exposes_exact_reranked_evidence_for_evaluation() -> None:
    sparse = RecordingRetriever((_candidate("sparse", RetrievalMethod.BM25, 1),))
    dense = RecordingRetriever((_candidate("dense", RetrievalMethod.DENSE, 1),))
    workflow, _, _ = _workflow(sparse, dense, config=WorkflowConfig(reranker_limit=1))

    execution = workflow.invoke_with_evidence(RetrievalQuery(text="question"))

    assert isinstance(execution.result.response, GroundedAnswer)
    assert len(execution.contexts) == 1
    assert execution.result.response.citations[0].chunk_id == execution.contexts[0].chunk.chunk_id
    assert execution.contexts[0].stages[-1].method is RetrievalMethod.RERANKER


def test_workflow_emits_safe_stage_metadata_and_generation_usage() -> None:
    sparse = RecordingRetriever((_candidate("sparse", RetrievalMethod.BM25, 1),))
    dense = RecordingRetriever((_candidate("dense", RetrievalMethod.DENSE, 1),))
    telemetry = RecordingWorkflowTelemetry()
    workflow, _, _ = _workflow(sparse, dense, telemetry=telemetry)

    result = workflow.invoke(RetrievalQuery(text="sensitive question with exit code 137"))

    assert {stage for stage, _, _ in telemetry.stages} == {
        "workflow",
        "retrieve_sparse",
        "retrieve_dense",
        "fuse",
        "rerank",
        "answer",
    }
    answer_observation = next(
        observation for stage, _, observation in telemetry.stages if stage == "answer"
    )
    assert answer_observation.updates[0]["model"] == "test-model"
    assert answer_observation.updates[0]["usage_details"] == {"input": 20, "output": 10}
    assert telemetry.results == [result]
    trace_payload = repr(telemetry.stages)
    assert "sensitive question" not in trace_payload
    assert "exit code 137" not in trace_payload


def test_empty_retrieval_returns_refusal_without_generation() -> None:
    telemetry = RecordingWorkflowTelemetry()
    workflow, _, backend = _workflow(
        RecordingRetriever(()), RecordingRetriever(()), telemetry=telemetry
    )

    result = workflow.invoke(RetrievalQuery(text="unknown operational issue"))

    assert isinstance(result.response, RefusalResponse)
    assert result.response.code is RefusalCode.INSUFFICIENT_CONTEXT
    assert backend.requests == []
    answer_metadata = next(metadata for stage, metadata, _ in telemetry.stages if stage == "answer")
    assert answer_metadata["observation_type"] == "guardrail"


def test_compiled_graph_exposes_explicit_pipeline_nodes() -> None:
    sparse = RecordingRetriever(())
    dense = RecordingRetriever(())
    reranker = RecordingReranker()
    backend = RecordingBackend(make_generation_result())
    graph = build_rag_graph(
        sparse_retriever=sparse,
        dense_retriever=dense,
        rank_fusion=ReciprocalRankFusion(),
        reranker=reranker,
        answer_service=GroundedAnswerService(GroundedDraftGenerator(backend)),
    )

    assert set(graph.get_graph().nodes) == {
        "__start__",
        "retrieve_sparse",
        "retrieve_dense",
        "fuse",
        "rerank",
        "answer",
        "__end__",
    }


@pytest.mark.parametrize(
    "config",
    [
        WorkflowConfig(sparse_limit=1),
        WorkflowConfig(dense_limit=1),
        WorkflowConfig(fusion_limit=1),
        WorkflowConfig(reranker_limit=1),
    ],
)
def test_workflow_config_accepts_positive_limits(config: WorkflowConfig) -> None:
    assert (
        min(
            config.sparse_limit,
            config.dense_limit,
            config.fusion_limit,
            config.reranker_limit,
        )
        > 0
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"sparse_limit": 0},
        {"dense_limit": 0},
        {"fusion_limit": 0},
        {"reranker_limit": 0},
    ],
)
def test_workflow_config_rejects_non_positive_limits(overrides: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        WorkflowConfig(**overrides)
