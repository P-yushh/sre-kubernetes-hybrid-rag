"""Tests for the guarded query API."""

from dataclasses import dataclass, field

import pytest
from httpx import ASGITransport, AsyncClient

from sre_rag.domain.answers import GroundedAnswer, RefusalCode, RefusalResponse
from sre_rag.domain.generation import GenerationProvider, GenerationUsage
from sre_rag.domain.retrieval import RetrievalQuery
from sre_rag.guardrails.service import GenerationTelemetry, GuardedAnswerResult
from sre_rag.main import create_app
from sre_rag.workflow.graph import RAGExecution
from tests.domain.test_answers import make_citation
from tests.generation.helpers import make_reranked_candidate


@dataclass
class StubWorkflow:
    execution: RAGExecution
    queries: list[RetrievalQuery] = field(default_factory=list)

    def invoke_with_evidence(self, query: RetrievalQuery) -> RAGExecution:
        self.queries.append(query)
        return self.execution


def _answer_execution() -> RAGExecution:
    candidate = make_reranked_candidate(
        "kubernetes:debug-application#exit-codes@0",
        "A container terminated with exit code 137 and reason OOMKilled.",
        1,
    )
    return RAGExecution(
        result=GuardedAnswerResult(
            response=GroundedAnswer(
                answer="Exit code 137 commonly indicates an out-of-memory termination [1].",
                citations=(make_citation(),),
            ),
            telemetry=GenerationTelemetry(
                provider=GenerationProvider.OPENAI,
                model="test-model",
                response_id="response-1",
                usage=GenerationUsage(input_tokens=30, output_tokens=12),
            ),
        ),
        contexts=(candidate,),
    )


@pytest.mark.anyio
async def test_query_endpoint_returns_guarded_answer_evidence_and_usage() -> None:
    workflow = StubWorkflow(_answer_execution())
    transport = ASGITransport(app=create_app(query_workflow=workflow))

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/query",
            json={"question": "Why was the pod OOMKilled with exit code 137?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response"]["outcome"] == "answer"
    assert payload["response"]["citations"][0]["chunk_id"] == payload["contexts"][0]["chunk_id"]
    assert payload["contexts"][0]["title"] == "Debug Pods > Exit codes"
    assert [stage["method"] for stage in payload["contexts"][0]["stages"]] == [
        "rrf",
        "reranker",
    ]
    assert payload["generation"] == {
        "provider": "openai",
        "model": "test-model",
        "input_tokens": 30,
        "output_tokens": 12,
    }
    assert payload["latency_ms"] >= 0
    assert workflow.queries[0].text == "Why was the pod OOMKilled with exit code 137?"


@pytest.mark.anyio
async def test_query_endpoint_preserves_deterministic_refusal() -> None:
    execution = RAGExecution(
        result=GuardedAnswerResult(
            response=RefusalResponse(
                code=RefusalCode.INSUFFICIENT_CONTEXT,
                message="The retrieved sources do not contain enough evidence.",
            ),
            telemetry=None,
        ),
        contexts=(),
    )
    transport = ASGITransport(app=create_app(query_workflow=StubWorkflow(execution)))

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/query", json={"question": "Unknown issue"})

    assert response.status_code == 200
    assert response.json()["response"]["code"] == "INSUFFICIENT_CONTEXT"
    assert response.json()["contexts"] == []
    assert response.json()["generation"] is None


@pytest.mark.anyio
async def test_query_endpoint_fails_closed_without_runtime_workflow() -> None:
    transport = ASGITransport(app=create_app())

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/query", json={"question": "Why OOMKilled?"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "WORKFLOW_UNAVAILABLE"


@pytest.mark.anyio
@pytest.mark.parametrize("question", ["", "   ", "x" * 2_001])
async def test_query_endpoint_rejects_invalid_questions(question: str) -> None:
    transport = ASGITransport(app=create_app(query_workflow=StubWorkflow(_answer_execution())))

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/query", json={"question": question})

    assert response.status_code == 422
