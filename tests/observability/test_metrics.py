"""Tests for bounded Prometheus metrics and the scrape endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient
from prometheus_client import CollectorRegistry, generate_latest

from sre_rag.config import Settings
from sre_rag.domain.answers import RefusalCode, RefusalResponse
from sre_rag.domain.generation import GenerationProvider, GenerationUsage
from sre_rag.guardrails.service import GenerationTelemetry, GuardedAnswerResult
from sre_rag.main import create_app
from sre_rag.observability.metrics import PrometheusMetrics


def _refusal_result() -> GuardedAnswerResult:
    return GuardedAnswerResult(
        response=RefusalResponse(
            code=RefusalCode.CITATION_VALIDATION_FAILED,
            message="The answer could not be verified.",
        ),
        telemetry=GenerationTelemetry(
            provider=GenerationProvider.OPENAI,
            model="test-model",
            response_id="response-1",
            usage=GenerationUsage(input_tokens=30, output_tokens=12),
        ),
    )


def test_prometheus_metrics_record_latency_outcome_and_tokens() -> None:
    registry = CollectorRegistry()
    metrics = PrometheusMetrics(registry)

    metrics.observe_stage(stage="answer", status="success", duration_seconds=0.25)
    metrics.record_result(_refusal_result())
    payload = generate_latest(registry).decode()

    assert 'sre_rag_stage_duration_seconds_count{stage="answer",status="success"} 1.0' in payload
    assert 'sre_rag_queries_total{outcome="refusal",provider="openai",' in payload
    assert 'refusal_code="CITATION_VALIDATION_FAILED"' in payload
    assert (
        'sre_rag_generation_tokens_total{model="test-model",provider="openai",'
        'token_type="input"} 30.0'
    ) in payload
    assert (
        'sre_rag_generation_tokens_total{model="test-model",provider="openai",'
        'token_type="output"} 12.0'
    ) in payload


def test_result_without_generation_does_not_create_token_metrics() -> None:
    registry = CollectorRegistry()
    metrics = PrometheusMetrics(registry)
    result = GuardedAnswerResult(
        response=RefusalResponse(
            code=RefusalCode.INSUFFICIENT_CONTEXT,
            message="No evidence was retrieved.",
        ),
        telemetry=None,
    )

    metrics.record_result(result)
    payload = generate_latest(registry).decode()

    assert 'provider="none"' in payload
    assert "sre_rag_generation_tokens_total{" not in payload


@pytest.mark.anyio
async def test_metrics_endpoint_exposes_selected_registry() -> None:
    registry = CollectorRegistry()
    metrics = PrometheusMetrics(registry)
    metrics.observe_stage(stage="workflow", status="success", duration_seconds=0.1)
    transport = ASGITransport(
        app=create_app(Settings(environment="test"), metrics_registry=registry)
    )

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "sre_rag_stage_duration_seconds" in response.text
