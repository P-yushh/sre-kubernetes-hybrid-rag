"""Prometheus metrics for latency, outcomes, and generation usage."""

from prometheus_client import CollectorRegistry, Counter, Histogram

from sre_rag.domain.answers import RefusalResponse
from sre_rag.guardrails.service import GuardedAnswerResult

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class PrometheusMetrics:
    """Record low-cardinality metrics suitable for percentile queries."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self._stage_latency = Histogram(
            "stage_duration_seconds",
            "Duration of each RAG pipeline stage.",
            labelnames=("stage", "status"),
            namespace="sre_rag",
            buckets=_LATENCY_BUCKETS,
            registry=registry,
        )
        self._queries = Counter(
            "queries_total",
            "Completed RAG queries by public outcome.",
            labelnames=("outcome", "refusal_code", "provider"),
            namespace="sre_rag",
            registry=registry,
        )
        self._tokens = Counter(
            "generation_tokens_total",
            "Provider-reported generation tokens.",
            labelnames=("provider", "model", "token_type"),
            namespace="sre_rag",
            registry=registry,
        )

    def observe_stage(self, *, stage: str, status: str, duration_seconds: float) -> None:
        self._stage_latency.labels(stage=stage, status=status).observe(duration_seconds)

    def record_result(self, result: GuardedAnswerResult) -> None:
        refusal_code = "none"
        if isinstance(result.response, RefusalResponse):
            refusal_code = result.response.code.value
        provider = result.telemetry.provider.value if result.telemetry else "none"
        self._queries.labels(
            outcome=result.response.outcome,
            refusal_code=refusal_code,
            provider=provider,
        ).inc()

        if result.telemetry is None:
            return
        labels = {
            "provider": result.telemetry.provider.value,
            "model": result.telemetry.model,
        }
        self._tokens.labels(**labels, token_type="input").inc(result.telemetry.usage.input_tokens)
        self._tokens.labels(**labels, token_type="output").inc(result.telemetry.usage.output_tokens)
