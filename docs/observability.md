# Observability

The query workflow emits privacy-safe Langfuse observations and bounded-cardinality Prometheus
metrics. Questions, retrieved text, prompts, and generated answers are excluded from telemetry by
default.

## Trace structure

Each query creates a root `rag.workflow` chain with child observations for sparse retrieval, dense
retrieval, rank fusion, reranking, and answer generation. Observations contain only the query ID,
configured limits, candidate counts, public outcome, refusal code, provider, model, and token usage.

Langfuse infers hosted-model cost when the recorded model name matches a model definition and the
generation contains input and output usage. Configure a custom model definition for local or
privately priced models.

## Prometheus metrics

- `sre_rag_stage_duration_seconds`: histogram labeled by bounded stage and status values.
- `sre_rag_queries_total`: counter labeled by outcome, refusal code, and provider.
- `sre_rag_generation_tokens_total`: counter labeled by provider, model, and token type.

Prometheus calculates percentiles from histogram buckets. Example five-minute queries:

```promql
histogram_quantile(
  0.50,
  sum by (le) (rate(sre_rag_stage_duration_seconds_bucket{stage="workflow",status="success"}[5m]))
)
```

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(sre_rag_stage_duration_seconds_bucket{stage="workflow",status="success"}[5m]))
)
```

Prometheus scrapes the service at `/metrics`. Query IDs are deliberately excluded from metric
labels because one label value per request would create unbounded time-series cardinality.

The local Compose stack provisions Prometheus and a read-only Grafana operations dashboard for
these metrics. Startup instructions and service URLs are in
[`local-development.md`](local-development.md).
