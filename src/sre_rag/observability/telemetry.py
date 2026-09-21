"""Composition of privacy-safe tracing and operational metrics."""

import logging
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from enum import StrEnum
from time import perf_counter

from sre_rag.guardrails.service import GuardedAnswerResult
from sre_rag.observability.base import (
    ActiveObservation,
    AttributeValue,
    MetricsRecorder,
    ObservationType,
    TraceProvider,
)


class PipelineStage(StrEnum):
    """Stable bounded stage names used by traces and metric labels."""

    WORKFLOW = "workflow"
    RETRIEVE_SPARSE = "retrieve_sparse"
    RETRIEVE_DENSE = "retrieve_dense"
    FUSE = "fuse"
    RERANK = "rerank"
    ANSWER = "answer"


_LOGGER = logging.getLogger(__name__)


class _NullObservation:
    def update(
        self,
        *,
        output: Mapping[str, AttributeValue] | None = None,
        model: str | None = None,
        usage_details: Mapping[str, int] | None = None,
    ) -> None:
        del output, model, usage_details


class NullWorkflowTelemetry:
    """Zero-I/O default for tests and deployments without observability."""

    @contextmanager
    def observe(
        self,
        *,
        stage: str,
        query_id: str,
        observation_type: ObservationType,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> Iterator[ActiveObservation]:
        del stage, query_id, observation_type, attributes
        yield _NullObservation()

    def record_result(self, result: GuardedAnswerResult) -> None:
        del result


class PipelineTelemetry:
    """Time every stage locally while nesting metadata-only cloud spans."""

    def __init__(self, tracer: TraceProvider, metrics: MetricsRecorder) -> None:
        self._tracer = tracer
        self._metrics = metrics

    @contextmanager
    def observe(
        self,
        *,
        stage: str,
        query_id: str,
        observation_type: ObservationType,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> Iterator[ActiveObservation]:
        metadata: dict[str, AttributeValue] = {"query_id": query_id}
        metadata.update(attributes or {})
        started_at = perf_counter()
        status = "success"
        trace_context: AbstractContextManager[ActiveObservation] | None = None
        observation: ActiveObservation = _NullObservation()
        try:
            trace_context = self._tracer.observe(
                name=f"rag.{stage}",
                observation_type=observation_type,
                metadata=metadata,
            )
            observation = trace_context.__enter__()
        except Exception:
            trace_context = None
            _LOGGER.exception("Trace observation could not be started", extra={"stage": stage})

        error: BaseException | None = None
        try:
            yield observation
        except BaseException as caught:
            status = "error"
            error = caught
            raise
        finally:
            if trace_context is not None:
                try:
                    trace_context.__exit__(
                        type(error) if error else None,
                        error,
                        error.__traceback__ if error else None,
                    )
                except Exception:
                    _LOGGER.exception(
                        "Trace observation could not be closed", extra={"stage": stage}
                    )
            try:
                self._metrics.observe_stage(
                    stage=stage,
                    status=status,
                    duration_seconds=perf_counter() - started_at,
                )
            except Exception:
                _LOGGER.exception("Stage metric could not be recorded", extra={"stage": stage})

    def record_result(self, result: GuardedAnswerResult) -> None:
        try:
            self._metrics.record_result(result)
        except Exception:
            _LOGGER.exception("Query metrics could not be recorded")
