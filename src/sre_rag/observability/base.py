"""Provider-neutral contracts for traces and metrics."""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Literal, Protocol, TypeAlias

from sre_rag.guardrails.service import GuardedAnswerResult

AttributeValue: TypeAlias = str | int | float | bool
ObservationType: TypeAlias = Literal["span", "chain", "retriever", "generation", "guardrail"]


class ActiveObservation(Protocol):
    """A currently active trace observation with privacy-safe updates."""

    def update(
        self,
        *,
        output: Mapping[str, AttributeValue] | None = None,
        model: str | None = None,
        usage_details: Mapping[str, int] | None = None,
    ) -> None: ...


class TraceProvider(Protocol):
    """Create nested observations without exposing pipeline payloads."""

    def observe(
        self,
        *,
        name: str,
        observation_type: ObservationType,
        metadata: Mapping[str, AttributeValue],
    ) -> AbstractContextManager[ActiveObservation]: ...


class MetricsRecorder(Protocol):
    """Record bounded-cardinality service metrics."""

    def observe_stage(self, *, stage: str, status: str, duration_seconds: float) -> None: ...

    def record_result(self, result: GuardedAnswerResult) -> None: ...


class WorkflowTelemetry(Protocol):
    """Combined tracing and metrics surface consumed by the workflow."""

    def observe(
        self,
        *,
        stage: str,
        query_id: str,
        observation_type: ObservationType,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> AbstractContextManager[ActiveObservation]: ...

    def record_result(self, result: GuardedAnswerResult) -> None: ...
