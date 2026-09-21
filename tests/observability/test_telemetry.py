"""Tests for resilient trace and metric composition."""

from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager

import pytest

from sre_rag.guardrails.service import GuardedAnswerResult
from sre_rag.observability.base import ActiveObservation, AttributeValue, ObservationType
from sre_rag.observability.telemetry import PipelineTelemetry


class RecordingObservation:
    def update(
        self,
        *,
        output: Mapping[str, AttributeValue] | None = None,
        model: str | None = None,
        usage_details: Mapping[str, int] | None = None,
    ) -> None:
        del output, model, usage_details


class RecordingTracer:
    def __init__(self, *, fail_start: bool = False, fail_close: bool = False) -> None:
        self.fail_start = fail_start
        self.fail_close = fail_close
        self.calls: list[dict[str, object]] = []

    def observe(
        self,
        *,
        name: str,
        observation_type: ObservationType,
        metadata: Mapping[str, AttributeValue],
    ) -> AbstractContextManager[ActiveObservation]:
        self.calls.append(
            {"name": name, "observation_type": observation_type, "metadata": dict(metadata)}
        )
        if self.fail_start:
            raise RuntimeError("trace start failed")

        @contextmanager
        def context() -> Iterator[ActiveObservation]:
            yield RecordingObservation()
            if self.fail_close:
                raise RuntimeError("trace close failed")

        return context()


class RecordingMetrics:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.stages: list[tuple[str, str, float]] = []
        self.results: list[GuardedAnswerResult] = []

    def observe_stage(self, *, stage: str, status: str, duration_seconds: float) -> None:
        if self.fail:
            raise RuntimeError("metric failed")
        self.stages.append((stage, status, duration_seconds))

    def record_result(self, result: GuardedAnswerResult) -> None:
        if self.fail:
            raise RuntimeError("metric failed")
        self.results.append(result)


def test_pipeline_telemetry_records_successful_stage() -> None:
    tracer = RecordingTracer()
    metrics = RecordingMetrics()
    telemetry = PipelineTelemetry(tracer, metrics)

    with telemetry.observe(
        stage="rerank",
        query_id="query-1",
        observation_type="span",
        attributes={"limit": 5},
    ):
        pass

    assert tracer.calls[0]["metadata"] == {"query_id": "query-1", "limit": 5}
    assert metrics.stages[0][0:2] == ("rerank", "success")
    assert metrics.stages[0][2] >= 0


def test_application_failure_is_recorded_and_propagated() -> None:
    metrics = RecordingMetrics()
    telemetry = PipelineTelemetry(RecordingTracer(), metrics)

    with pytest.raises(ValueError, match="pipeline failed"):
        with telemetry.observe(
            stage="fuse",
            query_id="query-1",
            observation_type="span",
        ):
            raise ValueError("pipeline failed")

    assert metrics.stages[0][0:2] == ("fuse", "error")


@pytest.mark.parametrize(
    "tracer",
    [RecordingTracer(fail_start=True), RecordingTracer(fail_close=True)],
)
def test_trace_failures_do_not_break_pipeline(tracer: RecordingTracer) -> None:
    metrics = RecordingMetrics()
    telemetry = PipelineTelemetry(tracer, metrics)

    with telemetry.observe(
        stage="answer",
        query_id="query-1",
        observation_type="generation",
    ):
        pass

    assert metrics.stages[0][0:2] == ("answer", "success")
