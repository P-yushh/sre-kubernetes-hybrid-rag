"""Tests for metadata-only Langfuse observations."""

from contextlib import AbstractContextManager
from types import TracebackType

from sre_rag.observability.langfuse import LangfuseTraceProvider


class FakeObservation:
    def __init__(self, *, fail_update: bool = False) -> None:
        self.fail_update = fail_update
        self.updates: list[dict[str, object]] = []

    def update(self, **kwargs: object) -> None:
        if self.fail_update:
            raise RuntimeError("telemetry unavailable")
        self.updates.append(kwargs)


class FakeContext(AbstractContextManager[object]):
    def __init__(self, observation: FakeObservation) -> None:
        self.observation = observation
        self.exited = False

    def __enter__(self) -> object:
        return self.observation

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.exited = True


class FakeClient:
    def __init__(self, observation: FakeObservation | None = None) -> None:
        self.observation = observation or FakeObservation()
        self.calls: list[dict[str, object]] = []
        self.context = FakeContext(self.observation)

    def start_as_current_observation(self, **kwargs: object) -> AbstractContextManager[object]:
        self.calls.append(kwargs)
        return self.context


def test_langfuse_provider_sends_metadata_without_pipeline_content() -> None:
    client = FakeClient()
    provider = LangfuseTraceProvider(client)

    with provider.observe(
        name="rag.answer",
        observation_type="generation",
        metadata={"query_id": "query-1", "context_count": 3},
    ) as observation:
        observation.update(
            output={"outcome": "answer", "provider": "openai"},
            model="test-model",
            usage_details={"input": 20, "output": 10},
        )

    assert client.calls == [
        {
            "name": "rag.answer",
            "as_type": "generation",
            "metadata": {"query_id": "query-1", "context_count": 3},
        }
    ]
    assert client.observation.updates == [
        {
            "output": {"outcome": "answer", "provider": "openai"},
            "model": "test-model",
            "usage_details": {"input": 20, "output": 10},
        }
    ]
    serialized = repr((client.calls, client.observation.updates))
    assert "question" not in serialized
    assert "chunk" not in serialized
    assert "prompt" not in serialized
    assert client.context.exited


def test_observation_update_failure_does_not_break_application() -> None:
    provider = LangfuseTraceProvider(FakeClient(FakeObservation(fail_update=True)))

    with provider.observe(
        name="rag.answer",
        observation_type="generation",
        metadata={"query_id": "query-1"},
    ) as observation:
        observation.update(output={"outcome": "answer"})
