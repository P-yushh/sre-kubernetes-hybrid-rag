"""Manual Langfuse spans that exclude questions and source content."""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Protocol, cast

from sre_rag.observability.base import (
    ActiveObservation,
    AttributeValue,
    ObservationType,
)


class _LangfuseClient(Protocol):
    def start_as_current_observation(self, **kwargs: object) -> AbstractContextManager[object]: ...


class _LangfuseObservation(Protocol):
    def update(self, **kwargs: object) -> object: ...


class _ObservationAdapter:
    def __init__(self, observation: object) -> None:
        self._observation = cast(_LangfuseObservation, observation)

    def update(
        self,
        *,
        output: Mapping[str, AttributeValue] | None = None,
        model: str | None = None,
        usage_details: Mapping[str, int] | None = None,
    ) -> None:
        values: dict[str, object] = {}
        if output is not None:
            values["output"] = dict(output)
        if model is not None:
            values["model"] = model
        if usage_details is not None:
            values["usage_details"] = dict(usage_details)
        try:
            self._observation.update(**values)
        except Exception:
            # Telemetry must never turn a successful query into a service failure.
            return


class _ObservationContext(AbstractContextManager[ActiveObservation]):
    def __init__(self, context: AbstractContextManager[object]) -> None:
        self._context = context

    def __enter__(self) -> ActiveObservation:
        return _ObservationAdapter(self._context.__enter__())

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self._context.__exit__(exc_type, exc_value, traceback)


class LangfuseTraceProvider:
    """Create typed Langfuse observations containing metadata only."""

    def __init__(self, client: _LangfuseClient) -> None:
        self._client = client

    @classmethod
    def from_environment(cls) -> "LangfuseTraceProvider":
        """Use standard ``LANGFUSE_*`` environment variables for credentials."""

        from langfuse import get_client

        return cls(cast(_LangfuseClient, get_client()))

    def observe(
        self,
        *,
        name: str,
        observation_type: ObservationType,
        metadata: Mapping[str, AttributeValue],
    ) -> AbstractContextManager[ActiveObservation]:
        context = self._client.start_as_current_observation(
            name=name,
            as_type=observation_type,
            metadata=dict(metadata),
        )
        return _ObservationContext(context)
