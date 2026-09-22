"""Collections-based Ragas faithfulness adapter."""

import math
from dataclasses import dataclass
from typing import Protocol


class _MetricResult(Protocol):
    @property
    def value(self) -> object: ...


class _FaithfulnessMetric(Protocol):
    def score(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
    ) -> _MetricResult: ...


@dataclass(frozen=True, slots=True)
class RagasConfig:
    """Hosted evaluator settings kept independent from the generation model."""

    model: str = "gpt-5.6-terra"
    timeout_seconds: float = 60.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("evaluation model must not be blank")
        if self.timeout_seconds <= 0:
            raise ValueError("evaluation timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("evaluation retries must be non-negative")


class RagasFaithfulnessScorer:
    """Score whether answer claims are supported by retrieved text."""

    def __init__(self, metric: _FaithfulnessMetric) -> None:
        self._metric = metric

    @classmethod
    def from_openai(
        cls,
        api_key: str,
        config: RagasConfig | None = None,
    ) -> "RagasFaithfulnessScorer":
        """Create the modern collections metric from an explicit API secret."""

        if not api_key.strip():
            raise ValueError("OpenAI API key must not be blank")
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import Faithfulness

        resolved = config or RagasConfig()
        client = AsyncOpenAI(
            api_key=api_key,
            timeout=resolved.timeout_seconds,
            max_retries=resolved.max_retries,
        )
        metric = Faithfulness(llm=llm_factory(resolved.model, client=client))
        return cls(metric)

    def score(
        self,
        *,
        question: str,
        response: str,
        contexts: tuple[str, ...],
    ) -> float:
        result = self._metric.score(
            user_input=question,
            response=response,
            retrieved_contexts=list(contexts),
        )
        value = result.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Ragas faithfulness result must be numeric")
        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Ragas faithfulness result must be between zero and one")
        return score
