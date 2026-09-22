"""Tests for the collections-based Ragas faithfulness adapter."""

from types import SimpleNamespace

import pytest

from sre_rag.evaluation.ragas import RagasConfig, RagasFaithfulnessScorer


class FakeMetric:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[dict[str, object]] = []

    def score(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value)


def test_ragas_adapter_uses_direct_collections_arguments() -> None:
    metric = FakeMetric(0.9)
    scorer = RagasFaithfulnessScorer(metric)

    score = scorer.score(
        question="Why OOMKilled?",
        response="The memory limit was exceeded [1].",
        contexts=("The container exceeded its memory limit.",),
    )

    assert score == 0.9
    assert metric.calls == [
        {
            "user_input": "Why OOMKilled?",
            "response": "The memory limit was exceeded [1].",
            "retrieved_contexts": ["The container exceeded its memory limit."],
        }
    ]


@pytest.mark.parametrize("value", [True, "0.9", float("nan"), -0.1, 1.1])
def test_ragas_adapter_rejects_invalid_metric_results(value: object) -> None:
    with pytest.raises(ValueError, match="faithfulness result"):
        RagasFaithfulnessScorer(FakeMetric(value)).score(
            question="question",
            response="response",
            contexts=("context",),
        )


def test_ragas_config_rejects_blank_model() -> None:
    with pytest.raises(ValueError):
        RagasConfig(model=" ")


def test_ragas_config_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError):
        RagasConfig(timeout_seconds=0)


def test_ragas_config_rejects_negative_retries() -> None:
    with pytest.raises(ValueError):
        RagasConfig(max_retries=-1)


def test_ragas_factory_rejects_blank_api_key_before_client_creation() -> None:
    with pytest.raises(ValueError, match="API key"):
        RagasFaithfulnessScorer.from_openai(" ")
