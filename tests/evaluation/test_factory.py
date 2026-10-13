"""Tests for trusted evaluation composition and resource ownership."""

import pytest

import sre_rag.evaluation.factory as factory_module
from sre_rag.config import Settings
from sre_rag.domain.evaluation import GoldenDataset
from sre_rag.evaluation.factory import ManagedEvaluationRunner


class StubRuntime:
    workflow = object()

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FailingRunner:
    def run(self, dataset: GoldenDataset) -> None:
        del dataset
        raise RuntimeError("evaluation failed")


def test_managed_runner_closes_runtime_when_evaluation_fails() -> None:
    runtime = StubRuntime()
    runner = ManagedEvaluationRunner(FailingRunner(), runtime)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="evaluation failed"):
        runner.run(GoldenDataset.model_validate({"samples": [_sample()]}))

    assert runtime.closed is True


def test_factory_requires_evaluation_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        factory_module,
        "get_settings",
        lambda: Settings(environment="test", openai_api_key=None),
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        factory_module.create_runner()


def _sample() -> dict[str, object]:
    return {
        "sample_id": "sample",
        "question": "question",
        "expected_answer": "answer",
        "ground_truth_context": ["chunk"],
    }
