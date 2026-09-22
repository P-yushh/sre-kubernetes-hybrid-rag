"""Tests for golden benchmark and quality-report contracts."""

import pytest
from pydantic import ValidationError

from sre_rag.domain.evaluation import (
    EvaluationReport,
    EvaluationScores,
    EvaluationThresholds,
    GoldenDataset,
    GoldenSample,
    SampleEvaluation,
)


def _sample(**updates: object) -> GoldenSample:
    values: dict[str, object] = {
        "sample_id": "oom-exit-code",
        "question": "Why did the container exit with code 137?",
        "expected_answer": "The container exceeded its memory limit.",
        "ground_truth_context": ("kubernetes:oom#exit-code@0",),
    }
    values.update(updates)
    return GoldenSample.model_validate(values)


def test_golden_dataset_accepts_versioned_human_reviewed_samples() -> None:
    dataset = GoldenDataset(samples=(_sample(),))

    assert dataset.schema_version == 1
    assert dataset.samples[0].ground_truth_context == ("kubernetes:oom#exit-code@0",)


def test_golden_sample_rejects_duplicate_context_ids() -> None:
    with pytest.raises(ValidationError, match="context IDs must be unique"):
        _sample(ground_truth_context=("same", "same"))


@pytest.mark.parametrize("field", ["sample_id", "question"])
def test_golden_dataset_rejects_duplicate_identity_fields(field: str) -> None:
    first = _sample()
    second_values = {
        "sample_id": "probe-failure",
        "question": "Why is the readiness probe failing?",
    }
    second_values[field] = getattr(first, field)
    second = _sample(**second_values)

    with pytest.raises(ValidationError, match="must be unique"):
        GoldenDataset(samples=(first, second))


def test_golden_dataset_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        GoldenDataset(samples=())


def test_evaluation_report_cannot_claim_a_false_pass() -> None:
    with pytest.raises(ValidationError, match="failed metrics"):
        EvaluationReport(
            passed=True,
            thresholds=EvaluationThresholds(),
            aggregate=EvaluationScores(
                context_precision=0.5,
                context_recall=1.0,
                faithfulness=1.0,
            ),
            answer_rate=1.0,
            samples=(
                SampleEvaluation(
                    sample_id="oom",
                    outcome="answer",
                    retrieved_context_ids=("chunk",),
                    scores=EvaluationScores(
                        context_precision=0.5,
                        context_recall=1.0,
                        faithfulness=1.0,
                    ),
                ),
            ),
            failed_metrics=(),
        )
