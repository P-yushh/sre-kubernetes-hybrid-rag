"""Tests for loading an evaluation runner without hard-wiring runtime composition."""

import json
from pathlib import Path

import pytest

from sre_rag.domain.evaluation import (
    EvaluationReport,
    EvaluationScores,
    EvaluationThresholds,
    GoldenDataset,
    SampleEvaluation,
)
from sre_rag.evaluation.cli import load_runner, main


class InvalidRunner:
    pass


def invalid_runner_factory() -> InvalidRunner:
    return InvalidRunner()


class FixedRunner:
    def __init__(self, report: EvaluationReport) -> None:
        self._report = report

    def run(self, dataset: GoldenDataset) -> EvaluationReport:
        assert dataset.samples[0].sample_id == "oom"
        return self._report


_active_report: EvaluationReport | None = None


def runner_factory() -> FixedRunner:
    assert _active_report is not None
    return FixedRunner(_active_report)


def _report(*, passed: bool) -> EvaluationReport:
    faithfulness = 0.9 if passed else 0.7
    return EvaluationReport(
        passed=passed,
        thresholds=EvaluationThresholds(),
        aggregate=EvaluationScores(
            context_precision=0.9,
            context_recall=0.9,
            faithfulness=faithfulness,
        ),
        answer_rate=1.0,
        samples=(
            SampleEvaluation(
                sample_id="oom",
                outcome="answer",
                retrieved_context_ids=("chunk-1",),
                scores=EvaluationScores(
                    context_precision=0.9,
                    context_recall=0.9,
                    faithfulness=faithfulness,
                ),
            ),
        ),
        failed_metrics=() if passed else ("faithfulness",),
    )


def _write_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "samples": [
                    {
                        "sample_id": "oom",
                        "question": "Why was the container OOMKilled?",
                        "expected_answer": "It exceeded its memory limit.",
                        "ground_truth_context": ["chunk-1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("factory_path", ["missing-separator", ":factory", "module:"])
def test_runner_factory_requires_module_and_attribute(factory_path: str) -> None:
    with pytest.raises(ValueError, match="module:function"):
        load_runner(factory_path)


def test_runner_factory_attribute_must_be_callable() -> None:
    with pytest.raises(ValueError, match="not callable"):
        load_runner("tests.evaluation.test_cli:InvalidRunner.__doc__")


def test_runner_factory_must_return_runner() -> None:
    with pytest.raises(TypeError, match="run method"):
        load_runner("tests.evaluation.test_cli:invalid_runner_factory")


@pytest.mark.parametrize(("passed", "expected_exit_code"), [(True, 0), (False, 1)])
def test_cli_exit_code_matches_quality_gate(
    tmp_path: Path,
    passed: bool,
    expected_exit_code: int,
) -> None:
    global _active_report
    _active_report = _report(passed=passed)
    dataset_path = tmp_path / "golden.json"
    report_path = tmp_path / "report.json"
    _write_dataset(dataset_path)

    exit_code = main(
        [
            "--runner-factory",
            "tests.evaluation.test_cli:runner_factory",
            "--dataset",
            str(dataset_path),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == expected_exit_code
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is passed
