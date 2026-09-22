"""Tests for persisted evaluation evidence and process gate semantics."""

import json
from dataclasses import dataclass
from pathlib import Path

from sre_rag.domain.evaluation import (
    EvaluationReport,
    EvaluationScores,
    EvaluationThresholds,
    GoldenDataset,
    SampleEvaluation,
)
from sre_rag.evaluation.gate import (
    quality_gate_exit_code,
    render_evaluation_summary,
    run_quality_gate,
)


@dataclass
class FakeRunner:
    report: EvaluationReport
    datasets: list[GoldenDataset]

    def run(self, dataset: GoldenDataset) -> EvaluationReport:
        self.datasets.append(dataset)
        return self.report


def _report(*, passed: bool = True) -> EvaluationReport:
    failed_metrics = () if passed else ("faithfulness",)
    return EvaluationReport(
        passed=passed,
        thresholds=EvaluationThresholds(),
        aggregate=EvaluationScores(
            context_precision=0.9,
            context_recall=0.8,
            faithfulness=0.9 if passed else 0.7,
        ),
        answer_rate=1.0,
        samples=(
            SampleEvaluation(
                sample_id="oom",
                outcome="answer",
                retrieved_context_ids=("chunk-1",),
                scores=EvaluationScores(
                    context_precision=0.9,
                    context_recall=0.8,
                    faithfulness=0.9 if passed else 0.7,
                ),
            ),
        ),
        failed_metrics=failed_metrics,
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


def test_quality_gate_persists_json_and_appends_summary(tmp_path: Path) -> None:
    dataset_path = tmp_path / "golden.json"
    report_path = tmp_path / "nested" / "report.json"
    summary_path = tmp_path / "summary.md"
    _write_dataset(dataset_path)
    summary_path.write_text("existing summary\n", encoding="utf-8")
    runner = FakeRunner(_report(), [])

    report = run_quality_gate(
        runner,
        dataset_path=dataset_path,
        report_path=report_path,
        summary_path=summary_path,
    )

    assert report.passed
    assert runner.datasets[0].samples[0].sample_id == "oom"
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True
    summary = summary_path.read_text(encoding="utf-8")
    assert summary.startswith("existing summary\n")
    assert "| Faithfulness | 0.900 | 0.850 | PASS |" in summary
    assert "**Overall: PASS**" in summary


def test_failed_report_lists_regressed_metrics() -> None:
    summary = render_evaluation_summary(_report(passed=False))

    assert "| Faithfulness | 0.700 | 0.850 | FAIL |" in summary
    assert "**Overall: FAIL**" in summary
    assert "Failed metrics: faithfulness" in summary
    assert quality_gate_exit_code(_report(passed=False)) == 1


def test_passing_report_returns_success_exit_code() -> None:
    assert quality_gate_exit_code(_report()) == 0
