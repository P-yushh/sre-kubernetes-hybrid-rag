"""Fail-closed quality-gate execution and auditable report artifacts."""

from pathlib import Path
from typing import Protocol

from sre_rag.domain.evaluation import EvaluationReport, GoldenDataset
from sre_rag.evaluation.dataset import load_golden_dataset


class ReportRunner(Protocol):
    """Minimal evaluation runner surface required by the command line gate."""

    def run(self, dataset: GoldenDataset) -> EvaluationReport: ...


def run_quality_gate(
    runner: ReportRunner,
    *,
    dataset_path: Path,
    report_path: Path,
    summary_path: Path | None = None,
) -> EvaluationReport:
    """Evaluate a golden set, persist evidence, and return the gated report."""

    dataset = load_golden_dataset(dataset_path)
    report = runner.run(dataset)
    _write_report(report_path, report)
    if summary_path is not None:
        _append_summary(summary_path, render_evaluation_summary(report))
    return report


def render_evaluation_summary(report: EvaluationReport) -> str:
    """Render a compact GitHub-compatible summary without source content."""

    metrics = (
        (
            "Context precision",
            report.aggregate.context_precision,
            report.thresholds.context_precision,
        ),
        ("Context recall", report.aggregate.context_recall, report.thresholds.context_recall),
        ("Faithfulness", report.aggregate.faithfulness, report.thresholds.faithfulness),
        ("Answer rate", report.answer_rate, report.thresholds.answer_rate),
    )
    rows = [
        "## RAG quality gate",
        "",
        "| Metric | Score | Required | Result |",
        "| --- | ---: | ---: | :---: |",
    ]
    rows.extend(
        f"| {name} | {score:.3f} | {threshold:.3f} | {'PASS' if score >= threshold else 'FAIL'} |"
        for name, score, threshold in metrics
    )
    rows.extend(
        (
            "",
            f"**Overall: {'PASS' if report.passed else 'FAIL'}**",
            "",
            f"Samples evaluated: {len(report.samples)}",
        )
    )
    if report.failed_metrics:
        rows.extend(("", f"Failed metrics: {', '.join(report.failed_metrics)}"))
    return "\n".join(rows) + "\n"


def quality_gate_exit_code(report: EvaluationReport) -> int:
    """Map the validated aggregate decision to shell success or failure."""

    return 0 if report.passed else 1


def _write_report(path: Path, report: EvaluationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _append_summary(path: Path, summary: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(summary)
