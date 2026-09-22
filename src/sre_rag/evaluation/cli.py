"""Command-line entry point for the live golden-set quality gate."""

from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from sre_rag.evaluation.gate import (
    ReportRunner,
    quality_gate_exit_code,
    render_evaluation_summary,
    run_quality_gate,
)

RunnerFactory = Callable[[], ReportRunner]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured evaluator and return a process-friendly gate status."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    factory_path = arguments.runner_factory or os.getenv("SRE_RAG_EVALUATION_RUNNER_FACTORY")
    if not factory_path:
        parser.error(
            "provide --runner-factory or set SRE_RAG_EVALUATION_RUNNER_FACTORY to a module:function"
        )

    runner = load_runner(factory_path)
    report = run_quality_gate(
        runner,
        dataset_path=arguments.dataset,
        report_path=arguments.report,
        summary_path=arguments.summary,
    )
    print(render_evaluation_summary(report), end="")
    return quality_gate_exit_code(report)


def load_runner(factory_path: str) -> ReportRunner:
    """Load a repository-controlled runner factory from ``module:function``."""

    module_name, separator, attribute_name = factory_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("runner factory must use module:function syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise ValueError(f"runner factory is not callable: {factory_path}")
    runner = cast(RunnerFactory, factory)()
    if not callable(getattr(runner, "run", None)):
        raise TypeError("runner factory must return an object with a run method")
    return runner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the golden RAG dataset and fail when a threshold regresses."
    )
    parser.add_argument(
        "--runner-factory",
        help="Import path of a zero-argument EvaluationRunner factory (module:function).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/golden_dataset.json"),
        help="Versioned golden dataset path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/evaluation-report.json"),
        help="Machine-readable report destination.",
    )
    github_summary = os.getenv("GITHUB_STEP_SUMMARY")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(github_summary) if github_summary else None,
        help="Optional Markdown summary destination.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
