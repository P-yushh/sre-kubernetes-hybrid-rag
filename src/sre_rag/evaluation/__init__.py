"""Golden-set evaluation and Ragas metric adapters."""

from sre_rag.evaluation.dataset import load_golden_dataset
from sre_rag.evaluation.gate import render_evaluation_summary, run_quality_gate
from sre_rag.evaluation.ragas import RagasFaithfulnessScorer
from sre_rag.evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationRunner",
    "RagasFaithfulnessScorer",
    "load_golden_dataset",
    "render_evaluation_summary",
    "run_quality_gate",
]
