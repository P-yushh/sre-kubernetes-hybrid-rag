"""Golden benchmark execution and deterministic quality-gate reporting."""

from collections.abc import Sequence
from statistics import fmean
from typing import Protocol

from sre_rag.domain.answers import GroundedAnswer, RefusalResponse
from sre_rag.domain.evaluation import (
    EvaluationReport,
    EvaluationScores,
    EvaluationThresholds,
    GoldenDataset,
    GoldenSample,
    SampleEvaluation,
)
from sre_rag.domain.retrieval import RetrievalQuery
from sre_rag.workflow.graph import RAGExecution


class EvaluationWorkflow(Protocol):
    """Run the production workflow while retaining its final evidence set."""

    def invoke_with_evidence(self, query: RetrievalQuery) -> RAGExecution: ...


class FaithfulnessScorer(Protocol):
    """Evaluate semantic support without coupling the runner to Ragas."""

    def score(
        self,
        *,
        question: str,
        response: str,
        contexts: tuple[str, ...],
    ) -> float: ...


class EvaluationRunner:
    """Execute a golden set and produce a deterministic threshold report."""

    def __init__(
        self,
        workflow: EvaluationWorkflow,
        faithfulness: FaithfulnessScorer,
        thresholds: EvaluationThresholds | None = None,
    ) -> None:
        self._workflow = workflow
        self._faithfulness = faithfulness
        self._thresholds = thresholds or EvaluationThresholds()

    def run(self, dataset: GoldenDataset) -> EvaluationReport:
        samples = tuple(self._evaluate_sample(sample) for sample in dataset.samples)
        aggregate = EvaluationScores(
            context_precision=fmean(sample.scores.context_precision for sample in samples),
            context_recall=fmean(sample.scores.context_recall for sample in samples),
            faithfulness=fmean(sample.scores.faithfulness for sample in samples),
        )
        answer_rate = sum(sample.outcome == "answer" for sample in samples) / len(samples)
        failed_metrics = _failed_metrics(aggregate, answer_rate, self._thresholds)
        return EvaluationReport(
            passed=not failed_metrics,
            thresholds=self._thresholds,
            aggregate=aggregate,
            answer_rate=answer_rate,
            samples=samples,
            failed_metrics=failed_metrics,
        )

    def _evaluate_sample(self, sample: GoldenSample) -> SampleEvaluation:
        execution = self._workflow.invoke_with_evidence(RetrievalQuery(text=sample.question))
        retrieved_ids = tuple(candidate.chunk.chunk_id for candidate in execution.contexts)
        _validate_unique_retrieved_ids(retrieved_ids)
        scores = _retrieval_scores(retrieved_ids, sample.ground_truth_context)

        response = execution.result.response
        refusal_code = response.code if isinstance(response, RefusalResponse) else None
        faithfulness = 0.0
        if isinstance(response, GroundedAnswer):
            faithfulness = self._faithfulness.score(
                question=sample.question,
                response=response.answer,
                contexts=tuple(candidate.chunk.text for candidate in execution.contexts),
            )
        return SampleEvaluation(
            sample_id=sample.sample_id,
            outcome=response.outcome,
            refusal_code=refusal_code,
            retrieved_context_ids=retrieved_ids,
            scores=EvaluationScores(
                context_precision=scores.context_precision,
                context_recall=scores.context_recall,
                faithfulness=faithfulness,
            ),
        )


def _retrieval_scores(
    retrieved_ids: Sequence[str],
    reference_ids: Sequence[str],
) -> EvaluationScores:
    overlap = len(set(retrieved_ids) & set(reference_ids))
    precision = overlap / len(retrieved_ids) if retrieved_ids else 0.0
    recall = overlap / len(reference_ids)
    return EvaluationScores(
        context_precision=precision,
        context_recall=recall,
        faithfulness=0.0,
    )


def _validate_unique_retrieved_ids(retrieved_ids: Sequence[str]) -> None:
    if len(retrieved_ids) != len(set(retrieved_ids)):
        raise ValueError("evaluation requires unique retrieved context IDs")


def _failed_metrics(
    scores: EvaluationScores,
    answer_rate: float,
    thresholds: EvaluationThresholds,
) -> tuple[str, ...]:
    values = {
        "context_precision": (scores.context_precision, thresholds.context_precision),
        "context_recall": (scores.context_recall, thresholds.context_recall),
        "faithfulness": (scores.faithfulness, thresholds.faithfulness),
        "answer_rate": (answer_rate, thresholds.answer_rate),
    }
    return tuple(name for name, (value, threshold) in values.items() if value < threshold)
