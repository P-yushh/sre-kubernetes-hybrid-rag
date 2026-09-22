"""Tests for deterministic retrieval scores and aggregate quality gates."""

from dataclasses import dataclass

import pytest

from sre_rag.domain.answers import GroundedAnswer, RefusalCode, RefusalResponse
from sre_rag.domain.evaluation import EvaluationThresholds, GoldenDataset, GoldenSample
from sre_rag.domain.retrieval import RetrievalCandidate, RetrievalQuery
from sre_rag.evaluation.runner import EvaluationRunner
from sre_rag.guardrails.service import GuardedAnswerResult
from sre_rag.workflow.graph import RAGExecution
from tests.domain.test_answers import make_citation
from tests.generation.helpers import make_reranked_candidate


@dataclass
class FakeWorkflow:
    execution: RAGExecution
    queries: list[RetrievalQuery]

    def invoke_with_evidence(self, query: RetrievalQuery) -> RAGExecution:
        self.queries.append(query)
        return self.execution


class FakeFaithfulness:
    def __init__(self, score: float) -> None:
        self.result = score
        self.calls: list[dict[str, object]] = []

    def score(
        self,
        *,
        question: str,
        response: str,
        contexts: tuple[str, ...],
    ) -> float:
        self.calls.append({"question": question, "response": response, "contexts": contexts})
        return self.result


def _dataset() -> GoldenDataset:
    return GoldenDataset(
        samples=(
            GoldenSample(
                sample_id="oom",
                question="Why did the container exit with code 137?",
                expected_answer="The memory limit was exceeded.",
                ground_truth_context=("relevant",),
            ),
        )
    )


def _answer_execution(
    contexts: tuple[RetrievalCandidate, ...] | None = None,
) -> RAGExecution:
    answer = GroundedAnswer(
        answer="The container exceeded its memory limit [1].",
        citations=(make_citation(),),
    )
    return RAGExecution(
        result=GuardedAnswerResult(response=answer, telemetry=None),
        contexts=contexts
        or (
            make_reranked_candidate("relevant", "Memory limit evidence.", 1),
            make_reranked_candidate("irrelevant", "Probe evidence.", 2),
        ),
    )


def test_runner_combines_id_metrics_with_ragas_faithfulness() -> None:
    workflow = FakeWorkflow(_answer_execution(), [])
    faithfulness = FakeFaithfulness(0.9)
    runner = EvaluationRunner(
        workflow,
        faithfulness,
        EvaluationThresholds(context_precision=0.5),
    )

    report = runner.run(_dataset())

    assert report.passed
    assert report.aggregate.context_precision == 0.5
    assert report.aggregate.context_recall == 1.0
    assert report.aggregate.faithfulness == 0.9
    assert report.answer_rate == 1.0
    assert report.failed_metrics == ()
    assert workflow.queries[0].text == _dataset().samples[0].question
    assert faithfulness.calls[0]["contexts"] == (
        "Memory limit evidence.",
        "Probe evidence.",
    )


def test_refusal_scores_zero_faithfulness_and_fails_gate() -> None:
    execution = RAGExecution(
        result=GuardedAnswerResult(
            response=RefusalResponse(
                code=RefusalCode.INSUFFICIENT_CONTEXT,
                message="The evidence is insufficient.",
            ),
            telemetry=None,
        ),
        contexts=(),
    )
    faithfulness = FakeFaithfulness(1.0)

    report = EvaluationRunner(FakeWorkflow(execution, []), faithfulness).run(_dataset())

    assert not report.passed
    assert report.aggregate.faithfulness == 0.0
    assert report.answer_rate == 0.0
    assert set(report.failed_metrics) == {
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_rate",
    }
    assert faithfulness.calls == []


def test_runner_rejects_duplicate_retrieved_chunk_ids() -> None:
    duplicate_contexts = (
        make_reranked_candidate("same", "first", 1),
        make_reranked_candidate("same", "second", 2),
    )
    runner = EvaluationRunner(
        FakeWorkflow(_answer_execution(duplicate_contexts), []),
        FakeFaithfulness(1.0),
    )

    with pytest.raises(ValueError, match="unique retrieved"):
        runner.run(_dataset())
