"""Golden benchmark and quality-gate result contracts."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from sre_rag.domain.answers import RefusalCode
from sre_rag.domain.base import DomainModel, NonEmptyText

EvaluationMetric = Literal[
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_rate",
]


class GoldenSample(DomainModel):
    """One human-reviewed benchmark case tied to stable chunk IDs."""

    sample_id: NonEmptyText
    question: NonEmptyText
    expected_answer: NonEmptyText
    ground_truth_context: tuple[NonEmptyText, ...] = Field(min_length=1)

    @field_validator("ground_truth_context")
    @classmethod
    def context_ids_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("ground-truth context IDs must be unique")
        return value


class GoldenDataset(DomainModel):
    """Versioned, non-empty benchmark used by local and CI evaluation."""

    schema_version: Literal[1] = 1
    samples: tuple[GoldenSample, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def sample_ids_and_questions_must_be_unique(self) -> "GoldenDataset":
        sample_ids = [sample.sample_id for sample in self.samples]
        questions = [sample.question for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("golden sample IDs must be unique")
        if len(questions) != len(set(questions)):
            raise ValueError("golden sample questions must be unique")
        return self


class EvaluationThresholds(DomainModel):
    """Minimum macro scores required for a release candidate."""

    context_precision: float = Field(default=0.80, ge=0, le=1)
    context_recall: float = Field(default=0.80, ge=0, le=1)
    faithfulness: float = Field(default=0.85, ge=0, le=1)
    answer_rate: float = Field(default=1.0, ge=0, le=1)


class EvaluationScores(DomainModel):
    """Normalized scores produced for one sample or the full benchmark."""

    context_precision: float = Field(ge=0, le=1)
    context_recall: float = Field(ge=0, le=1)
    faithfulness: float = Field(ge=0, le=1)


class SampleEvaluation(DomainModel):
    """Auditable metric output for one golden sample."""

    sample_id: NonEmptyText
    outcome: Literal["answer", "refusal"]
    refusal_code: RefusalCode | None = None
    retrieved_context_ids: tuple[NonEmptyText, ...]
    scores: EvaluationScores

    @model_validator(mode="after")
    def refusal_code_must_match_outcome(self) -> "SampleEvaluation":
        if self.outcome == "refusal" and self.refusal_code is None:
            raise ValueError("refused evaluations require a refusal code")
        if self.outcome == "answer" and self.refusal_code is not None:
            raise ValueError("answered evaluations cannot contain a refusal code")
        return self


class EvaluationReport(DomainModel):
    """Aggregate benchmark result consumed by a later CI quality gate."""

    passed: bool
    thresholds: EvaluationThresholds
    aggregate: EvaluationScores
    answer_rate: float = Field(ge=0, le=1)
    samples: tuple[SampleEvaluation, ...] = Field(min_length=1)
    failed_metrics: tuple[EvaluationMetric, ...]

    @model_validator(mode="after")
    def gate_result_must_match_scores(self) -> "EvaluationReport":
        values = {
            "context_precision": self.aggregate.context_precision,
            "context_recall": self.aggregate.context_recall,
            "faithfulness": self.aggregate.faithfulness,
            "answer_rate": self.answer_rate,
        }
        thresholds = {
            "context_precision": self.thresholds.context_precision,
            "context_recall": self.thresholds.context_recall,
            "faithfulness": self.thresholds.faithfulness,
            "answer_rate": self.thresholds.answer_rate,
        }
        expected = tuple(metric for metric, value in values.items() if value < thresholds[metric])
        if self.failed_metrics != expected:
            raise ValueError("failed metrics must exactly match threshold comparisons")
        if self.passed != (not expected):
            raise ValueError("gate result must match failed metrics")
        return self
