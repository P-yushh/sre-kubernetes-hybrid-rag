"""Repository-controlled factory for the trusted live quality gate."""

from dataclasses import dataclass

from sre_rag.config import get_settings
from sre_rag.domain.evaluation import EvaluationReport, GoldenDataset
from sre_rag.evaluation.ragas import RagasConfig, RagasFaithfulnessScorer
from sre_rag.evaluation.runner import EvaluationRunner
from sre_rag.runtime import RuntimeBundle, build_runtime


@dataclass(slots=True)
class ManagedEvaluationRunner:
    """Close runtime clients after the one-shot benchmark process completes."""

    runner: EvaluationRunner
    runtime: RuntimeBundle

    def run(self, dataset: GoldenDataset) -> EvaluationReport:
        try:
            return self.runner.run(dataset)
        finally:
            self.runtime.close()


def create_runner() -> ManagedEvaluationRunner:
    """Compose production retrieval/generation with an independent Ragas judge."""

    settings = get_settings()
    if settings.openai_api_key is None:
        raise ValueError("SRE_RAG_OPENAI_API_KEY is required for live evaluation")
    runtime = build_runtime(settings)
    try:
        faithfulness = RagasFaithfulnessScorer.from_openai(
            settings.openai_api_key.get_secret_value(),
            RagasConfig(
                model=settings.openai_evaluation_model,
                timeout_seconds=settings.generation_timeout_seconds,
            ),
        )
        runner = EvaluationRunner(runtime.workflow, faithfulness)
    except BaseException:
        runtime.close()
        raise
    return ManagedEvaluationRunner(runner, runtime)
