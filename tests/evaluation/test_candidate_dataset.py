"""Contract checks for the source-grounded benchmark review candidate."""

from pathlib import Path

from sre_rag.domain.evaluation import GoldenDataset


def test_candidate_benchmark_has_required_size_and_corpus_mix() -> None:
    dataset = GoldenDataset.model_validate_json(
        Path("data/golden_dataset.candidate.json").read_text(encoding="utf-8")
    )
    context_ids = tuple(sample.ground_truth_context[0] for sample in dataset.samples)

    assert len(dataset.samples) == 30
    assert sum(chunk_id.startswith("kubernetes:") for chunk_id in context_ids) == 20
    assert sum(chunk_id.startswith("post_mortems:") for chunk_id in context_ids) == 10
