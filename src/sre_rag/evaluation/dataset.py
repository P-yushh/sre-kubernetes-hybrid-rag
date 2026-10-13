"""Strict loading of the versioned golden benchmark."""

from collections.abc import Collection
from pathlib import Path

from sre_rag.domain.evaluation import GoldenDataset


def load_golden_dataset(path: Path) -> GoldenDataset:
    """Load a non-empty benchmark and validate every sample before evaluation."""

    return GoldenDataset.model_validate_json(path.read_text(encoding="utf-8"))


def validate_context_ids(dataset: GoldenDataset, available_ids: Collection[str]) -> None:
    """Reject benchmark references that are absent from the built corpus artifact."""

    missing = sorted(
        {
            context_id
            for sample in dataset.samples
            for context_id in sample.ground_truth_context
            if context_id not in available_ids
        }
    )
    if missing:
        rendered = "\n".join(f"- {context_id}" for context_id in missing)
        raise ValueError(f"golden dataset references missing chunk IDs:\n{rendered}")
