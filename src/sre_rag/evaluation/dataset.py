"""Strict loading of the versioned golden benchmark."""

from pathlib import Path

from sre_rag.domain.evaluation import GoldenDataset


def load_golden_dataset(path: Path) -> GoldenDataset:
    """Load a non-empty benchmark and validate every sample before evaluation."""

    return GoldenDataset.model_validate_json(path.read_text(encoding="utf-8"))
