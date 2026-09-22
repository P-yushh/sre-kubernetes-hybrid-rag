"""Tests for strict golden dataset loading."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sre_rag.evaluation.dataset import load_golden_dataset


def test_loader_reads_valid_versioned_dataset(tmp_path: Path) -> None:
    path = tmp_path / "golden_dataset.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "samples": [
                    {
                        "sample_id": "oom",
                        "question": "Why OOMKilled?",
                        "expected_answer": "The memory limit was exceeded.",
                        "ground_truth_context": ["chunk-1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_golden_dataset(path)

    assert dataset.samples[0].sample_id == "oom"


def test_loader_rejects_empty_placeholder_dataset(tmp_path: Path) -> None:
    path = tmp_path / "golden_dataset.json"
    path.write_text('{"schema_version":1,"samples":[]}', encoding="utf-8")

    with pytest.raises(ValidationError):
        load_golden_dataset(path)
