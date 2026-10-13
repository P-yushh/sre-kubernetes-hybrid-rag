"""Tests for benchmark-to-index reference validation."""

import json
from pathlib import Path

import pytest

from sre_rag.domain.documents import Corpus
from sre_rag.evaluation.benchmark_cli import main
from sre_rag.indexing.artifacts import ChunkArtifactStore, IndexedSource
from tests.domain.factories import make_chunk


def test_benchmark_command_validates_built_chunk_ids(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chunk = make_chunk()
    chunks_path = tmp_path / "chunks.jsonl"
    metadata_path = tmp_path / "index-manifest.json"
    dataset_path = tmp_path / "candidate.json"
    ChunkArtifactStore().write(
        (chunk,),
        chunks_path=chunks_path,
        metadata_path=metadata_path,
        corpus_manifest_sha256="a" * 64,
        dense_collection="chunks_aaaaaaaaaaaa",
        sources=(
            IndexedSource(
                name="kubernetes-docs",
                corpus=Corpus.KUBERNETES,
                revision="b" * 40,
                content_sha256="c" * 64,
                license="CC-BY-4.0",
                document_count=1,
                chunk_count=1,
            ),
        ),
    )
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "samples": [
                    {
                        "sample_id": "sample",
                        "question": "question",
                        "expected_answer": "answer",
                        "ground_truth_context": [chunk.chunk_id],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--dataset",
            str(dataset_path),
            "--chunks",
            str(chunks_path),
            "--metadata",
            str(metadata_path),
        ]
    )

    assert exit_code == 0
    assert "Validated 1 benchmark samples against 1 indexed chunks" in capsys.readouterr().out
