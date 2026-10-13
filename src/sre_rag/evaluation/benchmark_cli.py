"""Validate benchmark references against one immutable index artifact."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sre_rag.evaluation.dataset import load_golden_dataset, validate_context_ids
from sre_rag.indexing.artifacts import ChunkArtifactStore


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    dataset = load_golden_dataset(arguments.dataset)
    chunks = ChunkArtifactStore().load(
        chunks_path=arguments.chunks,
        metadata_path=arguments.metadata,
    )
    validate_context_ids(dataset, {chunk.chunk_id for chunk in chunks})
    print(
        f"Validated {len(dataset.samples)} benchmark samples against {len(chunks)} indexed chunks."
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate golden benchmark chunk references")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/index-manifest.json"),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
