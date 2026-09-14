"""Tests for deterministic document and chunk identifiers."""

from pathlib import PurePosixPath

import pytest

from sre_rag.domain.documents import Corpus
from sre_rag.ingestion.identifiers import chunk_id, document_id, heading_slug, sha256_text


def test_identifiers_are_readable_and_deterministic() -> None:
    path = PurePosixPath("content/en/docs/tasks/debug/pods.md")
    parent_id = document_id(Corpus.KUBERNETES, path)

    assert parent_id == "kubernetes:content/en/docs/tasks/debug/pods.md"
    assert chunk_id(parent_id, ("Débogage", "Exit Codes"), 2) == (
        "kubernetes:content/en/docs/tasks/debug/pods.md#debogage--exit-codes@2"
    )
    assert sha256_text("OOMKilled") == sha256_text("OOMKilled")
    assert len(sha256_text("OOMKilled")) == 64


def test_root_heading_has_a_stable_slug() -> None:
    assert heading_slug(()) == "root"
    assert heading_slug(("🔥",)) == "root"


@pytest.mark.parametrize("path", [PurePosixPath("/absolute.md"), PurePosixPath("../outside.md")])
def test_document_id_rejects_unsafe_paths(path: PurePosixPath) -> None:
    with pytest.raises(ValueError, match="relative"):
        document_id(Corpus.KUBERNETES, path)


def test_chunk_id_rejects_negative_ordinal() -> None:
    with pytest.raises(ValueError, match="negative"):
        chunk_id("kubernetes:pods.md", ("Pods",), -1)
