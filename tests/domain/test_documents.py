"""Tests for source-document and chunk contracts."""

import pytest
from pydantic import ValidationError

from sre_rag.domain.documents import Chunk, SourceDocument
from tests.domain.factories import CONTENT_HASH, make_chunk, make_provenance


def test_source_document_preserves_exact_content() -> None:
    content = "Run `kube-apiserver --enable-admission-plugins=NodeRestriction`.\n"

    document = SourceDocument(
        document_id="kubernetes:admission-controllers",
        title="Admission Controllers",
        content=content,
        content_sha256=CONTENT_HASH,
        provenance=make_provenance(),
    )

    assert document.content == content
    assert str(document.provenance.source_url).startswith("https://github.com/kubernetes/")


@pytest.mark.parametrize("field", ["content", "text"])
def test_document_text_fields_reject_blank_values(field: str) -> None:
    values: dict[str, object]
    model: type[SourceDocument] | type[Chunk]

    if field == "content":
        model = SourceDocument
        values = {
            "document_id": "kubernetes:blank",
            "title": "Blank",
            "content": "  \n",
            "content_sha256": CONTENT_HASH,
            "provenance": make_provenance(),
        }
    else:
        model = Chunk
        values = make_chunk().model_dump()
        values["text"] = "\t"

    with pytest.raises(ValidationError):
        model.model_validate(values)


def test_chunk_requires_a_sha256_hash_and_positive_token_count() -> None:
    values = make_chunk().model_dump()
    values.update(content_sha256="not-a-hash", token_count=0)

    with pytest.raises(ValidationError) as error:
        Chunk.model_validate(values)

    assert error.value.error_count() == 2


def test_chunk_is_immutable() -> None:
    chunk = make_chunk()

    with pytest.raises(ValidationError):
        chunk.ordinal = 2  # type: ignore[misc]
