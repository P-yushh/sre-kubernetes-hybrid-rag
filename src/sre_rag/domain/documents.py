"""Source-document and chunk contracts."""

from enum import StrEnum

from pydantic import Field, HttpUrl, field_validator

from sre_rag.domain.base import DomainModel, NonEmptyText, Revision, Sha256


class Corpus(StrEnum):
    """Knowledge corpora supported by the ingestion pipeline."""

    KUBERNETES = "kubernetes"
    POST_MORTEMS = "post_mortems"


class SourceProvenance(DomainModel):
    """Immutable information needed to reproduce and cite source content."""

    corpus: Corpus
    repository: NonEmptyText
    revision: Revision
    relative_path: NonEmptyText
    source_url: HttpUrl
    license: NonEmptyText


class SourceDocument(DomainModel):
    """One normalized source document before chunking."""

    document_id: NonEmptyText
    title: NonEmptyText
    content: str
    content_sha256: Sha256
    provenance: SourceProvenance

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        """Reject empty documents without modifying meaningful whitespace."""

        if not value.strip():
            raise ValueError("document content must not be blank")
        return value


class Chunk(DomainModel):
    """A citable, ordered section of a normalized source document."""

    chunk_id: NonEmptyText
    document_id: NonEmptyText
    text: str
    heading_path: tuple[NonEmptyText, ...] = ()
    ordinal: int = Field(ge=0)
    token_count: int = Field(gt=0)
    content_sha256: Sha256
    provenance: SourceProvenance

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject empty chunks while preserving flags, YAML, and whitespace."""

        if not value.strip():
            raise ValueError("chunk text must not be blank")
        return value
