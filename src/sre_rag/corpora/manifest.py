"""Validated manifest for immutable, remotely hosted knowledge corpora."""

from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import Field, HttpUrl, field_validator, model_validator

from sre_rag.domain.base import DomainModel, NonEmptyText, Revision, Sha256
from sre_rag.domain.documents import Corpus


class CorpusSource(DomainModel):
    """One pinned archive and the Markdown subset allowed into the index."""

    name: NonEmptyText
    corpus: Corpus
    repository: NonEmptyText
    revision: Revision
    archive_url: HttpUrl
    content_sha256: Sha256
    license: NonEmptyText
    license_url: HttpUrl
    attribution: NonEmptyText
    include_paths: tuple[PurePosixPath, ...] = Field(min_length=1)
    expected_markdown_files: int = Field(gt=0)
    max_archive_bytes: int = Field(gt=0)
    max_extracted_bytes: int = Field(gt=0)

    @field_validator("include_paths")
    @classmethod
    def include_paths_must_be_safe(
        cls,
        paths: tuple[PurePosixPath, ...],
    ) -> tuple[PurePosixPath, ...]:
        normalized: list[PurePosixPath] = []
        for path in paths:
            if path.is_absolute() or not path.parts or ".." in path.parts:
                raise ValueError("include paths must be non-empty relative paths without '..'")
            normalized.append(PurePosixPath(*path.parts))
        if len(normalized) != len(set(normalized)):
            raise ValueError("include paths must be unique")
        return tuple(normalized)


class CorpusManifest(DomainModel):
    """Versioned collection of corpus sources used for one index build."""

    schema_version: Literal[1]
    sources: tuple[CorpusSource, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def sources_must_be_unique(self) -> "CorpusManifest":
        names = [source.name for source in self.sources]
        corpora = [source.corpus for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("corpus source names must be unique")
        if len(corpora) != len(set(corpora)):
            raise ValueError("corpus types must be unique")
        return self


def load_manifest(path: Path) -> CorpusManifest:
    """Load YAML through the strict immutable manifest contract."""

    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return CorpusManifest.model_validate(payload)
