"""Tests for strict versioned corpus manifests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from sre_rag.corpora.manifest import CorpusManifest, CorpusSource, load_manifest
from sre_rag.domain.documents import Corpus

ROOT = Path(__file__).parents[2]


def source(**updates: object) -> CorpusSource:
    values: dict[str, object] = {
        "name": "kubernetes-docs",
        "corpus": Corpus.KUBERNETES,
        "repository": "kubernetes/website",
        "revision": "a" * 40,
        "archive_url": f"https://github.com/kubernetes/website/archive/{'a' * 40}.tar.gz",
        "content_sha256": "b" * 64,
        "license": "CC-BY-4.0",
        "license_url": "https://github.com/kubernetes/website/blob/main/LICENSE",
        "attribution": "Kubernetes documentation contributors",
        "include_paths": ("content/en/docs/concepts", "content/en/docs/tasks"),
        "expected_markdown_files": 10,
        "max_archive_bytes": 1000,
        "max_extracted_bytes": 500,
    }
    values.update(updates)
    return CorpusSource.model_validate(values)


def test_manifest_loads_yaml_into_immutable_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""schema_version: 1
sources:
  - name: kubernetes-docs
    corpus: kubernetes
    repository: kubernetes/website
    revision: {"a" * 40}
    archive_url: https://github.com/kubernetes/website/archive/{"a" * 40}.tar.gz
    content_sha256: {"b" * 64}
    license: CC-BY-4.0
    license_url: https://github.com/kubernetes/website/blob/main/LICENSE
    attribution: Kubernetes documentation contributors
    include_paths: [content/en/docs/concepts]
    expected_markdown_files: 10
    max_archive_bytes: 1000
    max_extracted_bytes: 500
""",
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)

    assert manifest.sources[0].corpus is Corpus.KUBERNETES
    assert manifest.sources[0].include_paths[0].as_posix() == "content/en/docs/concepts"


def test_repository_manifest_pins_reviewed_source_revisions() -> None:
    manifest = load_manifest(ROOT / "corpora/manifest.yaml")

    assert [source.corpus for source in manifest.sources] == [
        Corpus.KUBERNETES,
        Corpus.POST_MORTEMS,
    ]
    assert all(len(source.revision) == 40 for source in manifest.sources)
    assert [source.expected_markdown_files for source in manifest.sources] == [407, 242]
    assert [source.license for source in manifest.sources] == ["CC-BY-4.0", "GPL-3.0"]


@pytest.mark.parametrize("include_path", ["", "/absolute", "../outside", "safe/../../outside"])
def test_source_rejects_unsafe_include_paths(include_path: str) -> None:
    with pytest.raises(ValidationError, match="include paths"):
        source(include_paths=(include_path,))


def test_source_rejects_duplicate_include_paths() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        source(include_paths=("docs", "docs"))


@pytest.mark.parametrize(
    "field", ["expected_markdown_files", "max_archive_bytes", "max_extracted_bytes"]
)
def test_source_rejects_non_positive_limits(field: str) -> None:
    with pytest.raises(ValidationError):
        source(**{field: 0})


def test_manifest_rejects_duplicate_names_and_corpus_types() -> None:
    first = source()

    with pytest.raises(ValidationError, match="source names"):
        CorpusManifest(schema_version=1, sources=(first, first))

    with pytest.raises(ValidationError, match="corpus types"):
        CorpusManifest(
            schema_version=1,
            sources=(first, source(name="other", repository="other/repository")),
        )
