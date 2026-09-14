"""Tests for safe, deterministic corpus loading."""

from pathlib import Path, PurePosixPath

import pytest

from sre_rag.domain.documents import Corpus
from sre_rag.ingestion.loaders import CorpusConfig, MarkdownCorpusLoader
from sre_rag.ingestion.markdown import MarkdownNormalizer
from tests.ingestion.helpers import PassThroughConverter

REVISION = "c" * 40


def make_config(root: Path, *include_paths: str, max_size: int = 1024) -> CorpusConfig:
    return CorpusConfig(
        root=root,
        corpus=Corpus.KUBERNETES,
        repository="kubernetes/website",
        revision=REVISION,
        license="CC-BY-4.0",
        include_paths=tuple(PurePosixPath(path) for path in include_paths),
        max_file_size_bytes=max_size,
    )


def test_loader_discovers_allowed_markdown_in_stable_order(tmp_path: Path) -> None:
    concepts = tmp_path / "content/en/docs/concepts"
    tasks = tmp_path / "content/en/docs/tasks"
    concepts.mkdir(parents=True)
    tasks.mkdir(parents=True)
    (concepts / "z-pods.md").write_text("# Pods\nPod text.", encoding="utf-8")
    (concepts / "a-services.md").write_text(
        "---\ntitle: Services\n---\nService text.", encoding="utf-8"
    )
    (concepts / "ignore.txt").write_text("not markdown", encoding="utf-8")
    (tasks / ".hidden.md").write_text("hidden", encoding="utf-8")

    loader = MarkdownCorpusLoader(MarkdownNormalizer(PassThroughConverter()))
    documents = loader.load(
        make_config(tmp_path, "content/en/docs/concepts", "content/en/docs/tasks")
    )

    assert [document.title for document in documents] == ["Services", "Pods"]
    assert documents[0].document_id.endswith("content/en/docs/concepts/a-services.md")
    assert str(documents[0].provenance.source_url) == (
        f"https://github.com/kubernetes/website/blob/{REVISION}/"
        "content/en/docs/concepts/a-services.md"
    )
    assert len(documents[0].content_sha256) == 64


def test_loader_accepts_a_filtered_reference_file(tmp_path: Path) -> None:
    reference = tmp_path / "content/en/docs/reference/admission controllers.md"
    reference.parent.mkdir(parents=True)
    reference.write_text("# Admission\n--enable-admission-plugins", encoding="utf-8")
    loader = MarkdownCorpusLoader(MarkdownNormalizer(PassThroughConverter()))

    documents = loader.load(
        make_config(tmp_path, "content/en/docs/reference/admission controllers.md")
    )

    assert len(documents) == 1
    assert "%20" in str(documents[0].provenance.source_url)
    assert "--enable-admission-plugins" in documents[0].content


def test_loader_rejects_paths_outside_root(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (corpus_root / "linked.md").symlink_to(outside)
    loader = MarkdownCorpusLoader(MarkdownNormalizer(PassThroughConverter()))

    with pytest.raises(ValueError, match="escapes"):
        loader.discover(make_config(corpus_root, "linked.md"))

    with pytest.raises(ValueError, match="inside"):
        loader.discover(make_config(corpus_root, "../outside.md"))


def test_loader_rejects_oversized_file(tmp_path: Path) -> None:
    source = tmp_path / "large.md"
    source.write_text("# Large\nToo much content", encoding="utf-8")
    loader = MarkdownCorpusLoader(MarkdownNormalizer(PassThroughConverter()))

    with pytest.raises(ValueError, match="size limit"):
        loader.load(make_config(tmp_path, "large.md", max_size=5))


@pytest.mark.parametrize(
    ("include_paths", "max_size", "message"),
    [((), 100, "include path"), ((PurePosixPath("docs"),), 0, "file size")],
)
def test_corpus_config_rejects_invalid_limits(
    tmp_path: Path,
    include_paths: tuple[PurePosixPath, ...],
    max_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CorpusConfig(
            root=tmp_path,
            corpus=Corpus.KUBERNETES,
            repository="kubernetes/website",
            revision=REVISION,
            license="CC-BY-4.0",
            include_paths=include_paths,
            max_file_size_bytes=max_size,
        )
