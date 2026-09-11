"""Small factories that keep domain tests focused on behavior."""

from sre_rag.domain.documents import Chunk, Corpus, SourceProvenance

CONTENT_HASH = "a" * 64
REVISION = "b" * 40


def make_provenance() -> SourceProvenance:
    return SourceProvenance(
        corpus=Corpus.KUBERNETES,
        repository="kubernetes/website",
        revision=REVISION,
        relative_path="content/en/docs/tasks/debug/debug-application.md",
        source_url=(
            "https://github.com/kubernetes/website/blob/"
            f"{REVISION}/content/en/docs/tasks/debug/debug-application.md"
        ),
        license="CC-BY-4.0",
    )


def make_chunk() -> Chunk:
    return Chunk(
        chunk_id="kubernetes:debug-application#exit-codes@0",
        document_id="kubernetes:debug-application",
        text="A container terminated with exit code 137 and reason OOMKilled.",
        heading_path=("Debug Pods", "Exit codes"),
        ordinal=0,
        token_count=11,
        content_sha256=CONTENT_HASH,
        provenance=make_provenance(),
    )
