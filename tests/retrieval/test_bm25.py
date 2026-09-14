"""Behavioral tests for exact-token BM25 retrieval."""

from hashlib import sha256

import pytest

from sre_rag.domain.documents import Chunk
from sre_rag.domain.retrieval import RetrievalMethod, RetrievalQuery
from sre_rag.retrieval.bm25 import BM25Config, BM25Retriever, SRETokenizer
from tests.domain.factories import make_chunk


def _chunk(chunk_id: str, text: str, ordinal: int) -> Chunk:
    return make_chunk().model_copy(
        update={
            "chunk_id": chunk_id,
            "text": text,
            "ordinal": ordinal,
            "token_count": len(text.split()),
            "content_sha256": sha256(text.encode()).hexdigest(),
        }
    )


def _corpus() -> tuple[Chunk, ...]:
    return (
        _chunk(
            "crash-loop",
            "A CrashLoopBackOff indicates that Kubernetes repeatedly restarts a container.",
            0,
        ),
        _chunk(
            "oom-killed",
            "A container with reason OOMKilled may terminate with Linux exit code 137.",
            1,
        ),
        _chunk(
            "admission",
            "Configure the API server with --enable-admission-plugins for admission control.",
            2,
        ),
        _chunk(
            "rollout",
            "Set spec.strategy.rollingUpdate.maxSurge when configuring a Deployment rollout.",
            3,
        ),
        _chunk(
            "probe",
            "A readiness probe controls whether a Pod receives Service traffic.",
            4,
        ),
    )


def test_tokenizer_preserves_exact_identifiers_and_adds_components() -> None:
    tokens = SRETokenizer().tokenize(
        "CrashLoopBackOff --enable-admission-plugins spec.strategy.rollingUpdate.maxSurge exit 137"
    )

    assert "crashloopbackoff" in tokens
    assert "--enable-admission-plugins" in tokens
    assert {"enable", "admission", "plugins"} <= set(tokens)
    assert "spec.strategy.rollingupdate.maxsurge" in tokens
    assert {"spec", "strategy", "rollingupdate", "maxsurge"} <= set(tokens)
    assert "137" in tokens


@pytest.mark.parametrize(
    ("query_text", "expected_chunk_id"),
    [
        ("Why is my pod in CrashLoopBackOff?", "crash-loop"),
        ("What does OOMKilled exit code 137 mean?", "oom-killed"),
        ("How do I use --enable-admission-plugins?", "admission"),
        ("What should maxSurge be during a rollout?", "rollout"),
    ],
)
def test_retrieve_ranks_exact_sre_tokens_first(
    query_text: str,
    expected_chunk_id: str,
) -> None:
    results = BM25Retriever(_corpus()).retrieve(RetrievalQuery(text=query_text))

    assert results[0].chunk.chunk_id == expected_chunk_id
    assert results[0].stages[0].method is RetrievalMethod.BM25
    assert results[0].stages[0].rank == 1
    assert results[0].stages[0].score > 0


def test_retrieve_returns_only_lexical_matches_and_honors_limit() -> None:
    results = BM25Retriever(_corpus()).retrieve(
        RetrievalQuery(text="container Kubernetes rollout probe"),
        limit=2,
    )

    assert len(results) == 2
    assert [candidate.stages[0].rank for candidate in results] == [1, 2]


def test_equal_scores_use_chunk_id_as_deterministic_tie_breaker() -> None:
    chunks = (
        _chunk("z-chunk", "rare-token alpha", 0),
        _chunk("a-chunk", "rare-token alpha", 1),
        _chunk("unrelated-one", "readiness probe", 2),
        _chunk("unrelated-two", "storage volume", 3),
        _chunk("unrelated-three", "network policy", 4),
    )

    results = BM25Retriever(chunks).retrieve(RetrievalQuery(text="rare-token"))

    assert [candidate.chunk.chunk_id for candidate in results] == ["a-chunk", "z-chunk"]


def test_empty_index_and_tokenless_query_return_no_candidates() -> None:
    assert BM25Retriever(()).retrieve(RetrievalQuery(text="OOMKilled")) == ()
    assert BM25Retriever(_corpus()).retrieve(RetrievalQuery(text="???")) == ()


def test_tokenless_chunks_are_not_added_to_sparse_index() -> None:
    chunks = (_chunk("symbols", "???", 0), _chunk("match", "OOMKilled", 1))

    results = BM25Retriever(chunks).retrieve(RetrievalQuery(text="OOMKilled"))

    assert [candidate.chunk.chunk_id for candidate in results] == ["match"]


def test_lexical_match_is_returned_even_when_small_corpus_idf_is_negative() -> None:
    chunk = _chunk("only-chunk", "OOMKilled", 0)

    results = BM25Retriever((chunk,)).retrieve(RetrievalQuery(text="OOMKilled"))

    assert results[0].chunk.chunk_id == "only-chunk"


def test_duplicate_chunk_ids_are_rejected() -> None:
    chunk = _corpus()[0]

    with pytest.raises(ValueError, match="unique chunk IDs"):
        BM25Retriever((chunk, chunk))


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        BM25Config(k1=-0.1)
    with pytest.raises(ValueError):
        BM25Config(b=-0.1)
    with pytest.raises(ValueError):
        BM25Config(b=1.1)
    with pytest.raises(ValueError):
        BM25Config(epsilon=-0.1)


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="default limit"):
        BM25Config(default_limit=0)

    retriever = BM25Retriever(_corpus())
    with pytest.raises(ValueError, match="limit"):
        retriever.retrieve(RetrievalQuery(text="OOMKilled"), limit=0)
