"""Versioned remote-corpus acquisition contracts."""

from sre_rag.corpora.fetch import CorpusFetcher, FetchedCorpus, HttpxArchiveDownloader
from sre_rag.corpora.manifest import CorpusManifest, CorpusSource, load_manifest

__all__ = [
    "CorpusFetcher",
    "CorpusManifest",
    "CorpusSource",
    "FetchedCorpus",
    "HttpxArchiveDownloader",
    "load_manifest",
]
