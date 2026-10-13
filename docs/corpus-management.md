# Corpus management

The raw knowledge sources remain online and are never committed to this repository. The versioned
[`corpora/manifest.yaml`](../corpora/manifest.yaml) pins the exact source revision, selected Markdown
paths, expected file count, compression-independent content checksum, size limits, license, and
attribution for each corpus.

## Pinned sources

| Corpus | Selected content | License |
| --- | --- | --- |
| `kubernetes/website` | English concepts and tasks | CC-BY-4.0 |
| `icco/postmortems` | Structured postmortem records under `data/` | GPL-3.0 |

The postmortem records are consumed for local indexing but are not redistributed from this public
repository. Generated raw snapshots, processed documents, and vector data remain ignored. Source
repository, revision, URL, and license metadata are retained in every normalized document and chunk.

## Verify the remote snapshots

```bash
uv run sre-rag-corpus-verify
```

For every source, the command:

1. Streams the immutable GitHub archive into a bounded temporary file.
2. Rejects absolute paths, parent traversal, multiple archive roots, links, devices, duplicates, and
   oversized content.
3. Extracts only selected regular Markdown files.
4. Confirms the expected file count and deterministic content SHA-256.
5. Deletes the downloaded archive and extracted documents when the verification context closes.

The checksum covers each selected file's relative path, byte length, and content. It therefore stays
stable even if GitHub changes tar compression while the pinned source tree remains identical.

## Updating a source

Source upgrades are deliberate data changes, not automatic application updates:

1. Review the upstream changes and license.
2. Replace the manifest revision with the reviewed full commit SHA.
3. Materialize only the selected paths and calculate their deterministic content digest.
4. Update the expected Markdown count and checksum.
5. Run ingestion, retrieval regression tests, and the complete golden benchmark.
6. Commit the manifest change together with its evaluation evidence.

Never point the manifest at a mutable branch name such as `main` or disable checksum validation to
work around an upstream change.
