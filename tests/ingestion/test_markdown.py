"""Tests for Markdown preprocessing and Docling conversion."""

import pytest

from sre_rag.ingestion.markdown import (
    DoclingMarkdownConverter,
    MarkdownNormalizer,
    first_heading,
    split_front_matter,
    split_markdown_sections,
)
from tests.ingestion.helpers import PassThroughConverter


def test_normalizer_extracts_front_matter_and_preserves_exact_tokens() -> None:
    source = """---
title: Pod failure diagnosis
weight: 10
---
# Troubleshooting

{{< note >}}
Inspect `CrashLoopBackOff` and exit code `137`.
{{< /note >}}

{{< glossary_tooltip text="Pod" term_id="pod" >}} status matters.

```yaml
template: "{{< preserve-inside-code >}}"
maxSurge: 1
```
"""
    normalizer = MarkdownNormalizer(PassThroughConverter())

    result = normalizer.normalize(source, name="pods.md")

    assert result.title == "Pod failure diagnosis"
    assert "weight: 10" not in result.content
    assert "{{< note >}}" not in result.content
    assert "Pod status matters." in result.content
    assert "CrashLoopBackOff" in result.content
    assert 'template: "{{< preserve-inside-code >}}"' in result.content
    assert "maxSurge: 1" in result.content


def test_normalizer_falls_back_to_heading_then_filename() -> None:
    normalizer = MarkdownNormalizer(PassThroughConverter())

    assert normalizer.normalize("# Runtime failures\nBody", name="fallback.md").title == (
        "Runtime failures"
    )
    assert normalizer.normalize("No heading", name="pod-lifecycle.md").title == "Pod Lifecycle"


def test_sections_track_hierarchy_and_ignore_headings_inside_code() -> None:
    content = """Prelude.
# Pods
Pod body.
## Debugging
```bash
# this is a shell comment
kubectl describe pod example
```
More details.
# Services
Service body.
"""

    sections = split_markdown_sections(content)

    assert [section.heading_path for section in sections] == [
        (),
        ("Pods",),
        ("Pods", "Debugging"),
        ("Services",),
    ]
    assert "# this is a shell comment" in sections[2].body
    assert first_heading(content) == "Pods"


@pytest.mark.parametrize(
    "source",
    ["---\n- invalid\n---\nBody", "---\ntitle: Missing closing boundary\nBody"],
)
def test_invalid_front_matter_is_rejected(source: str) -> None:
    with pytest.raises(ValueError, match="front matter"):
        split_front_matter(source)


def test_docling_adapter_preserves_kubernetes_code_tokens() -> None:
    converter = DoclingMarkdownConverter()

    converted = converter.convert(
        "# Deployment\n\n```yaml\nmaxSurge: 1\nmaxUnavailable: 0\n```\n",
        name="deployment.md",
    )

    assert "maxSurge: 1" in converted
    assert "maxUnavailable: 0" in converted
