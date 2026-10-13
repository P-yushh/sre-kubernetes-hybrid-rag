# Golden evaluation

The quality suite separates deterministic retrieval checks from LLM-judged answer faithfulness.
Context precision uses average precision over the ranked stable chunk IDs, so relevant evidence is
rewarded for appearing early without automatically penalizing every additional candidate. Context
recall measures how much of the reviewed reference set was retrieved. Faithfulness uses the Ragas
collections API to determine whether answer claims are supported by the retrieved text.

The release benchmark file is `data/golden_dataset.json` and has this shape:

```json
{
  "schema_version": 1,
  "samples": [
    {
      "sample_id": "unique-case-name",
      "question": "A Kubernetes or SRE question",
      "expected_answer": "A human-reviewed reference answer",
      "ground_truth_context": ["stable-ingested-chunk-id"]
    }
  ]
}
```

Do not invent chunk IDs. Build the file only after corpus repositories and revisions are pinned,
then select IDs from actual ingestion output. The loader rejects empty datasets, duplicate sample
IDs, duplicate questions, and duplicate ground-truth IDs so incomplete benchmark setup cannot pass
silently. `expected_answer` is retained for a later answer-correctness metric; faithfulness compares
the generated answer directly with retrieved evidence and does not use the reference answer.

`data/golden_dataset.candidate.json` contains 30 source-grounded review candidates: 20 Kubernetes
cases and 10 incident cases. It is intentionally not named `golden_dataset.json`, so CI cannot treat
machine-assisted curation as human approval. For each sample, a reviewer must open the referenced
chunk in the built artifact and verify that the question is unambiguous, the expected answer contains
only supported claims, and the listed chunk is sufficient evidence. After all cases are approved,
copy the reviewed content to `data/golden_dataset.json` and validate its references:

```bash
uv run sre-rag-benchmark-verify --dataset data/golden_dataset.json
```

Default aggregate thresholds are context precision `0.80`, context recall `0.80`, faithfulness
`0.85`, and answer rate `1.00`. A refusal receives zero faithfulness and lowers answer rate rather
than being omitted from the report.

## Quality-gate command

Runtime composition stays outside the evaluator so local, staging, and CI environments can build
the same interfaces with different infrastructure. Supply a zero-argument factory that returns an
`EvaluationRunner`:

```bash
uv run sre-rag-quality-gate \
  --runner-factory sre_rag.evaluation.factory:create_runner \
  --dataset data/golden_dataset.json \
  --report artifacts/evaluation-report.json
```

The command writes the complete per-sample JSON report, appends a score table to
`GITHUB_STEP_SUMMARY` when GitHub provides that path, and exits with status `1` when any threshold
fails. Dataset/configuration errors remain hard failures rather than being converted into a passing
or skipped score.

The secret-bearing workflow is deliberately separate from ordinary pull-request checks. Enable it
as a required branch-protection check only after the real golden dataset and production runner
factory have been added. Public fork code must never run in a context that exposes evaluation API
keys.

The live workflow targets a dedicated self-hosted runner labeled `macOS`, `ARM64`, and `sre-rag`.
This keeps PyTorch on the M3 host, uses its model cache, and starts only Qdrant in Docker. Protect the
`rag-quality` GitHub Environment with a required reviewer; do not attach the label to a general-use
runner. The job rebuilds the pinned index before evaluation so changes to ingestion, embeddings, or
retrieval cannot be tested against stale vectors.

For trusted same-repository pull requests, configure these GitHub repository settings:

- Variable `RAG_QUALITY_GATE_ENABLED=true`.
- Variable `RAG_EVALUATION_RUNNER_FACTORY` containing the reviewed `module:function` path.
- Secret `OPENAI_API_KEY` for generation and Ragas judging.
- A protected `rag-quality` Environment and a dedicated self-hosted macOS ARM64 runner with Docker,
  Python 3.11, `uv`, and sufficient model-cache disk space.

The CI caller first requires the deterministic Python checks to pass, then invokes the reusable RAG
workflow. Pull requests from forks never receive the secret-bearing job. Add `RAG quality /
Golden-set thresholds` to the protected branch's required checks only after activation; before
that point, the workflow is infrastructure rather than evidence that the corpus meets its targets.
