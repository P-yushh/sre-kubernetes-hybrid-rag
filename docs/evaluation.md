# Golden evaluation

The quality suite separates deterministic retrieval checks from LLM-judged answer faithfulness.
Context precision and recall compare stable retrieved chunk IDs against human-reviewed reference
IDs. Faithfulness uses the Ragas collections API to determine whether answer claims are supported by
the retrieved text.

The benchmark file is `data/golden_dataset.json` and has this shape:

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

Default aggregate thresholds are context precision `0.80`, context recall `0.80`, faithfulness
`0.85`, and answer rate `1.00`. A refusal receives zero faithfulness and lowers answer rate rather
than being omitted from the report.

## Quality-gate command

Runtime composition stays outside the evaluator so local, staging, and CI environments can build
the same interfaces with different infrastructure. Supply a zero-argument factory that returns an
`EvaluationRunner`:

```bash
uv run sre-rag-quality-gate \
  --runner-factory your_package.evaluation:create_runner \
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

For trusted same-repository pull requests, configure these GitHub repository settings:

- Variable `RAG_QUALITY_GATE_ENABLED=true`.
- Variable `RAG_EVALUATION_RUNNER_FACTORY` containing the reviewed `module:function` path.
- Secret `OPENAI_API_KEY` for generation and Ragas judging.

The CI caller first requires the deterministic Python checks to pass, then invokes the reusable RAG
workflow. Pull requests from forks never receive the secret-bearing job. Add `RAG quality /
Golden-set thresholds` to the protected branch's required checks only after activation; before
that point, the workflow is infrastructure rather than evidence that the corpus meets its targets.
