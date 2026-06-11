# Pi Harness SWE-bench Localization Testing Suite Plan

## Purpose

Build an autonomous test suite for comparing LLMs used by the Pi issue harness.

The product task is not "fix the repository." The product task is:

> Given a GitHub issue and an analyzed repository graph, identify the code that most likely corresponds to the issue.

SWE-bench is useful because every instance already contains:

- an issue-like problem statement,
- a repository name,
- the base commit before the fix,
- a gold solution patch.

The suite will convert the gold solution patch into hidden localization labels. The model will see only the issue text and bounded repo/code tools. The model will never see the gold patch, test patch, hidden labels, or any marker that identifies which repository files are expected.

## Non-Goals

- Do not ask models to generate patches.
- Do not run SWE-bench Docker patch evaluation.
- Do not use weighted quality scores.
- Do not use LLM judges as ground truth.
- Do not count a partial checklist score as success.
- Do not expose `patch`, `test_patch`, `expect`, gold files, or expected node IDs to the system under test.

## Primary Success Definition

Each model run is a binary result for each SWE-bench instance.

```text
PASS = the final harness output names at least one source-code node whose repository-relative path is in the hidden gold source-file set.

FAIL = timeout, invalid JSON, harness contract failure, no final nodes, or no returned node path overlaps the hidden gold source-file set.
```

The suite report will show only counts and rates:

```text
model_id     passed   total   pass_rate   median_latency_ms   p95_latency_ms   total_cost_usd   avg_cost_usd
kimi-k2.5    73       100     73.0%       8200                19100            1.20             0.012
model-b      68       100     68.0%       5100                14400            0.90             0.009
```

Latency and cost are descriptive operational columns. They do not change pass/fail.

## Secondary Binary Checks

These checks are optional columns, not weighted components.

```text
strict_symbol_pass = at least one returned node overlaps a gold patch hunk's owning function, method, class, or module.
all_gold_files_found = every hidden gold source file appears in the returned final paths.
contract_pass = the runner produced valid output and satisfied required tool-use rules.
```

The first implementation must support `PASS` and `contract_pass`. `strict_symbol_pass` can be added after file-level localization is stable.

## Why File-Level First

SWE-bench patches are solution patches, not localization annotations. A fix may touch helper functions, imports, compatibility shims, and nearby code that is not always the conceptual origin. File-level labels are the least ambiguous autonomous signal.

Symbol-level labels are useful later, but only after the importer can map patch hunks to graph nodes reliably.

## Dataset Choice

Start with `SWE-bench/SWE-bench_Verified`, split `test`.

Reasons:

- 500 instances is large enough for model comparison.
- It is smaller than full SWE-bench and cheaper to run.
- It is higher quality than a raw mined issue/PR set.
- It contains the required fields: `instance_id`, `repo`, `base_commit`, `problem_statement`, `patch`, `test_patch`, `issue_url`, and `pr_url`.

Use `SWE-bench/SWE-bench_Lite` only for fast development if Verified is too slow. Do not mix datasets in one headline report.

## Required New Files

```text
harness_eval/swebench/
  __init__.py
  importer.py
  diff_labels.py
  sample_builder.py
  evaluator.py
  runner.py
  README.md

harness_eval/swebench_samples/
  .gitkeep

temp/harness_eval/swebench/
  datasets/
  artifacts/
  transcripts/
  reports/
  artifact_manifest.json

playground/harness_eval/swebench/
  repos/
```

`temp/` is used for downloaded datasets, generated artifacts, transcripts, and reports. `playground/` is used for cloned external repositories, matching the repository scratch-file policy. Generated large files must not be committed.

Small committed fixtures may be added later under `harness_eval/fixtures/` if unit tests need stable examples.

## Dependencies

Add these dependencies only if implementation begins:

```text
datasets
unidiff
```

`datasets` loads SWE-bench from Hugging Face. `unidiff` parses gold patches without fragile string splitting.

If dependency growth becomes a concern, the importer may also accept a local JSONL export of SWE-bench rows. The evaluation design remains the same.

## Data Model

The importer will convert each accepted SWE-bench row into a local sample file.

Example generated sample shape:

```json
{
  "schema_version": 1,
  "id": "swebench__django__django-12345",
  "source": {
    "kind": "swebench",
    "dataset": "SWE-bench/SWE-bench_Verified",
    "split": "test",
    "instance_id": "django__django-12345",
    "issue_url": "https://github.com/django/django/issues/...",
    "pr_url": "https://github.com/django/django/pull/..."
  },
  "task": "Analyze the provided repository and identify the code nodes most relevant to the selected issue.",
  "repo": {
    "kind": "github",
    "full_name": "django/django",
    "revision": "<base_commit>",
    "analysis_profile": "python-v1"
  },
  "issue": {
    "number": 12345,
    "title": "SWE-bench issue django__django-12345",
    "body": "<problem_statement>",
    "labels": []
  },
  "expect": {
    "label_source": "swebench_patch",
    "label_confidence": "gold_file_silver_origin",
    "gold_source_files": ["django/core/handlers/base.py"],
    "gold_hunks": [
      {
        "path": "django/core/handlers/base.py",
        "old_start": 100,
        "old_length": 8,
        "new_start": 100,
        "new_length": 9
      }
    ],
    "forbidden_tools": ["shell", "filesystem", "network", "github_api"]
  }
}
```

The `expect` block is hidden from the system under test by `build_job_packet`, exactly like the existing synthetic harness samples.

## Patch-To-Label Rules

The label builder reads only the SWE-bench `patch` field.

It must not read `test_patch` for labels or model input.

Patch parsing rules:

1. Parse the unified diff using `unidiff.PatchSet`.
2. For each patched file, normalize to repository-relative POSIX path.
3. Prefer the `b/` target path when present.
4. For deleted files, use the `a/` source path because the file exists at `base_commit` and can still be the corresponding code.
5. Drop paths that are not source files for the active profile.
6. Drop test files.
7. Drop docs, examples, changelog, generated files, and metadata-only files.
8. Keep every remaining path as a `gold_source_file`.
9. Keep every hunk range for later strict symbol matching.
10. Skip the instance if no source file remains after filtering.

Initial source-file allowlist:

```text
.py
```

Initial test/docs/generated filters:

```text
tests/
test/
testing/
*/tests/*
*/test/*
test_*.py
*_test.py
docs/
doc/
examples/
example/
changelog*
release*
*.md
*.rst
*.txt
*.json
*.yml
*.yaml
*.toml
*.ini
```

This can expand later for JS/TS after the current multi-language branch lands cleanly.

## Repository Preparation

For each accepted sample:

1. Clone `repo` into `playground/harness_eval/swebench/repos/<owner>/<repo>`.
2. Checkout `base_commit`.
3. Build a graph artifact using the same parser/artifact path used by the application.
4. Store generated artifacts under `temp/harness_eval/swebench/artifacts/<sample_id>.json`.
5. Write `temp/harness_eval/swebench/artifact_manifest.json` mapping each sample ID to its artifact path, checkout path, repo, and revision.
6. Do not modify the repository with the gold patch.
7. Do not install project dependencies or run project tests.

The suite evaluates localization, not patch correctness.

Artifact preparation may add a local-checkout helper rather than calling GitHub issue APIs. Any helper must still reuse the parser and graph artifact builders so runtime job shape matches the application. Local checkouts must stay under `playground/harness_eval/swebench/repos/`.

## Job Packet Sent To The Harness

The job packet must contain:

- `job_id`,
- task objective,
- repo metadata,
- issue title/body from `problem_statement`,
- graph nodes and edges,
- bounded file contents,
- available bounded tools,
- output format.

The job packet must not contain:

- `patch`,
- `test_patch`,
- `expect`,
- `gold_source_files`,
- `gold_hunks`,
- `FAIL_TO_PASS`,
- `PASS_TO_PASS`.

`FAIL_TO_PASS` and `PASS_TO_PASS` are SWE-bench patch-evaluation signals. They are not needed for localization evaluation.

## Harness Execution

The suite will run each model through the runtime Pi issue harness contract used by the app.

Target runtime contract:

```text
llm.issue_harness.build_issue_harness_job
llm.issue_harness.run_issue_harness
llm.pi_issue_runner
llm/pi_issue_extension.ts
```

Required runtime tools:

```text
get_issue_context
list_repo_files
search_repo_symbols or search_repo_text
read_repo_file, read_node_context, or get_neighbors before final nodes
finish_issue_map_transcript
```

The older `harness_eval/pi_runner.py` repo-fixture adapter has been retired. This prevents a model from passing SWE-bench localization with a different tool surface than production.

Preferred command shape:

```bash
python -m harness_eval.swebench.runner run-matrix \
  --samples-dir harness_eval/swebench_samples \
  --matrix harness_eval/harness_matrix.sample.json \
  --limit 100 \
  --write-report
```

Single-sample command:

```bash
python -m harness_eval.swebench.runner run-sample \
  harness_eval/swebench_samples/swebench__django__django-12345.json \
  --model kimi-k2.5 \
  --write-transcript
```

The runner must record:

- model ID,
- sample ID,
- pass/fail,
- failure reason,
- latency milliseconds,
- token usage if provider metadata exposes it,
- estimated cost if pricing is configured,
- transcript path,
- report path.

## Evaluator Logic

The SWE-bench localization evaluator is separate from the current synthetic evaluator because its labels are file-based at first.

Input:

- generated sample with hidden `expect.gold_source_files`,
- transcript emitted by the Pi harness.

Algorithm:

1. Validate transcript shape.
2. Validate required Pi contract:
   - tool calls exist,
   - issue context was read,
   - repo files were listed,
   - symbol or text search was used,
   - code, node context, or neighbors were inspected before final nodes.
3. Extract final paths from `final.hypotheses` and `final.investigation_path`.
4. If a final item has only `node_id`, resolve its path from the generated graph artifact.
5. Normalize all paths to POSIX repository-relative paths.
6. Compute `matched_gold_files = returned_paths ∩ gold_source_files`.
7. Return `passed = contract_pass and matched_gold_files is not empty`.

Failure reasons must be explicit and mutually understandable:

```text
invalid_json
missing_final
contract_failed
no_final_nodes
no_resolved_paths
no_gold_file_overlap
timeout
runner_error
```

## Report Format

Write JSON and Markdown reports under `temp/harness_eval/swebench/reports/`.

JSON report shape:

```json
{
  "schema_version": 1,
  "dataset": "SWE-bench/SWE-bench_Verified",
  "split": "test",
  "sample_count": 100,
  "generated_at_utc": "2026-06-10T00:00:00Z",
  "variants": [
    {
      "variant_id": "opencode-kimi-k25-issue-tools",
      "model_id": "kimi-k2.5",
      "passed": 73,
      "failed": 27,
      "total": 100,
      "pass_rate": 0.73,
      "median_latency_ms": 8200,
      "p95_latency_ms": 19100,
      "total_cost_usd": 1.2,
      "avg_cost_usd": 0.012
    }
  ],
  "results": [
    {
      "sample_id": "swebench__django__django-12345",
      "variant_id": "opencode-kimi-k25-issue-tools",
      "passed": true,
      "failure_reason": null,
      "gold_source_files": ["django/core/handlers/base.py"],
      "returned_paths": ["django/core/handlers/base.py"],
      "matched_gold_files": ["django/core/handlers/base.py"],
      "latency_ms": 8200,
      "cost_usd": 0.012,
      "transcript_path": "temp/harness_eval/swebench/transcripts/..."
    }
  ]
}
```

Markdown report shape:

```text
Model Comparison

| variant | model | passed | total | pass rate | median latency | p95 latency | total cost |
|---|---:|---:|---:|---:|---:|---:|---:|

Failures By Reason

| reason | count |
|---|---:|

Worst Misses

| sample | gold files | returned paths | reason |
|---|---|---|---|
```

## CLI Plan

Add subcommands under `harness_eval/swebench/runner.py`.

```bash
python -m harness_eval.swebench.runner import-samples \
  --dataset SWE-bench/SWE-bench_Verified \
  --split test \
  --limit 100 \
  --output-dir harness_eval/swebench_samples
```

Creates sample JSON files with hidden `expect` labels.

```bash
python -m harness_eval.swebench.runner prepare-artifacts \
  --samples-dir harness_eval/swebench_samples \
  --limit 100
```

Clones repos, checks out `base_commit`, builds graph artifacts, and stores them in `temp/`.

This command also writes `temp/harness_eval/swebench/artifact_manifest.json`. Later run and evaluation commands must fail with `runner_error` if a required artifact manifest entry is missing.

```bash
python -m harness_eval.swebench.runner validate-samples \
  --samples-dir harness_eval/swebench_samples
```

Checks sample schema and confirms no model-visible job packet leaks hidden labels.

```bash
python -m harness_eval.swebench.runner run-sample \
  harness_eval/swebench_samples/<sample>.json \
  --variant-id opencode-kimi-k25-issue-tools \
  --model kimi-k2.5 \
  --write-transcript
```

Runs one model on one sample.

```bash
python -m harness_eval.swebench.runner run-matrix \
  --samples-dir harness_eval/swebench_samples \
  --matrix harness_eval/harness_matrix.sample.json \
  --limit 100 \
  --write-report
```

Runs all selected variants and writes aggregate reports.

```bash
python -m harness_eval.swebench.runner evaluate-transcript \
  harness_eval/swebench_samples/<sample>.json \
  temp/harness_eval/swebench/transcripts/<transcript>.json
```

Evaluates a saved transcript without calling a model.

## Integration With Existing Harness

Reuse these existing concepts:

- hidden `expect` fields,
- stripped job packets,
- Pi JSON transcript extraction,
- bounded tool surface,
- explicit live-call gates,
- reports under `temp/harness_eval`.

Do not force SWE-bench samples into the current `harness_eval/evaluator.py` checklist. That evaluator is intentionally strict for synthetic samples with exact node IDs. SWE-bench file localization needs its own binary file-overlap evaluator.

Fixed deterministic preprocessing is part of the product harness. The same parser, graph artifact builder, issue evidence extractor, deterministic ranker, bounded file selection, and seed candidate generation must be used for every model variant in a comparison. A report is invalid if model variants were run with different non-model preprocessing.

## Model Matrix

Keep the matrix simple.

Each variant must specify:

```json
{
  "id": "opencode-kimi-k25-issue-tools",
  "model": {
    "opencode_model_id": "opencode/kimi-k2.5",
    "raw_model_id": "kimi-k2.5"
  },
  "endpoint": {
    "provider": "opencode_zen",
    "chat_completions_url": "https://opencode.ai/zen/v1/chat/completions"
  }
}
```

The first comparison should use the same tools and prompt for every model. Only the model ID changes. Tool or prompt experiments should be separate variants with names that make the changed factor obvious.

## Cost And Latency

The runner will measure wall-clock latency around each harness invocation.

Cost is recorded only when enough usage metadata and pricing configuration exist.

Pricing config shape:

```json
{
  "kimi-k2.5": {
    "input_per_1m_tokens_usd": 0.0,
    "output_per_1m_tokens_usd": 0.0
  }
}
```

If pricing is missing, set `cost_usd` to `null`. Do not block evaluation.

## Leakage Controls

Every generated sample must pass a leakage test.

The rendered job packet must not contain:

- `patch`,
- `test_patch`,
- `gold_source_files`,
- `gold_hunks`,
- `expect`,
- known hunk line numbers,
- `FAIL_TO_PASS`,
- `PASS_TO_PASS`.

Important nuance: a gold file path may naturally appear in the graph or file listing because it is part of the repository. That is allowed. The forbidden leak is telling the model that the path is gold.

The leakage test should therefore assert absence of hidden field names and patch/test content, not absence of every path string that also exists in the repository graph.

## Instance Filtering

Skip an instance when:

- `patch` is empty,
- `repo` or `base_commit` is missing,
- patch parsing fails,
- no source file remains after filtering,
- checkout fails,
- graph artifact generation fails,
- the repo exceeds existing analysis limits,
- the generated job would exceed Pi harness limits so heavily that no gold file content or graph node can be made available.

Each skipped instance must be recorded with a reason in:

```text
temp/harness_eval/swebench/reports/import_skips.json
```

Skipped instances are not counted in model pass rates.

## Strict Symbol Matching Later

After file-level evaluation works, add strict symbol matching.

Algorithm:

1. For each gold hunk, use the graph artifact from `base_commit`.
2. Find nodes in the same file whose `[start_line, end_line]` overlaps the hunk's old-file line range.
3. Prefer the smallest overlapping function or method.
4. Fall back to class.
5. Fall back to module/file node.
6. Store these as `expect.gold_node_ids`.
7. `strict_symbol_pass = returned_node_ids ∩ gold_node_ids is not empty`.

Do not make strict symbol pass the primary metric until hunk-to-node mapping has unit tests across multi-hunk patches, class methods, decorators, module-level changes, and imports.

## Unit Tests

Add tests for:

- diff parser extracts modified source files from a patch,
- diff parser ignores `test_patch`,
- source filter excludes tests and docs,
- importer skips rows without usable labels,
- job packet hides hidden labels,
- evaluator passes when returned path overlaps a gold file,
- evaluator fails when returned path is only a non-gold file,
- evaluator resolves paths from returned node IDs,
- evaluator fails invalid JSON and missing final fields,
- report aggregation computes counts and pass rates without weighted scores.

Minimum test command:

```bash
python -m unittest harness_eval.tests
python -m unittest harness_eval.swebench.tests
```

If `harness_eval.swebench.tests` is not a package, use:

```bash
python -m unittest discover harness_eval
```

## Implementation Order

1. Add `diff_labels.py` and tests using tiny embedded unified diffs.
2. Add `importer.py` that loads SWE-bench rows and writes local sample JSON.
3. Add sample validation and leakage tests.
4. Add artifact preparation that reuses existing repo clone, parser, and graph artifact code.
5. Add file-level evaluator.
6. Add transcript-only evaluation command.
7. Add single-sample live runner.
8. Add matrix runner and aggregate reports.
9. Run 5 samples against one model.
10. Run 50 samples against two models.
11. Only then run 100+ samples.

## First Useful Milestone

The first milestone is complete when this works:

```bash
python -m harness_eval.swebench.runner import-samples --limit 10
python -m harness_eval.swebench.runner prepare-artifacts --limit 10
python -m harness_eval.swebench.runner validate-samples
python -m harness_eval.swebench.runner evaluate-transcript <sample> <known-good-transcript>
```

No provider API call is required for the first milestone.

## First Live Milestone

The first live milestone is complete when this works:

```bash
export OPENCODE_API_KEY=...
export RUN_OPENCODE_LIVE_TESTS=true
python -m harness_eval.swebench.runner run-sample <sample> --model kimi-k2.5 --write-transcript
```

Expected output:

- one transcript,
- one evaluation result,
- binary pass/fail,
- explicit failure reason if failed,
- latency recorded,
- usage metadata recorded if available.

## Operational Defaults

Use these defaults unless there is a clear reason to change them:

```text
dataset = SWE-bench/SWE-bench_Verified
split = test
initial_limit = 100
analysis_profile = python-v1
timeout_seconds = 180
max_parallel_model_runs = 1
max_parallel_repo_prepares = 2
```

Keep model runs serial at first to avoid confusing provider rate limits with model quality.

## Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Gold patch file is not the conceptual origin | Start with file-level pass/fail and label the confidence as `gold_file_silver_origin`. |
| Model finds a valid related file not touched by patch | Count as fail for this autonomous benchmark; inspect manually only when deciding whether to improve the dataset later. |
| Patch touches only tests/docs/config | Skip the instance. |
| Graph parser misses a gold file or node | Skip the instance for headline reports and record the reason. |
| Large repos exceed analysis limits | Skip initially; evaluate parser limit changes separately. |
| SWE-bench issue text mentions exact files | Accept it; real issues sometimes mention files. The patch remains hidden. |
| Provider returns malformed JSON | Fail the sample with `invalid_json`. |
| Provider is cheap but often violates tool contract | It will have low `contract_pass`; do not hide this behind quality scoring. |

## Final Decision Rules

Use this suite to answer simple questions:

```text
Which model has the highest pass_rate on the same sample set?
Which model is cheaper per attempted sample?
Which model is cheaper per passed sample?
Which model has lower median and p95 latency?
Which model violates the harness contract least often?
```

Do not collapse those into one score. Pick the model based on the tradeoff that matters for the deployment.

For this product, the default choice should be:

```text
highest pass_rate among models whose median latency and average cost are acceptable for the user experience
```
