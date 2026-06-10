Issue-Map Harness Eval Suite
============================

Purpose
-------

This suite checks whether changing models, tools, MCPs, or skills actually improves the issue-map investigation runner. It is separate from Django application code and does not run live provider calls unless explicitly requested.

It covers two questions:

1. Did a real runner command produce the right behavior for a normal issue job?
   - required bounded tools were used,
   - first-use tool order and required repo file reads were satisfied,
   - forbidden tools were not used,
   - final node IDs and file paths stayed on the allowlists,
   - every expected node appeared in the investigation path with its exact repository-relative file path,
   - confidence values stayed in range.

2. Did OpenCode Zen actually receive a request?
   - optional live smoke calls `https://opencode.ai/zen/v1/chat/completions`,
   - raw model ID defaults to `kimi-k2.5`,
   - OpenCode config model ID is tracked separately as `opencode/kimi-k2.5`,
   - response metadata and usage presence are written to `temp/harness_eval/` when requested.

Tracked Files
-------------

- `samples/*.json`: synthetic issue-map jobs. Hidden `expect` fields are used only by the deterministic evaluator.
- `golden/repo_issue_consensus.json`: repo-fixture reference answers agreed by independent GPT-5.5 xhigh and Kimi judges.
- `golden/judge_logs/*.txt`: prompts and raw reviewer outputs used to audit the repo-fixture golden set.
- `sample_transcripts/*.json`: example runner transcripts for evaluator development.
- `harness_matrix.sample.json`: model/tool/MCP/skill comparison matrix.
- `evaluator.py`: schema and transcript scoring logic.
- `runner.py`: CLI for validation, transcript evaluation, dry-run matrix expansion, and optional live OpenCode calls.
- `pi_runner.py` and `pi/issue_map_extension.ts`: real Pi/OpenCode SUT adapter with bounded issue-map tools.

Offline Commands
----------------

Run these in normal development and CI:

```
python -m unittest harness_eval.tests
python -m harness_eval.runner validate-samples
python -m harness_eval.runner validate-matrix
python -m harness_eval.runner validate-golden
python -m harness_eval.runner render-job harness_eval/samples/origin_trace.json
python -m harness_eval.runner run-harness harness_eval/samples/origin_trace.json -- python harness_eval/fixtures/fake_good_harness.py
python -m harness_eval.runner evaluate-transcript harness_eval/samples/origin_trace.json harness_eval/sample_transcripts/good_origin_trace.json
python -m harness_eval.runner dry-run-matrix
```

`run-harness` passes only the stripped job packet on stdin. It does not send the hidden expected nodes, required tools, or allowed paths to the command under test.

`evaluate-transcripts` intentionally returns non-zero while the `bad_forbidden_tool.json` sample remains in the directory. Use it as a negative example, or evaluate known-good transcripts individually.

Live OpenCode Smoke
-------------------

Live calls are gated so they cannot happen accidentally:

```
export OPENCODE_API_KEY=...
export RUN_OPENCODE_LIVE_TESTS=true
python -m harness_eval.runner live-smoke --live --model kimi-k2.5 --write-result
```

Expected result:

- exit code 0,
- `passed: true`,
- `raw_model_id: kimi-k2.5`,
- `opencode_model_id: opencode/kimi-k2.5`,
- `response_id` present,
- `usage_present: true`,
- `usage_receipt_present: true`,
- `dashboard_correlation` contains the timestamp, model, endpoint, and response ID to compare with the dashboard.

`live-smoke` now fails if OpenCode returns HTTP 200 without a usage object and response ID. If the API usage dashboard still shows nothing after a successful live smoke, preserve the generated `temp/harness_eval/live_smoke_*.json` file and compare its response ID/time with the dashboard. The CLI can prove that Zen returned usage metadata for the request; it cannot prove that the web dashboard rendered it.

Live Sample Eval
----------------

This sends a normal synthetic issue job directly to OpenCode Zen and evaluates the returned JSON as a runner transcript:

```
export OPENCODE_API_KEY=...
export RUN_OPENCODE_LIVE_TESTS=true
python -m harness_eval.runner live-sample harness_eval/samples/origin_trace.json --live --model kimi-k2.5 --write-result
```

This is not a replacement for a real Pi run with tools. It tests the model/provider's ability to complete the issue job shape and makes an actual OpenCode Zen request for dashboard verification.

`live-sample` fails loudly when the model output is not valid transcript JSON. The report includes:

- `live_call_passed`: HTTP response, choices, response ID, and usage receipt were present,
- `content_json_valid`: provider output parsed as a JSON object,
- `json_parse_error`: parser failure when the model ignored JSON-only instructions,
- `eval_passed`: deterministic transcript checks passed.

Live Pi SUT Eval
----------------

This is the production-like live check: `run-harness` stays the deterministic judge, while `pi_runner.py` is the system under test. The Pi runner installs/uses `@earendil-works/pi-coding-agent@0.79.1` through `npx`, binds only bounded repo-analysis tools from `pi/issue_map_extension.ts`, and emits the final transcript from a terminating `finish_issue_map_transcript` tool result.

For repo samples, the job packet contains a local repo fixture path and issue text, not a precomputed graph artifact. Pi must call `list_repo_files`, `search_repo_symbols`, and `read_repo_file`; the deterministic evaluator compares the final transcript against hidden `expect` values derived from `golden/repo_issue_consensus.json`. The SUT never receives that golden file or the hidden `expect` block.

Repo-fixture samples are not smoke tests. They cover multi-file call chains, same-file decoys, permission classes, GitHub ingestion failures, graph filtering, graph label conversion, parser visitor gaps, worker/cache key mismatch, and a negative needs-info issue where the correct answer is no node. These catch cases where the model includes every plausible node from a file, guesses from labels, or skips file reads before naming origin symbols.

```
export OPENCODE_API_KEY=...
export RUN_OPENCODE_LIVE_TESTS=true
python -m harness_eval.runner run-harness harness_eval/samples/repo_parser_timeout.json -- python -m harness_eval.pi_runner --live --model kimi-k2.5
python -m harness_eval.runner run-harness harness_eval/samples/repo_same_file_precision.json -- python -m harness_eval.pi_runner --live --model kimi-k2.5
python -m harness_eval.runner run-harness harness_eval/samples/repo_fetch_none_crash.json -- python -m harness_eval.pi_runner --live --model kimi-k2.5
```

Expected result:

- exit code 0,
- required repo-analysis tools were called,
- required tool order and required file reads were satisfied,
- forbidden filesystem/shell/network/GitHub tools were not called,
- expected node IDs and paths passed the deterministic evaluator,
- `pi_metadata.response_ids` and `pi_metadata.usage` are present in the transcript for API/dashboard correlation.

If this fails while `live-smoke` passes, the provider is reachable but the Pi tool loop/output contract is broken.

Evaluating Real Pi Runs
-----------------------

Run the real Pi command as a black box:

```
python -m harness_eval.runner run-harness harness_eval/samples/repo_parser_timeout.json -- <your-pi-command-and-args>
```

Or export a real run as a transcript shaped like:

```
{
  "sample_id": "origin_trace",
  "variant_id": "opencode-kimi-k25-issue-tools",
  "tool_calls": [{"name": "rank_issue_candidates", "arguments": {}}],
  "final": {
    "hypotheses": [{"node_id": "api/services.py::_build_and_store_analysis", "confidence": 0.8}],
    "investigation_path": [{"node_id": "api/services.py::_build_and_store_analysis", "path": "api/services.py"}],
    "confidence": {"score": 0.8}
  }
}
```

Then run:

```
python -m harness_eval.runner evaluate-transcript harness_eval/samples/origin_trace.json path/to/transcript.json
```

Use `harness_matrix.sample.json` as the comparison grid for models, tools, MCPs, and skills.
