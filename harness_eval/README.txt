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
- `swebench/runner.py`: SWE-bench localization benchmark wrapper for the active runtime Pi issue harness.

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

Runtime Pi Benchmark Eval
-------------------------

The old `harness_eval.pi_runner` adapter has been retired. Production-like Pi
benchmarks now use one active contract:

```
llm.issue_harness -> llm.pi_issue_runner -> llm/pi_issue_extension.ts
```

For SWE-bench localization samples, `harness_eval.swebench.runner` builds the
bounded runtime job, calls `llm.issue_harness.run_issue_harness()`, and defaults
to `python -m llm.pi_issue_runner --provider opencode --model <model>`. Hidden
SWE-bench patch labels stay inside the evaluator and are never sent to Pi.

```
export OPENCODE_API_KEY=...
export RUN_OPENCODE_LIVE_TESTS=true
python -m harness_eval.swebench.runner run-sample harness_eval/swebench_samples/<sample>.json --live --model kimi-k2.5 --write-transcript
python -m harness_eval.swebench.runner run-matrix --live --samples-dir harness_eval/swebench_samples --matrix harness_eval/harness_matrix.sample.json --limit 20 --write-report
```

Expected result:

- exit code 0,
- runtime tools such as `get_issue_context`, `list_repo_files`, `search_repo_symbols` or `search_repo_text`, and an inspection tool were called,
- forbidden filesystem/shell/network/GitHub tools were not called,
- returned source paths overlap the hidden SWE-bench gold source files,
- `pi_metadata.response_ids` and `pi_metadata.usage` are present when the provider reports them.

If this fails while `live-smoke` passes, the provider is reachable but the
runtime Pi tool loop/output contract is broken.

Evaluating External Runner Transcripts
--------------------------------------

`harness_eval.runner run-harness` still accepts any black-box command that
speaks the synthetic issue-map sample transcript shape:

```
python -m harness_eval.runner run-harness harness_eval/samples/repo_parser_timeout.json -- <your-pi-command-and-args>
```

Use this for external experiments against the legacy synthetic samples. Runtime
Pi/SWE-bench benchmarks should use `harness_eval.swebench.runner`, which targets
`llm.pi_issue_runner` by default.

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
