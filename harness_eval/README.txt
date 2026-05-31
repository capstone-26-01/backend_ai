Issue-Map Harness Eval Suite
============================

Purpose
-------

This suite checks whether changing models, tools, MCPs, or skills actually improves the issue-map investigation runner. It is separate from Django application code and does not run live provider calls unless explicitly requested.

It covers two questions:

1. Did a real runner command produce the right behavior for a normal issue job?
   - required bounded tools were used,
   - forbidden tools were not used,
   - final node IDs and file paths stayed on the allowlist,
   - confidence values stayed in range.

2. Did OpenCode Zen actually receive a request?
   - optional live smoke calls `https://opencode.ai/zen/v1/chat/completions`,
   - raw model ID defaults to `kimi-k2.5`,
   - OpenCode config model ID is tracked separately as `opencode/kimi-k2.5`,
   - response metadata and usage presence are written to `temp/harness_eval/` when requested.

Tracked Files
-------------

- `samples/*.json`: synthetic issue-map jobs. Hidden `expect` fields are used only by the deterministic evaluator.
- `sample_transcripts/*.json`: example runner transcripts for evaluator development.
- `harness_matrix.sample.json`: model/tool/MCP/skill comparison matrix.
- `evaluator.py`: schema and transcript scoring logic.
- `runner.py`: CLI for validation, transcript evaluation, dry-run matrix expansion, and optional live OpenCode calls.

Offline Commands
----------------

Run these in normal development and CI:

```
python -m unittest harness_eval.tests
python -m harness_eval.runner validate-samples
python -m harness_eval.runner validate-matrix
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
- `response_id` present when provider returns one,
- `usage_present` shows whether token usage metadata came back in the response.

If the API usage dashboard still shows nothing after a successful live smoke, preserve the generated `temp/harness_eval/live_smoke_*.json` file and compare its response ID/time with the dashboard.

Live Sample Eval
----------------

This sends a normal synthetic issue job to OpenCode Zen and evaluates the returned JSON as a runner transcript:

```
export OPENCODE_API_KEY=...
export RUN_OPENCODE_LIVE_TESTS=true
python -m harness_eval.runner live-sample harness_eval/samples/origin_trace.json --live --model kimi-k2.5 --write-result
```

This is not a replacement for a real Pi run with tools. It tests the model/provider's ability to complete the issue job shape and makes an actual OpenCode Zen request for dashboard verification.

Evaluating Real Pi Runs
-----------------------

Run the real Pi command as a black box:

```
python -m harness_eval.runner run-harness harness_eval/samples/origin_trace.json -- <your-pi-command-and-args>
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
