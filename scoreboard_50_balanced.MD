# Pi Harness — 50-Sample Model Scoreboard (SWE-bench balanced)

Fixed 50-sample SWE-bench-derived localization suite (`samples_50_balanced`), run
through the Pi issue harness. All models run as **thinking models at `--thinking high`**.
A sample passes when a returned path matches one of the issue's gold source files.
Per-sample timeouts were recovered at a 300s Pi cap.

**Winner: `kimi-k2.6`** — highest accuracy; the three-way 94% tie (kimi-k2.6,
deepseek-v4-flash, glm-5.1) was broken by smallest average duration.

| model | completed | pass | fail | accuracy | avg duration (s) | p95 duration (s) | failure counts |
|---|---:|---:|---:|---:|---:|---:|---|
| kimi-k2.6 | 50 | 47 | 3 | 94.0% | 51.7 | 110.1 | no_gold_file:3 |
| deepseek-v4-flash | 50 | 47 | 3 | 94.0% | 66.0 | 142.4 | no_gold_file:3 |
| glm-5.1 | 50 | 47 | 3 | 94.0% | 79.3 | 140.1 | no_gold_file:3 |
| minimax-m2.7 | 50 | 46 | 4 | 92.0% | 42.0 | 91.6 | no_gold_file:4 |
| kimi-k2.5 | 50 | 46 | 4 | 92.0% | 64.2 | 167.5 | no_gold_file:3, pi_timeout:1 |
| claude-haiku-4-5 | 50 | 44 | 6 | 88.0% | 55.7 | 98.4 | no_gold_file:6 |
| gpt-5.4-nano | 50 | 38 | 12 | 76.0% | 80.6 | 148.5 | no_gold_file:12 |

Notes:
- `avg duration` / `p95 duration` = time from the model receiving the question until its
  final answer (full agent loop), averaged and 95th-percentile across the 50 samples.
- `no_gold_file` = the model returned a valid repository path, but it was not one of the
  issue's gold source files. (Pass requires overlapping at least one gold file.)
- `pi_timeout` = the Pi investigation loop exceeded the per-sample 300s cap and was
  aborted, so the sample is scored as a fail. Only one occurred: `kimi-k2.5` on
  `sphinx-doc__sphinx-7454`, where the loop ran ~322s on that heavy sample. It is a
  time-budget overrun, not a wrong-answer or API error.
