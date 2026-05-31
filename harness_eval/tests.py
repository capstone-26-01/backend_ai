from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import json
import subprocess
import sys
import unittest

from harness_eval.runner import _ensure_live_enabled
from harness_eval.evaluator import (
    DEFAULT_RAW_MODEL_ID,
    OPENCODE_ZEN_CHAT_COMPLETIONS_ENDPOINT,
    build_chat_completion_payload,
    build_job_packet,
    evaluate_transcript,
    load_json,
    sample_paths,
    validate_matrix,
    validate_sample,
)


ROOT = Path(__file__).resolve().parent


class HarnessEvalSampleTests(unittest.TestCase):
    def test_all_samples_are_valid(self):
        paths = sample_paths()
        self.assertGreaterEqual(len(paths), 3)
        for path in paths:
            with self.subTest(path=path.name):
                checks = validate_sample(load_json(path))
                self.assertTrue(all(check.passed for check in checks), [check.__dict__ for check in checks])

    def test_matrix_tracks_raw_and_opencode_model_ids_separately(self):
        matrix = load_json(ROOT / 'harness_matrix.sample.json')
        checks = validate_matrix(matrix)
        self.assertTrue(all(check.passed for check in checks), [check.__dict__ for check in checks])

        kimi_variant = next(variant for variant in matrix['variants'] if variant['id'] == 'opencode-kimi-k25-review-only')
        self.assertEqual(kimi_variant['model']['raw_model_id'], 'kimi-k2.5')
        self.assertEqual(kimi_variant['model']['opencode_model_id'], 'opencode/kimi-k2.5')
        self.assertEqual(kimi_variant['endpoint']['chat_completions_url'], OPENCODE_ZEN_CHAT_COMPLETIONS_ENDPOINT)

    def test_good_transcript_passes_expected_tool_and_node_checks(self):
        sample = load_json(ROOT / 'samples' / 'origin_trace.json')
        transcript = load_json(ROOT / 'sample_transcripts' / 'good_origin_trace.json')

        report = evaluate_transcript(sample, transcript)

        self.assertTrue(report['passed'])
        self.assertEqual(report['score'], 1.0)
        self.assertIn('api/services.py::_build_and_store_analysis', report['found_node_ids'])
        self.assertIn('parser/services.py::parse_repo', report['found_node_ids'])

    def test_forbidden_tool_transcript_fails(self):
        sample = load_json(ROOT / 'samples' / 'prompt_injection_guard.json')
        transcript = load_json(ROOT / 'sample_transcripts' / 'bad_forbidden_tool.json')

        report = evaluate_transcript(sample, transcript)
        checks = {check['name']: check for check in report['checks']}

        self.assertFalse(report['passed'])
        self.assertFalse(checks['forbidden_tools']['passed'])
        self.assertIn('network', report['tool_calls'])

    def test_hallucinated_paths_and_missing_nodes_fail(self):
        sample = load_json(ROOT / 'samples' / 'schema_adherence.json')
        transcript = {
            'sample_id': 'schema_adherence',
            'variant_id': 'bad-paths',
            'tool_calls': [{'name': 'rank_issue_candidates', 'arguments': {}}],
            'final': {
                'hypotheses': [{'node_id': 'api/services.py::missing', 'confidence': 0.8}],
                'investigation_path': [{'node_id': 'api/services.py::missing', 'path': '../secret.py'}],
                'confidence': {'score': 1.2},
            },
        }

        report = evaluate_transcript(sample, transcript)
        checks = {check['name']: check for check in report['checks']}

        self.assertFalse(report['passed'])
        self.assertFalse(checks['expected_nodes']['passed'])
        self.assertFalse(checks['path_allowlist']['passed'])
        self.assertFalse(checks['confidence_range']['passed'])

    def test_chat_completion_payload_uses_raw_model_id_for_direct_api(self):
        sample = load_json(ROOT / 'samples' / 'origin_trace.json')
        variant = {
            'model': {
                'opencode_model_id': 'opencode/kimi-k2.5',
                'raw_model_id': DEFAULT_RAW_MODEL_ID,
            }
        }

        payload = build_chat_completion_payload(sample, variant)

        self.assertEqual(payload['model'], 'kimi-k2.5')
        self.assertIn('messages', payload)
        self.assertIn('origin_trace', payload['messages'][1]['content'])
        self.assertNotIn('expect', payload['messages'][1]['content'])
        self.assertNotIn('you are a harness', payload['messages'][1]['content'].lower())

    def test_job_packet_hides_expected_answers_from_system_under_test(self):
        sample = load_json(ROOT / 'samples' / 'origin_trace.json')

        job = build_job_packet(sample)
        rendered = json.dumps(job, ensure_ascii=False)

        self.assertEqual(job['job_id'], 'origin_trace')
        self.assertNotIn('expect', job)
        self.assertNotIn('required_tools', rendered)
        self.assertNotIn('allowed_paths', rendered)
        self.assertIn('api/services.py::_build_and_store_analysis', rendered)


class HarnessEvalLiveGuardTests(unittest.TestCase):
    def test_live_commands_require_explicit_live_flag(self):
        enabled, reason = _ensure_live_enabled(Namespace(live=False))

        self.assertFalse(enabled)
        self.assertIn('--live', reason)


class HarnessEvalBlackBoxRunnerTests(unittest.TestCase):
    def _run_harness(self, fixture: str, sample: str = 'origin_trace.json') -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                '-m',
                'harness_eval.runner',
                'run-harness',
                str(ROOT / 'samples' / sample),
                '--',
                sys.executable,
                str(ROOT / 'fixtures' / fixture),
            ],
            cwd=ROOT.parent,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_black_box_runner_scores_good_harness_command(self):
        completed = self._run_harness('fake_good_harness.py')
        report = json.loads(completed.stdout)

        self.assertEqual(completed.returncode, 0)
        self.assertTrue(report['passed'])
        self.assertEqual(report['harness_returncode'], 0)
        self.assertIn('rank_issue_candidates', report['tool_calls'])

    def test_black_box_runner_fails_bad_harness_command(self):
        completed = self._run_harness('fake_bad_harness.py')
        report = json.loads(completed.stdout)
        checks = {check['name']: check for check in report['checks']}

        self.assertEqual(completed.returncode, 1)
        self.assertFalse(report['passed'])
        self.assertFalse(checks['forbidden_tools']['passed'])
        self.assertFalse(checks['expected_nodes']['passed'])
        self.assertFalse(checks['path_allowlist']['passed'])


if __name__ == '__main__':
    unittest.main()
