from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import io
import json
import subprocess
import sys
import unittest

from harness_eval.runner import _ensure_live_enabled, cmd_live_sample, cmd_live_smoke
from harness_eval import pi_runner
from harness_eval.pi_runner import extract_transcript_from_pi_jsonl
from harness_eval.evaluator import (
    DEFAULT_RAW_MODEL_ID,
    OPENCODE_ZEN_CHAT_COMPLETIONS_ENDPOINT,
    build_chat_completion_payload,
    build_job_packet,
    evaluate_transcript,
    load_json,
    sample_paths,
    validate_golden_alignment,
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

    def test_extra_node_in_allowed_path_fails_node_allowlist(self):
        sample = {
            'id': 'same_file_precision',
            'expect': {
                'required_tools': ['rank_issue_candidates', 'load_focus_graph', 'load_code_context'],
                'forbidden_tools': ['shell', 'filesystem', 'network'],
                'node_ids': ['api/services.py::get_repo_analysis', 'api/services.py::_build_and_store_analysis'],
                'allowed_node_ids': ['api/services.py::get_repo_analysis', 'api/services.py::_build_and_store_analysis'],
                'allowed_paths': ['api/services.py'],
            },
        }
        transcript = {
            'sample_id': 'same_file_precision',
            'variant_id': 'overbroad-same-file',
            'tool_calls': [
                {'name': 'rank_issue_candidates', 'arguments': {}},
                {'name': 'load_focus_graph', 'arguments': {}},
                {'name': 'load_code_context', 'arguments': {}},
            ],
            'final': {
                'hypotheses': [
                    {'node_id': 'api/services.py::get_repo_analysis', 'confidence': 0.8},
                    {'node_id': 'api/services.py::_build_and_store_analysis', 'confidence': 0.7},
                    {'node_id': 'api/services.py::delete_share_token', 'confidence': 0.2},
                ],
                'investigation_path': [
                    {'node_id': 'api/services.py::get_repo_analysis', 'path': 'api/services.py'},
                    {'node_id': 'api/services.py::_build_and_store_analysis', 'path': 'api/services.py'},
                    {'node_id': 'api/services.py::delete_share_token', 'path': 'api/services.py'},
                ],
                'confidence': {'score': 0.7},
            },
        }

        report = evaluate_transcript(sample, transcript)
        checks = {check['name']: check for check in report['checks']}

        self.assertFalse(report['passed'])
        self.assertTrue(checks['expected_nodes']['passed'])
        self.assertTrue(checks['path_allowlist']['passed'])
        self.assertFalse(checks['node_allowlist']['passed'])

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

    def test_repo_job_packet_sends_repo_not_precomputed_artifact(self):
        sample = load_json(ROOT / 'samples' / 'repo_parser_timeout.json')

        job = build_job_packet(sample)
        rendered = json.dumps(job, ensure_ascii=False)

        self.assertIn('repo', job)
        self.assertNotIn('artifact', job)
        self.assertIn('harness_eval/fixtures/repos/parser_timeout_repo', rendered)
        self.assertNotIn('expect', rendered)
        self.assertNotIn('allowed_node_ids', rendered)

    def test_repo_samples_match_judge_consensus_golden(self):
        checks = validate_golden_alignment(ROOT / 'samples', ROOT / 'golden' / 'repo_issue_consensus.json')

        self.assertTrue(all(check.passed for check in checks), [check.__dict__ for check in checks])


class HarnessEvalLiveGuardTests(unittest.TestCase):
    def test_live_commands_require_explicit_live_flag(self):
        enabled, reason = _ensure_live_enabled(Namespace(live=False))

        self.assertFalse(enabled)
        self.assertIn('--live', reason)

    def _live_args(self, **overrides):
        values = {
            'live': True,
            'model': 'kimi-k2.5',
            'endpoint': OPENCODE_ZEN_CHAT_COMPLETIONS_ENDPOINT,
            'timeout': 10,
            'write_result': False,
        }
        values.update(overrides)
        return Namespace(**values)

    def _capture_json(self, func, args):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = func(args)
        return code, json.loads(stream.getvalue())

    def test_live_smoke_fails_without_usage_receipt(self):
        response = {'id': 'chatcmpl-test', 'choices': [{'message': {'content': '{"ok":true}'}}]}

        with patch.dict('os.environ', {'RUN_OPENCODE_LIVE_TESTS': 'true', 'OPENCODE_API_KEY': 'test-key'}, clear=False):
            with patch('harness_eval.runner._post_json', return_value=(200, {}, response)):
                code, payload = self._capture_json(cmd_live_smoke, self._live_args())

        self.assertEqual(code, 1)
        self.assertFalse(payload['passed'])
        self.assertFalse(payload['usage_present'])
        self.assertFalse(payload['usage_receipt_present'])

    def test_live_smoke_passes_with_usage_receipt(self):
        response = {
            'id': 'chatcmpl-test',
            'choices': [{'message': {'content': '{"ok":true}'}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        }

        with patch.dict('os.environ', {'RUN_OPENCODE_LIVE_TESTS': 'true', 'OPENCODE_API_KEY': 'test-key'}, clear=False):
            with patch('harness_eval.runner._post_json', return_value=(200, {}, response)):
                code, payload = self._capture_json(cmd_live_smoke, self._live_args())

        self.assertEqual(code, 0)
        self.assertTrue(payload['passed'])
        self.assertTrue(payload['usage_present'])
        self.assertTrue(payload['usage_receipt_present'])
        self.assertEqual(payload['dashboard_correlation']['response_id'], 'chatcmpl-test')

    def test_live_sample_reports_non_json_model_output_as_parse_failure(self):
        response = {
            'id': 'chatcmpl-test',
            'choices': [{'message': {'content': 'I will explain first, then maybe JSON.'}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        }
        args = self._live_args(sample=str(ROOT / 'samples' / 'origin_trace.json'))

        with patch.dict('os.environ', {'RUN_OPENCODE_LIVE_TESTS': 'true', 'OPENCODE_API_KEY': 'test-key'}, clear=False):
            with patch('harness_eval.runner._post_json', return_value=(200, {}, response)):
                code, payload = self._capture_json(cmd_live_sample, args)

        self.assertEqual(code, 1)
        self.assertFalse(payload['passed'])
        self.assertTrue(payload['live_call_passed'])
        self.assertFalse(payload['content_json_valid'])
        self.assertIsNotNone(payload['json_parse_error'])
        self.assertTrue(payload['usage_receipt_present'])

    def test_live_sample_passes_only_with_usage_json_and_eval_success(self):
        transcript = load_json(ROOT / 'sample_transcripts' / 'good_origin_trace.json')
        response = {
            'id': 'chatcmpl-test',
            'choices': [{'message': {'content': json.dumps(transcript)}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        }
        args = self._live_args(sample=str(ROOT / 'samples' / 'origin_trace.json'))

        with patch.dict('os.environ', {'RUN_OPENCODE_LIVE_TESTS': 'true', 'OPENCODE_API_KEY': 'test-key'}, clear=False):
            with patch('harness_eval.runner._post_json', return_value=(200, {}, response)):
                code, payload = self._capture_json(cmd_live_sample, args)

        self.assertEqual(code, 0)
        self.assertTrue(payload['passed'])
        self.assertTrue(payload['eval_passed'])
        self.assertTrue(payload['live_call_passed'])
        self.assertTrue(payload['content_json_valid'])


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


class PiRunnerEventParserTests(unittest.TestCase):
    def test_extracts_finish_tool_result_details_from_pi_jsonl(self):
        transcript = load_json(ROOT / 'sample_transcripts' / 'good_origin_trace.json')
        event = {
            'type': 'message_update',
            'message': {
                'responseId': 'chatcmpl-test',
                'usage': {'input': 10, 'output': 20, 'totalTokens': 30},
                'content': [
                    {
                        'role': 'toolResult',
                        'toolName': 'finish_issue_map_transcript',
                        'details': transcript,
                    }
                ],
            },
        }

        parsed, metadata = extract_transcript_from_pi_jsonl(json.dumps(event))

        self.assertEqual(parsed['sample_id'], 'origin_trace')
        self.assertEqual(parsed['final']['confidence']['score'], 0.84)
        self.assertEqual(metadata['response_ids'], ['chatcmpl-test'])
        self.assertEqual(metadata['usage']['totalTokens'], 30)

    def test_extracts_finish_tool_result_from_text_content(self):
        transcript = load_json(ROOT / 'sample_transcripts' / 'good_origin_trace.json')
        event = {
            'type': 'message_update',
            'message': {
                'content': [
                    {
                        'role': 'toolResult',
                        'toolName': 'finish_issue_map_transcript',
                        'content': [{'type': 'text', 'text': json.dumps(transcript)}],
                    }
                ],
            },
        }

        parsed, metadata = extract_transcript_from_pi_jsonl('not-json\n' + json.dumps(event))

        self.assertEqual(parsed['sample_id'], 'origin_trace')
        self.assertIn('non_json_line_preview', metadata)

    def test_extracts_finish_tool_result_from_pi_tool_results_array(self):
        transcript = load_json(ROOT / 'sample_transcripts' / 'good_origin_trace.json')
        event = {
            'type': 'message_update',
            'message': {'content': [{'type': 'toolCall', 'name': 'finish_issue_map_transcript'}]},
            'toolResults': [
                {
                    'role': 'toolResult',
                    'toolName': 'finish_issue_map_transcript',
                    'details': transcript,
                }
            ],
        }

        parsed, _metadata = extract_transcript_from_pi_jsonl(json.dumps(event))

        self.assertEqual(parsed['sample_id'], 'origin_trace')
        self.assertEqual(parsed['tool_calls'][0]['name'], 'rank_issue_candidates')


class PiRunnerCommandTests(unittest.TestCase):
    def test_pi_runner_emits_transcript_from_finish_tool_result(self):
        job = build_job_packet(load_json(ROOT / 'samples' / 'origin_trace.json'))
        transcript = load_json(ROOT / 'sample_transcripts' / 'good_origin_trace.json')
        event = {
            'type': 'message_update',
            'message': {
                'responseId': 'chatcmpl-test',
                'usage': {'input': 10, 'output': 20, 'totalTokens': 30},
            },
            'toolResults': [
                {
                    'role': 'toolResult',
                    'toolName': 'finish_issue_map_transcript',
                    'details': transcript,
                }
            ],
        }
        completed = subprocess.CompletedProcess(args=['npx'], returncode=0, stdout=json.dumps(event), stderr='')
        stream = io.StringIO()

        with patch.dict('os.environ', {'RUN_OPENCODE_LIVE_TESTS': 'true', 'OPENCODE_API_KEY': 'test-key'}, clear=False):
            with patch('sys.stdin', io.StringIO(json.dumps(job))):
                with patch('harness_eval.pi_runner.subprocess.run', return_value=completed) as run:
                    with redirect_stdout(stream):
                        code = pi_runner.main(['--live'])

        payload = json.loads(stream.getvalue())
        _command, kwargs = run.call_args

        self.assertEqual(code, 0)
        self.assertEqual(payload['sample_id'], 'origin_trace')
        self.assertEqual(payload['pi_metadata']['response_ids'], ['chatcmpl-test'])
        self.assertIn('--no-builtin-tools', kwargs['args'] if 'args' in kwargs else run.call_args.args[0])
        self.assertIn('HARNESS_EVAL_JOB', kwargs['env'])

    def test_pi_runner_uses_repo_tools_for_repo_jobs(self):
        job = build_job_packet(load_json(ROOT / 'samples' / 'repo_parser_timeout.json'))
        args = Namespace(
            pi_bin='npx',
            pi_package='@earendil-works/pi-coding-agent@0.78.0',
            extension=ROOT / 'pi' / 'issue_map_extension.ts',
            provider='opencode',
            model='kimi-k2.5',
        )

        command = pi_runner.build_pi_command(args, job)

        self.assertIn('list_repo_files,search_repo_symbols,read_repo_file,finish_issue_map_transcript', command)
        self.assertNotIn('rank_issue_candidates,load_focus_graph,load_code_context,finish_issue_map_transcript', command)


if __name__ == '__main__':
    unittest.main()
