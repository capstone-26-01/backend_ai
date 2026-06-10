import os
import json
import shutil
import sys
import tempfile
from unittest.mock import Mock, patch

from django.conf import settings
from django.test import TestCase, override_settings

from typing import cast

from api.services import get_repo_analysis
from api.test_utils import create_git_fixture_repo
from llm.issue_harness import (
    IssueHarnessUnavailable,
    build_issue_harness_job,
    run_issue_harness,
)
from llm.services import (
    _build_context,
    _generate_answer,
    _question_tokens,
    _rank_files,
    _stream_generate_answer,
    answer_question,
    stream_answer_question,
)
from llm.summaries import SUMMARY_KIND_ONBOARDING, SummaryUnavailable, generate_summary
from llm.tools import ArtifactToolbox, ToolLimitExceeded


class SelectiveQuestionAnsweringTests(TestCase):
    @patch('llm.services._answer_with_opencode_zen')
    def test_answer_question_limits_context_to_ranked_files(self, answer_with_opencode_zen):
        analysis = {
            'revision': 'abc123',
            'file_contents': {
                'sample_pkg/factory.py': '# sample_pkg/factory.py\nprint("ok")\n',
                'sample_pkg/runner.py': '# sample_pkg/runner.py\nprint("ok")\n',
            },
            'nodes': [
                {'id': 'sample_pkg/factory.py::load_component', 'label': 'load_component', 'type': 'function', 'file': 'sample_pkg/factory.py'},
                {'id': 'sample_pkg/runner.py::run_check', 'label': 'run_check', 'type': 'function', 'file': 'sample_pkg/runner.py'},
            ],
            'edges': [],
        }
        answer_with_opencode_zen.return_value = 'builder.py에서 처리합니다.'

        with patch.dict(os.environ, {'OPENCODE_API_KEY': 'test-opencode-key'}, clear=False):
            response = answer_question('owner/repo', analysis, 'Where is load_component defined?')

        self.assertEqual(response['citations'], ['sample_pkg/factory.py'])
        messages = answer_with_opencode_zen.call_args.args[0]
        self.assertIn('sample_pkg/factory.py', messages[1]['content'])
        self.assertNotIn('sample_pkg/runner.py', messages[1]['content'])
        self.assertEqual(response['answer'], 'builder.py에서 처리합니다.')

    def test_build_context_reports_only_included_files(self):
        analysis = {
            'file_contents': {
                'a.py': 'A' * 7900,
                'b.py': 'B' * 500,
            },
        }

        context, included_files = _build_context(analysis, ['a.py', 'b.py'], max_chars=8000)

        self.assertIn('a.py', context)
        self.assertEqual(included_files, ['a.py'])

    def test_build_context_keeps_first_large_ranked_file(self):
        context, included_files = _build_context({'file_contents': {'a.py': 'A' * 9000}}, ['a.py'], max_chars=8000)

        self.assertTrue(context)
        self.assertEqual(included_files, ['a.py'])

    def test_question_tokens_preserve_korean_terms(self):
        tokens = _question_tokens('이 저장소의 시작 지점은 어디인가요?')

        self.assertIn('저장소의', tokens)
        self.assertIn('시작', tokens)
        self.assertIn('지점은', tokens)

    @patch('llm.services._generate_answer')
    def test_selected_node_context_prioritizes_neighbors(self, generate_answer):
        analysis = {
            'revision': 'abc123',
            'file_contents': {
                'service/api.py': (
                    'from service.core import build_payload\n\n'
                    '@app.get("/")\n'
                    'def route():\n'
                    '    return build_payload()\n'
                ),
                'service/core.py': (
                    'def build_payload():\n'
                    '    return {"ok": True}\n'
                ),
                'service/unused.py': 'def unused():\n    return None\n',
            },
            'nodes': [
                {'id': 'service/api.py::route', 'kind': 'function', 'label': 'route', 'path': 'service/api.py', 'start_line': 4, 'end_line': 5},
                {'id': 'service/core.py::build_payload', 'kind': 'function', 'label': 'build_payload', 'path': 'service/core.py', 'start_line': 1, 'end_line': 2},
                {'id': 'service/unused.py::unused', 'kind': 'function', 'label': 'unused', 'path': 'service/unused.py', 'start_line': 1, 'end_line': 2},
            ],
            'edges': [
                {'id': 'e1', 'kind': 'calls', 'source': 'service/api.py::route', 'target': 'service/core.py::build_payload', 'path': 'service/api.py'},
            ],
            'entrypoints': [{'id': 'service/api.py::route', 'kind': 'web_route', 'path': 'service/api.py'}],
            'key_modules': [],
        }
        generate_answer.return_value = 'route가 build_payload를 호출합니다.'

        response = answer_question(
            'owner/repo',
            analysis,
            '이 route는 무엇을 호출하나요?',
            selected_node_id='service/api.py::route',
            max_context_files=2,
        )

        self.assertEqual(response['citations'], ['service/api.py', 'service/core.py'])
        self.assertIn('service/api.py::route', response['selected_nodes'])
        self.assertIn('service/core.py::build_payload', response['selected_nodes'])
        self.assertEqual(response['context_files'], ['service/api.py', 'service/core.py'])
        messages = generate_answer.call_args.args[0]
        self.assertIn('service/api.py::route', messages[1]['content'])
        self.assertIn('service/core.py::build_payload', messages[1]['content'])
        self.assertNotIn('service/unused.py', messages[1]['content'])

    @patch('llm.services._generate_answer')
    def test_invalid_selected_node_returns_warning_and_ranked_context(self, generate_answer):
        analysis = {
            'revision': 'abc123',
            'file_contents': {
                'pkg/factory.py': 'def load_component():\n    return "ok"\n',
            },
            'nodes': [
                {'id': 'pkg/factory.py::load_component', 'kind': 'function', 'label': 'load_component', 'path': 'pkg/factory.py', 'start_line': 1, 'end_line': 2},
            ],
            'edges': [],
        }
        generate_answer.return_value = 'factory입니다.'

        response = answer_question(
            'owner/repo',
            analysis,
            'Where is load_component defined?',
            selected_node_id='missing.py::missing',
        )

        self.assertEqual(response['citations'], ['pkg/factory.py'])
        self.assertEqual(response['warnings'][0]['code'], 'invalid_selected_node')

    def test_rank_files_can_use_cached_summary_text_hook(self):
        analysis = {
            'revision': 'abc123',
            'summaries': {
                'repo_overview:summary.v1': {
                    'text': '인증 처리는 auth.py의 AuthService가 담당합니다.',
                    'source_nodes': ['pkg/auth.py::AuthService'],
                    'source_files': ['pkg/auth.py'],
                },
            },
            'nodes': [
                {'id': 'pkg/auth.py::AuthService', 'kind': 'class', 'label': 'AuthService', 'path': 'pkg/auth.py'},
                {'id': 'pkg/billing.py::BillingService', 'kind': 'class', 'label': 'BillingService', 'path': 'pkg/billing.py'},
            ],
        }

        ranked_files = _rank_files(analysis, '인증 처리는 어디에서 하나요?')

        self.assertEqual(ranked_files[0], 'pkg/auth.py')

    @patch('llm.summaries._generate_answer')
    def test_generate_onboarding_summary_uses_entrypoints_and_key_modules(self, generate_answer):
        analysis = {
            'repo': 'owner/repo',
            'revision': 'abc123',
            'file_contents': {
                'service/api.py': 'def route():\n    return "ok"\n',
                'service/core.py': 'def build_payload():\n    return {"ok": True}\n',
            },
            'nodes': [
                {'id': 'service/api.py::route', 'kind': 'function', 'label': 'route', 'path': 'service/api.py', 'start_line': 1, 'end_line': 2},
                {'id': 'module::service.core', 'kind': 'module', 'label': 'service.core', 'path': 'service/core.py'},
            ],
            'edges': [],
            'entrypoints': [{'id': 'service/api.py::route', 'kind': 'web_route', 'path': 'service/api.py'}],
            'key_modules': [{'id': 'module::service.core', 'path': 'service/core.py', 'score': 10}],
            'warnings': [],
        }
        generate_answer.return_value = 'API에서 시작해 core를 읽으세요.'

        summary = generate_summary(analysis, SUMMARY_KIND_ONBOARDING)

        self.assertEqual(summary['text'], 'API에서 시작해 core를 읽으세요.')
        self.assertIn('service/api.py::route', summary['source_nodes'])
        self.assertIn('module::service.core', summary['source_nodes'])
        messages = generate_answer.call_args.args[0]
        self.assertIn('entrypoints', messages[1]['content'])
        self.assertIn('key_modules', messages[1]['content'])

    @patch('llm.summaries._generate_answer', side_effect=RuntimeError('OPENCODE_API_KEY가 설정되지 않았습니다.'))
    def test_generate_summary_maps_model_unavailable(self, generate_answer):
        analysis = {
            'repo': 'owner/repo',
            'revision': 'abc123',
            'file_contents': {},
            'nodes': [],
            'edges': [],
            'entrypoints': [],
            'key_modules': [],
            'warnings': [],
        }

        with self.assertRaises(SummaryUnavailable):
            generate_summary(analysis, SUMMARY_KIND_ONBOARDING)

    @patch('llm.services._stream_generate_answer')
    def test_stream_answer_question_yields_meta_tokens_and_final_payload(self, stream_generate_answer):
        analysis = {
            'revision': 'abc123',
            'file_contents': {
                'pkg/app.py': 'def main():\n    return "ok"\n',
            },
            'nodes': [
                {'id': 'pkg/app.py::main', 'kind': 'function', 'label': 'main', 'path': 'pkg/app.py', 'start_line': 1, 'end_line': 2},
            ],
            'edges': [],
        }
        stream_generate_answer.return_value = iter(['main', '입니다'])

        events = list(stream_answer_question('owner/repo', analysis, 'main은 무엇인가요?'))

        self.assertEqual([event['event'] for event in events], ['meta', 'token', 'token', 'final'])
        self.assertEqual(events[0]['data']['context_files'], ['pkg/app.py'])
        self.assertEqual(events[1]['data']['text'], 'main')
        self.assertEqual(events[2]['data']['text'], '입니다')
        self.assertEqual(events[3]['data']['answer'], 'main입니다')
        self.assertEqual(events[3]['data']['citations'], ['pkg/app.py'])

    @patch('llm.services._stream_generate_answer')
    def test_stream_answer_question_returns_final_without_model_when_no_context(self, stream_generate_answer):
        analysis = {
            'revision': 'abc123',
            'file_contents': {},
            'nodes': [{'id': 'README.md', 'label': 'README.md', 'type': 'file', 'file': 'README.md'}],
            'edges': [],
        }

        events = list(stream_answer_question('owner/repo', analysis, '무엇을 하나요?'))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event'], 'final')
        self.assertEqual(events[0]['data']['answer'], '분석 가능한 Python 코드 문맥을 찾지 못했습니다.')
        self.assertEqual(events[0]['data']['warnings'][0]['code'], 'no_context')
        stream_generate_answer.assert_not_called()


class IssueHarnessRuntimeTests(TestCase):
    def _analysis(self):
        return {
            'repo': 'owner/repo',
            'revision': 'abc123',
            'file_contents': {
                'api/services.py': 'def get_repo_analysis():\n    return parse_repo()\n',
                'parser/services.py': 'def parse_repo():\n    return {"nodes": []}\n',
            },
            'nodes': [
                {'id': 'api/services.py::get_repo_analysis', 'kind': 'function', 'label': 'get_repo_analysis', 'path': 'api/services.py', 'start_line': 1, 'end_line': 2},
                {'id': 'parser/services.py::parse_repo', 'kind': 'function', 'label': 'parse_repo', 'path': 'parser/services.py', 'start_line': 1, 'end_line': 2},
            ],
            'edges': [
                {'source': 'api/services.py::get_repo_analysis', 'target': 'parser/services.py::parse_repo', 'kind': 'calls', 'path': 'api/services.py'},
            ],
            'entrypoints': [{'id': 'api/services.py::get_repo_analysis', 'path': 'api/services.py'}],
            'key_modules': [],
        }

    def _job(self):
        return build_issue_harness_job(
            repo_path='owner/repo',
            revision='abc123',
            issue={'number': 42, 'title': 'Parser timeout starts in parse_repo', 'body': 'Trace shows parser/services.py parse_repo timeout.'},
            comments=[],
            evidence={'query': 'parser/services.py parse_repo timeout', 'file_mentions': [{'path': 'parser/services.py'}], 'symbol_mentions': [{'symbol': 'parse_repo'}]},
            candidates=[
                {
                    'rank': 1,
                    'score': 0.7,
                    'node_id': 'parser/services.py::parse_repo',
                    'node': {'path': 'parser/services.py'},
                    'reason': 'seed',
                    'evidence': [],
                }
            ],
            analysis=self._analysis(),
        )

    def _command_that_prints(self, payload: dict[str, object]) -> list[str]:
        code = 'import json; print(json.dumps(%r))' % payload
        return [sys.executable, '-c', code]

    def test_issue_harness_job_contains_bounded_graph_and_file_tools(self):
        job = self._job()

        tool_names = {tool['name'] for tool in job['available_tools']}
        self.assertEqual(job['task'], 'investigate_github_issue_origin')
        self.assertIn('get_issue_context', tool_names)
        self.assertIn('search_repo_symbols', tool_names)
        self.assertIn('read_node_context', tool_names)
        self.assertIn('parser/services.py::parse_repo', {node['id'] for node in job['graph']['nodes']})
        self.assertIn('parser/services.py', job['file_contents'])
        self.assertNotIn('/etc/passwd', job['file_contents'])

    def test_issue_harness_job_preserves_new_evidence_lists_with_caps(self):
        analysis = self._analysis()
        evidence = {
            'query': 'route config exception test quoted',
            'exception_mentions': [{'class': f'Value{index}Error'} for index in range(21)],
            'route_mentions': [{'route': f'/api/items/{index}/'} for index in range(41)],
            'config_mentions': [{'name': f'SERVICE_{index}_KEY'} for index in range(41)],
            'test_mentions': [{'name': f'test_case_{index}'} for index in range(41)],
            'quoted_strings': [{'text': f'quoted output {index}'} for index in range(21)],
        }

        job = build_issue_harness_job(
            repo_path='owner/repo',
            revision='abc123',
            issue={'number': 42, 'title': 'Parser timeout', 'body': ''},
            comments=[],
            evidence=evidence,
            candidates=[],
            analysis=analysis,
        )

        bounded = job['evidence']
        self.assertEqual(len(bounded['exception_mentions']), 20)
        self.assertEqual(len(bounded['route_mentions']), 40)
        self.assertEqual(len(bounded['config_mentions']), 40)
        self.assertEqual(len(bounded['test_mentions']), 40)
        self.assertEqual(len(bounded['quoted_strings']), 20)

    def test_issue_harness_job_ignores_malformed_artifact_collections(self):
        analysis = self._analysis()
        analysis['nodes'] = {'bad': 'not a node list'}
        analysis['edges'] = {'bad': 'not an edge list'}
        analysis['entrypoints'] = {'bad': 'not an entrypoint list'}
        evidence = {
            'query': 'parse_repo',
            'file_mentions': {'bad': 'not a mention list'},
            'exception_mentions': {'bad': 'not a mention list'},
            'route_mentions': {'bad': 'not a mention list'},
            'config_mentions': {'bad': 'not a mention list'},
            'test_mentions': {'bad': 'not a mention list'},
            'quoted_strings': {'bad': 'not a mention list'},
        }

        job = build_issue_harness_job(
            repo_path='owner/repo',
            revision='abc123',
            issue={'number': 42, 'title': 'Parser timeout', 'body': ''},
            comments=[],
            evidence=evidence,
            candidates=[],
            analysis=analysis,
        )

        self.assertEqual(job['graph']['nodes'], [])
        self.assertEqual(job['graph']['edges'], [])
        self.assertEqual(job['graph']['entrypoints'], [])
        self.assertEqual(job['evidence']['file_mentions'], [])
        self.assertEqual(job['evidence']['exception_mentions'], [])
        self.assertEqual(job['evidence']['route_mentions'], [])
        self.assertEqual(job['evidence']['config_mentions'], [])
        self.assertEqual(job['evidence']['test_mentions'], [])
        self.assertEqual(job['evidence']['quoted_strings'], [])

    def test_run_issue_harness_accepts_tool_backed_transcript(self):
        payload = {
            'variant_id': 'test-harness',
            'tool_calls': [
                {'name': 'get_issue_context', 'arguments': {}},
                {'name': 'list_repo_files', 'arguments': {}},
                {'name': 'search_repo_symbols', 'arguments': {'query': 'parse_repo'}},
                {'name': 'read_repo_file', 'arguments': {'path': 'parser/services.py'}},
            ],
            'final': {
                'hypotheses': [{'node_id': 'parser/services.py::parse_repo', 'confidence': 0.8, 'rationale': 'read file'}],
                'investigation_path': [{'node_id': 'parser/services.py::parse_repo', 'path': 'parser/services.py', 'why': 'inspect parser'}],
                'confidence': {'score': 0.8, 'reasons': ['tool-backed']},
            },
        }

        result = run_issue_harness(self._job(), command=self._command_that_prints(payload), timeout_seconds=5)

        self.assertEqual(result.output['hypotheses'][0]['node_id'], 'parser/services.py::parse_repo')
        self.assertEqual([call['name'] for call in result.tool_calls], ['get_issue_context', 'list_repo_files', 'search_repo_symbols', 'read_repo_file'])
        self.assertEqual(result.metadata['variant_id'], 'test-harness')

    def test_run_issue_harness_accepts_read_node_context_as_inspection(self):
        payload = {
            'variant_id': 'test-harness',
            'tool_calls': [
                {'name': 'get_issue_context', 'arguments': {}},
                {'name': 'list_repo_files', 'arguments': {}},
                {'name': 'search_repo_symbols', 'arguments': {'query': 'parse_repo'}},
                {'name': 'read_node_context', 'arguments': {'node_id': 'parser/services.py::parse_repo'}},
            ],
            'final': {
                'hypotheses': [{'node_id': 'parser/services.py::parse_repo', 'confidence': 0.8, 'rationale': 'read node context'}],
                'investigation_path': [{'node_id': 'parser/services.py::parse_repo', 'path': 'parser/services.py', 'why': 'inspect parser node context'}],
                'confidence': {'score': 0.8, 'reasons': ['tool-backed']},
            },
        }

        result = run_issue_harness(self._job(), command=self._command_that_prints(payload), timeout_seconds=5)

        self.assertEqual(result.output['hypotheses'][0]['node_id'], 'parser/services.py::parse_repo')
        self.assertEqual([call['name'] for call in result.tool_calls], ['get_issue_context', 'list_repo_files', 'search_repo_symbols', 'read_node_context'])

    def test_run_issue_harness_rejects_final_answer_without_tool_work(self):
        payload = {
            'tool_calls': [],
            'final': {
                'hypotheses': [{'node_id': 'parser/services.py::parse_repo', 'confidence': 0.8, 'rationale': 'guess'}],
                'investigation_path': [{'node_id': 'parser/services.py::parse_repo', 'path': 'parser/services.py', 'why': 'guess'}],
                'confidence': {'score': 0.8},
            },
        }

        with self.assertRaisesMessage(IssueHarnessUnavailable, 'without tool calls'):
            run_issue_harness(self._job(), command=self._command_that_prints(payload), timeout_seconds=5)

    def test_run_issue_harness_rejects_named_nodes_without_code_or_graph_inspection(self):
        payload = {
            'tool_calls': [
                {'name': 'get_issue_context', 'arguments': {}},
                {'name': 'list_repo_files', 'arguments': {}},
                {'name': 'search_repo_symbols', 'arguments': {'query': 'parse_repo'}},
            ],
            'final': {
                'hypotheses': [{'node_id': 'parser/services.py::parse_repo', 'confidence': 0.8, 'rationale': 'search only'}],
                'investigation_path': [{'node_id': 'parser/services.py::parse_repo', 'path': 'parser/services.py', 'why': 'search only'}],
                'confidence': {'score': 0.8},
            },
        }

        with self.assertRaisesMessage(IssueHarnessUnavailable, 'inspect code, node context, or graph neighbors'):
            run_issue_harness(self._job(), command=self._command_that_prints(payload), timeout_seconds=5)

    def test_run_issue_harness_rejects_missing_issue_context(self):
        payload = {
            'tool_calls': [
                {'name': 'list_repo_files', 'arguments': {}},
                {'name': 'search_repo_symbols', 'arguments': {'query': 'parse_repo'}},
                {'name': 'read_repo_file', 'arguments': {'path': 'parser/services.py'}},
            ],
            'final': {'hypotheses': [], 'investigation_path': [], 'confidence': {'score': 0.0}},
        }

        with self.assertRaisesMessage(IssueHarnessUnavailable, 'bounded issue context'):
            run_issue_harness(self._job(), command=self._command_that_prints(payload), timeout_seconds=5)

    def test_run_issue_harness_rejects_tool_call_budget_overrun(self):
        tool_calls = [
            {'name': 'get_issue_context', 'arguments': {}},
            {'name': 'list_repo_files', 'arguments': {}},
            {'name': 'search_repo_symbols', 'arguments': {'query': 'parse_repo'}},
            {'name': 'read_repo_file', 'arguments': {'path': 'parser/services.py'}},
        ]
        tool_calls.extend({'name': 'get_node', 'arguments': {'node_id': 'parser/services.py::parse_repo'}} for _ in range(31))
        payload = {
            'tool_calls': tool_calls,
            'final': {
                'hypotheses': [{'node_id': 'parser/services.py::parse_repo', 'confidence': 0.8, 'rationale': 'read file'}],
                'investigation_path': [{'node_id': 'parser/services.py::parse_repo', 'path': 'parser/services.py', 'why': 'inspect parser'}],
                'confidence': {'score': 0.8},
            },
        }

        with self.assertRaisesMessage(IssueHarnessUnavailable, 'exceeded 30 tool calls'):
            run_issue_harness(self._job(), command=self._command_that_prints(payload), timeout_seconds=5)

    def test_run_issue_harness_reports_failed_non_json_command_as_harness_failure(self):
        command = [sys.executable, '-c', 'import sys; sys.stderr.write("boom"); raise SystemExit(3)']

        with self.assertRaises(IssueHarnessUnavailable) as caught:
            run_issue_harness(self._job(), command=command, timeout_seconds=5)

        self.assertEqual(caught.exception.code, 'harness_failed')
        self.assertIn('boom', caught.exception.message)

    def test_default_pi_runner_fails_closed_without_opencode_key(self):
        with patch.dict(os.environ, {'OPENCODE_API_KEY': ''}, clear=False):
            with self.assertRaisesMessage(IssueHarnessUnavailable, 'OPENCODE_API_KEY'):
                run_issue_harness(self._job(), timeout_seconds=5)

    def test_pi_runner_deletes_bounded_job_file_after_success(self):
        from llm import pi_issue_runner

        transcript = {
            'sample_id': 'github:owner/repo#42@abc123',
            'variant_id': 'runtime-pi-issue-harness',
            'tool_calls': [
                {'name': 'get_issue_context', 'arguments': {}},
                {'name': 'list_repo_files', 'arguments': {}},
            ],
            'final': {'hypotheses': [], 'investigation_path': [], 'confidence': {'score': 0.0}},
        }
        event = {'role': 'toolResult', 'toolName': 'finish_issue_map_transcript', 'details': transcript}

        with tempfile.TemporaryDirectory() as temp_dir:
            job_dir = os.path.join(temp_dir, 'jobs')
            completed = Mock(returncode=0, stdout=json.dumps(event), stderr='')
            with (
                patch.object(pi_issue_runner, 'DEFAULT_JOB_DIR', pi_issue_runner.Path(job_dir)),
                patch('llm.pi_issue_runner.subprocess.run', return_value=completed),
                patch.dict(os.environ, {'OPENCODE_API_KEY': 'test-key', 'ISSUE_HARNESS_KEEP_JOB_FILES': ''}, clear=False),
                patch('sys.stdin', Mock(read=Mock(return_value=json.dumps(self._job())))),
                patch('builtins.print'),
            ):
                status = pi_issue_runner.main(['--timeout', '5'])

            self.assertEqual(status, 0)
            self.assertEqual(os.listdir(job_dir), [])

    def test_pi_runner_transcript_parser_ignores_malformed_finish_tool_text(self):
        from llm.pi_issue_runner import extract_transcript

        event = {
            'role': 'toolResult',
            'toolName': 'finish_issue_map_transcript',
            'content': [{'type': 'text', 'text': 'not json'}],
        }

        transcript, metadata = extract_transcript(json.dumps(event))

        self.assertIsNone(transcript)
        self.assertEqual(metadata['event_count'], 1)

    def test_pi_runner_command_and_prompt_include_read_node_context(self):
        from argparse import Namespace
        from llm import pi_issue_runner

        args = Namespace(
            pi_bin='npx',
            pi_package='@earendil-works/pi-coding-agent@0.78.0',
            extension=pi_issue_runner.DEFAULT_EXTENSION,
            provider='opencode',
            model='kimi-k2.5',
        )

        command = pi_issue_runner.build_command(args, self._job())
        tools = command[command.index('--tools') + 1]
        prompt = pi_issue_runner.build_prompt(self._job())

        self.assertIn('get_neighbors,read_node_context,finish_issue_map_transcript', tools)
        self.assertIn('read_node_context', prompt)
        self.assertLess(tools.index('read_node_context'), tools.index('finish_issue_map_transcript'))

    def test_runtime_finish_gate_accepts_read_node_context_tool_name(self):
        extension_path = settings.BASE_DIR / 'llm' / 'pi_issue_extension.ts'
        text = extension_path.read_text(encoding='utf-8')

        self.assertIn('name: "read_node_context"', text)
        self.assertIn('recordToolCall("read_node_context"', text)
        self.assertIn('toolNames.includes("read_node_context")', text)
        self.assertNotIn('toolNames.includes("readNodeContext")', text)


class ArtifactToolboxTests(TestCase):
    def _analysis(self):
        return {
            'repo': 'owner/repo',
            'revision': 'abc123',
            'file_contents': {
                'pkg/app.py': 'def main():\n    return helper()\n',
                'pkg/utils.py': 'def helper():\n    return "ok"\n',
            },
            'nodes': [
                {'id': 'pkg/app.py::main', 'kind': 'function', 'label': 'main', 'path': 'pkg/app.py', 'start_line': 1, 'end_line': 2},
                {'id': 'pkg/utils.py::helper', 'kind': 'function', 'label': 'helper', 'path': 'pkg/utils.py', 'start_line': 1, 'end_line': 2},
            ],
            'edges': [
                {'id': 'e1', 'kind': 'calls', 'source': 'pkg/app.py::main', 'target': 'pkg/utils.py::helper', 'path': 'pkg/app.py'},
            ],
            'entrypoints': [{'id': 'pkg/app.py::main', 'kind': 'main_function', 'path': 'pkg/app.py'}],
            'key_modules': [{'id': 'module::pkg.app', 'path': 'pkg/app.py', 'score': 5}],
            'summaries': {
                'repo_overview:summary.v1': {
                    'kind': 'repo_overview',
                    'text': 'main이 helper를 호출합니다.',
                    'source_nodes': ['pkg/app.py::main'],
                    'source_files': ['pkg/app.py'],
                },
            },
            'warnings': [],
        }

    def test_artifact_toolbox_only_reads_stored_file_contents(self):
        toolbox = ArtifactToolbox(self._analysis())

        valid = toolbox.get_file_excerpt('pkg/app.py')
        missing = toolbox.get_file_excerpt('/etc/passwd')

        self.assertIn('def main', valid['excerpt'])
        self.assertEqual(missing['error'], 'file_not_found')
        self.assertEqual(missing['path'], '/etc/passwd')

    def test_artifact_toolbox_rejects_missing_nodes_and_enforces_call_limit(self):
        toolbox = ArtifactToolbox(self._analysis(), max_tool_calls=1)

        missing = toolbox.get_node('pkg/app.py::missing')

        self.assertEqual(missing['error'], 'node_not_found')
        with self.assertRaises(ToolLimitExceeded):
            toolbox.get_entrypoints()


@override_settings(
    TEMP_DIR=settings.BASE_DIR / 'temp' / 'llm-snapshot-tests',
    PLAYGROUND_DIR=settings.BASE_DIR / 'temp' / 'llm-snapshot-tests' / 'playground',
)
class CachedQaSnapshotTests(TestCase):
    def setUp(self):
        self.source_repo = settings.TEMP_DIR / 'source-repo'
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        settings.PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)
        create_git_fixture_repo(self.source_repo, {'pkg/builder.py': 'def load_component():\n    return "ok"\n'}, commit_message='init')

    def tearDown(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)

    @patch('github_repo.services._repo_clone_url')
    @patch('llm.services._answer_with_opencode_zen')
    def test_cached_analysis_supports_qa_after_playground_cleanup(self, answer_with_opencode_zen, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)
        analysis = get_repo_analysis('owner/repo')
        self.assertIsNotNone(analysis)
        analysis = cast(dict[str, object], analysis)
        shutil.rmtree(settings.PLAYGROUND_DIR, ignore_errors=True)
        answer_with_opencode_zen.return_value = 'builder.py입니다.'

        with patch.dict(os.environ, {'OPENCODE_API_KEY': 'test-opencode-key'}, clear=False):
            response = answer_question('owner/repo', analysis, 'Where is load_component defined?')

        self.assertEqual(response['citations'], ['pkg/builder.py'])
        messages = answer_with_opencode_zen.call_args.args[0]
        self.assertIn('pkg/builder.py', messages[1]['content'])

    def test_rank_files_prefers_exact_symbol_match(self):
        analysis = {
            'revision': 'abc123',
            'nodes': [
                {'id': 'sample_pkg/factory.py::load_component', 'label': 'load_component', 'type': 'function', 'file': 'sample_pkg/factory.py'},
                {'id': 'sample_pkg/chat_model.py::forward', 'label': 'forward', 'type': 'function', 'file': 'sample_pkg/chat_model.py'},
            ],
        }

        ranked_files = _rank_files(analysis, 'Where is load_component defined?')

        self.assertEqual(ranked_files[0], 'sample_pkg/factory.py')

    def test_rank_files_boosts_eval_entrypoint_candidates(self):
        analysis = {
            'revision': 'abc123',
            'nodes': [
                {'id': 'sample_pkg/run_checks.py::main', 'label': 'main', 'type': 'function', 'file': 'sample_pkg/run_checks.py'},
                {'id': 'sample_pkg/model_layout.py::build_graph', 'label': 'build_graph', 'type': 'function', 'file': 'sample_pkg/model_layout.py'},
            ],
        }

        ranked_files = _rank_files(analysis, 'What is the evaluation entry point?')

        self.assertEqual(ranked_files[0], 'sample_pkg/run_checks.py')

    def test_rank_files_does_not_translate_controller_or_worker_to_serve(self):
        analysis = {
            'revision': 'abc123',
            'nodes': [
                {'id': 'infra/controller.py::main', 'label': 'main', 'type': 'function', 'file': 'infra/controller.py'},
                {'id': 'service/worker.py::main', 'label': 'main', 'type': 'function', 'file': 'service/worker.py'},
                {'id': 'serve/api.py::main', 'label': 'main', 'type': 'function', 'file': 'serve/api.py'},
            ],
        }

        ranked_files = _rank_files(analysis, 'How does controller connect to worker?')

        self.assertIn('infra/controller.py', ranked_files)
        self.assertIn('service/worker.py', ranked_files)
        self.assertNotEqual(ranked_files[0], 'serve/api.py')

    def test_answer_question_returns_non_answer_when_no_python_context_exists(self):
        analysis = {
            'revision': 'abc123',
            'file_contents': {},
            'nodes': [{'id': 'README.md', 'label': 'README.md', 'type': 'file', 'file': 'README.md'}],
            'edges': [],
        }

        response = answer_question('owner/repo', analysis, 'What does this repo do?')

        self.assertEqual(response['citations'], [])
        self.assertEqual(response['answer'], '분석 가능한 Python 코드 문맥을 찾지 못했습니다.')

    @patch('llm.services._generate_answer')
    def test_answer_question_does_not_call_model_without_python_context(self, generate_answer):
        analysis = {
            'revision': 'abc123',
            'file_contents': {},
            'nodes': [{'id': 'docs/guide.md', 'label': 'guide.md', 'type': 'file', 'file': 'docs/guide.md'}],
            'edges': [],
        }

        response = answer_question('owner/repo', analysis, '이 저장소는 무슨 일을 하나요?')

        self.assertEqual(response['citations'], [])
        generate_answer.assert_not_called()

    @patch('llm.services.requests.post')
    def test_generate_answer_calls_opencode_zen_chat_completion(self, requests_post):
        zen_response = Mock()
        zen_response.json.return_value = {
            'choices': [
                {'message': {'content': 'Zen 답변입니다.'}}
            ]
        }
        zen_response.raise_for_status.return_value = None
        requests_post.return_value = zen_response

        with (
            patch.dict(os.environ, {'OPENCODE_API_KEY': 'zen-key'}, clear=True),
            patch(
                'llm.services._opencode_config',
                return_value={
                    'chat_completions_url': 'https://opencode.ai/zen/v1/chat/completions',
                    'model': 'kimi-k2.6',
                    'max_tokens': 777,
                    'timeout_seconds': 60,
                },
            ),
        ):
            answer = _generate_answer([{'role': 'user', 'content': 'ping'}])

        self.assertEqual(answer, 'Zen 답변입니다.')
        requests_post.assert_called_once()
        self.assertIn('opencode.ai/zen/v1/chat/completions', requests_post.call_args.args[0])
        self.assertEqual(requests_post.call_args.kwargs['headers']['Authorization'], 'Bearer zen-key')
        self.assertEqual(requests_post.call_args.kwargs['headers']['Accept'], 'application/json')
        payload = requests_post.call_args.kwargs['json']
        self.assertEqual(payload['model'], 'kimi-k2.6')
        self.assertEqual(payload['max_tokens'], 777)
        self.assertNotIn('stream', payload)

    def test_generate_answer_requires_opencode_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesMessage(RuntimeError, 'OPENCODE_API_KEY'):
                _generate_answer([{'role': 'user', 'content': 'ping'}])

    @patch('llm.services.requests.post')
    def test_stream_generate_answer_parses_opencode_zen_sse_chunks(self, requests_post):
        zen_response = Mock()
        zen_response.iter_lines.return_value = iter(
            [
                'data: {"choices":[{"delta":{"content":"안"}}]}',
                'data: {"choices":[{"delta":{"content":"녕"}}]}',
                'data: [DONE]',
            ]
        )
        zen_response.raise_for_status.return_value = None
        requests_post.return_value = zen_response

        with patch.dict(os.environ, {'OPENCODE_API_KEY': 'zen-key'}, clear=True):
            chunks = list(_stream_generate_answer([{'role': 'user', 'content': 'ping'}], model='opencode/kimi-k2.5'))

        self.assertEqual(chunks, ['안', '녕'])
        self.assertTrue(requests_post.call_args.kwargs['stream'])
        self.assertEqual(requests_post.call_args.kwargs['headers']['Accept'], 'text/event-stream')
        payload = requests_post.call_args.kwargs['json']
        self.assertEqual(payload['model'], 'kimi-k2.5')
        self.assertTrue(payload['stream'])
        zen_response.close.assert_called_once()

    @patch('llm.services.requests.post')
    def test_generate_answer_rejects_model_outside_allowlist(self, requests_post):
        with (
            patch.dict(os.environ, {'OPENCODE_API_KEY': 'zen-key'}, clear=True),
            patch('llm.services._opencode_config', return_value={'model': 'kimi-k2.5', 'allowed_models': ['kimi-k2.5']}),
        ):
            with self.assertRaisesMessage(RuntimeError, '허용되지 않은 OpenCode Zen 모델입니다.'):
                _generate_answer([{'role': 'user', 'content': 'ping'}], model='kimi-k2.6')

        requests_post.assert_not_called()
