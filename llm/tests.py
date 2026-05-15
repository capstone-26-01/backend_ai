import os
import shutil
from unittest.mock import Mock, patch

from django.conf import settings
from django.test import TestCase, override_settings

from typing import cast

from api.services import get_repo_analysis
from llm.services import answer_question, _build_context, _rank_files


class SelectiveQuestionAnsweringTests(TestCase):
    @patch('llm.services._answer_with_openai')
    def test_answer_question_limits_context_to_ranked_files(self, answer_with_openai):
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
        answer_with_openai.return_value = 'builder.py에서 처리합니다.'

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'}, clear=False):
            response = answer_question('owner/repo', analysis, 'Where is load_component defined?')

        self.assertEqual(response['citations'], ['sample_pkg/factory.py'])
        messages = answer_with_openai.call_args.args[0]
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
        self.source_repo.mkdir(parents=True, exist_ok=True)

        import subprocess

        subprocess.run(['git', 'init', '-b', 'main'], cwd=self.source_repo, check=True, capture_output=True, text=True)
        (self.source_repo / 'pkg').mkdir(parents=True, exist_ok=True)
        (self.source_repo / 'pkg' / 'builder.py').write_text('def load_component():\n    return "ok"\n', encoding='utf-8')
        subprocess.run(['git', 'add', '.'], cwd=self.source_repo, check=True, capture_output=True, text=True)
        subprocess.run(
            ['git', '-c', 'user.name=Test User', '-c', 'user.email=test@example.com', 'commit', '-m', 'init'],
            cwd=self.source_repo,
            check=True,
            capture_output=True,
            text=True,
        )

    def tearDown(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)

    @patch('github_repo.services._repo_clone_url')
    @patch('llm.services._answer_with_openai')
    def test_cached_analysis_supports_qa_after_playground_cleanup(self, answer_with_openai, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)
        analysis = get_repo_analysis('owner/repo')
        self.assertIsNotNone(analysis)
        analysis = cast(dict[str, object], analysis)
        shutil.rmtree(settings.PLAYGROUND_DIR, ignore_errors=True)
        answer_with_openai.return_value = 'builder.py입니다.'

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'}, clear=False):
            response = answer_question('owner/repo', analysis, 'Where is load_component defined?')

        self.assertEqual(response['citations'], ['pkg/builder.py'])
        messages = answer_with_openai.call_args.args[0]
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

    @patch('llm.services.requests.post')
    @patch('llm.services._answer_with_openai')
    def test_answer_question_falls_back_to_gemini_when_openai_call_fails(self, answer_with_openai, requests_post):
        analysis = {
            'revision': 'abc123',
            'file_contents': {
                'sample_pkg/factory.py': 'def load_component():\n    return "ok"\n',
            },
            'nodes': [
                {'id': 'sample_pkg/factory.py::load_component', 'label': 'load_component', 'type': 'function', 'file': 'sample_pkg/factory.py'},
            ],
            'edges': [],
        }
        answer_with_openai.side_effect = RuntimeError('401 not_authorized_invalid_project')
        gemini_response = Mock()
        gemini_response.json.return_value = {
            'candidates': [
                {'content': {'parts': [{'text': 'Gemini가 대신 답변했습니다.'}]}}
            ]
        }
        gemini_response.raise_for_status.return_value = None
        requests_post.return_value = gemini_response

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'bad-key', 'GEMINI_API_KEY': 'gemini-key'}, clear=False):
            response = answer_question('owner/repo', analysis, 'Where is load_component defined?')

        self.assertEqual(response['answer'], 'Gemini가 대신 답변했습니다.')
        self.assertEqual(response['citations'], ['sample_pkg/factory.py'])
        requests_post.assert_called_once()
        self.assertEqual(requests_post.call_args.kwargs['headers']['x-goog-api-key'], 'gemini-key')
        payload = requests_post.call_args.kwargs['json']
        self.assertIn('질문: Where is load_component defined?', payload['contents'][0]['parts'][0]['text'])

    @patch('llm.services.requests.post')
    @patch('llm.services._answer_with_openai')
    def test_answer_question_uses_gemini_when_openai_key_is_missing(self, answer_with_openai, requests_post):
        analysis = {
            'revision': 'abc123',
            'file_contents': {
                'sample_pkg/factory.py': 'def load_component():\n    return "ok"\n',
            },
            'nodes': [
                {'id': 'sample_pkg/factory.py::load_component', 'label': 'load_component', 'type': 'function', 'file': 'sample_pkg/factory.py'},
            ],
            'edges': [],
        }
        gemini_response = Mock()
        gemini_response.json.return_value = {
            'candidates': [
                {'content': {'parts': [{'text': 'Gemini direct response'}]}}
            ]
        }
        gemini_response.raise_for_status.return_value = None
        requests_post.return_value = gemini_response

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'gemini-key'}, clear=True):
            response = answer_question('owner/repo', analysis, 'Where is load_component defined?')

        self.assertEqual(response['answer'], 'Gemini direct response')
        answer_with_openai.assert_not_called()
        requests_post.assert_called_once()
