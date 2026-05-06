from django.test import TestCase
from unittest.mock import patch
import shutil

from django.conf import settings
from django.test import override_settings

from api.services import get_repo_analysis
from llm.services import answer_question, _build_context, _rank_files


class SelectiveQuestionAnsweringTests(TestCase):
    @patch('llm.services.client.chat.completions.create')
    def test_answer_question_limits_context_to_ranked_files(self, create_mock):
        analysis = {
            'revision': 'abc123',
            'file_contents': {
                'llava/model/builder.py': '# llava/model/builder.py\nprint("ok")\n',
                'llava/eval/run_llava.py': '# llava/eval/run_llava.py\nprint("ok")\n',
            },
            'nodes': [
                {'id': 'llava/model/builder.py::load_pretrained_model', 'label': 'load_pretrained_model', 'type': 'function', 'file': 'llava/model/builder.py'},
                {'id': 'llava/eval/run_llava.py::eval_model', 'label': 'eval_model', 'type': 'function', 'file': 'llava/eval/run_llava.py'},
            ],
            'edges': [],
        }
        create_mock.return_value = type(
            'Completion',
            (),
            {'choices': [type('Choice', (), {'message': type('Message', (), {'content': 'builder.py에서 처리합니다.'})()})]},
        )()

        response = answer_question('owner/repo', analysis, 'Where is load_pretrained_model defined?')

        self.assertEqual(response['citations'], ['llava/model/builder.py'])
        prompt = create_mock.call_args.kwargs['messages'][1]['content']
        self.assertIn('llava/model/builder.py', prompt)
        self.assertNotIn('llava/eval/run_llava.py', prompt)
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
        (self.source_repo / 'pkg' / 'builder.py').write_text('def load_pretrained_model():\n    return "ok"\n', encoding='utf-8')
        subprocess.run(['git', 'add', '.'], cwd=self.source_repo, check=True, capture_output=True, text=True)
        subprocess.run(['git', '-c', 'user.name=Test User', '-c', 'user.email=test@example.com', 'commit', '-m', 'init'], cwd=self.source_repo, check=True, capture_output=True, text=True)

    def tearDown(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)

    @patch('github_repo.services._repo_clone_url')
    @patch('llm.services.client.chat.completions.create')
    def test_cached_analysis_supports_qa_after_playground_cleanup(self, create_mock, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)
        analysis = get_repo_analysis('owner/repo')
        self.assertIsNotNone(analysis)
        shutil.rmtree(settings.PLAYGROUND_DIR, ignore_errors=True)
        create_mock.return_value = type(
            'Completion',
            (),
            {'choices': [type('Choice', (), {'message': type('Message', (), {'content': 'builder.py입니다.'})()})]},
        )()

        response = answer_question('owner/repo', analysis, 'Where is load_pretrained_model defined?')

        self.assertEqual(response['citations'], ['pkg/builder.py'])
        prompt = create_mock.call_args.kwargs['messages'][1]['content']
        self.assertIn('pkg/builder.py', prompt)

    def test_rank_files_prefers_exact_symbol_match(self):
        analysis = {
            'revision': 'abc123',
            'nodes': [
                {'id': 'llava/model/builder.py::load_pretrained_model', 'label': 'load_pretrained_model', 'type': 'function', 'file': 'llava/model/builder.py'},
                {'id': 'llava/model/language_model/llava_mistral.py::forward', 'label': 'forward', 'type': 'function', 'file': 'llava/model/language_model/llava_mistral.py'},
            ],
        }

        ranked_files = _rank_files(analysis, 'Where is load_pretrained_model defined?')

        self.assertEqual(ranked_files[0], 'llava/model/builder.py')

    def test_rank_files_boosts_eval_entrypoint_candidates(self):
        analysis = {
            'revision': 'abc123',
            'nodes': [
                {'id': 'llava/eval/run_llava.py::eval_model', 'label': 'eval_model', 'type': 'function', 'file': 'llava/eval/run_llava.py'},
                {'id': 'llava/model/llava_arch.py::build_vision_tower', 'label': 'build_vision_tower', 'type': 'function', 'file': 'llava/model/llava_arch.py'},
            ],
        }

        ranked_files = _rank_files(analysis, 'What is the evaluation entry point?')

        self.assertEqual(ranked_files[0], 'llava/eval/run_llava.py')

    @patch('llm.services.client.chat.completions.create')
    def test_answer_question_returns_non_answer_when_no_python_context_exists(self, create_mock):
        analysis = {
            'revision': 'abc123',
            'file_contents': {},
            'nodes': [{'id': 'README.md', 'label': 'README.md', 'type': 'file', 'file': 'README.md'}],
            'edges': [],
        }

        response = answer_question('owner/repo', analysis, 'What does this repo do?')

        self.assertEqual(response['citations'], [])
        self.assertEqual(response['answer'], '분석 가능한 Python 코드 문맥을 찾지 못했습니다.')
        create_mock.assert_not_called()
