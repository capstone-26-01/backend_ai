from typing import cast
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
import importlib
import json
import subprocess
import shutil

from django.conf import settings
from django.http import HttpResponse
from django.test import TestCase, override_settings
from django.utils import timezone
from github_repo.services import (
    RepoIngestionError,
    get_file_content,
    get_file_content_or_raise,
    get_file_tree,
    get_repo_revision,
    get_repo_snapshot_or_raise,
    _repo_lock,
    _repo_lock_path,
    _run_git,
)
from api.artifacts import (
    GRAPH_ARTIFACT_SCHEMA_VERSION,
    ArtifactValidationError,
    build_graph_artifact,
    coerce_graph_artifact,
    validate_graph_artifact,
)
from api.diff import compare_graph_artifacts
from api.models import AnalysisArtifact, AnalysisRun, Repository, ShareLink
from api.serializers import extract_repo_path, is_safe_graph_id, is_safe_ref, is_safe_repo_file_path, is_safe_revision, is_safe_share_id
from api.services import get_artifact_by_revision, get_repo_analysis
from api.test_utils import (
    EVAL_RUBRIC,
    GOLDEN_FIXTURE_REPOS,
    commit_all,
    create_git_fixture_repo,
    create_named_fixture_repo,
    run_git,
    write_files,
)
from parser.services import parse_repo
import yaml

get_repo_analysis = importlib.import_module('api.services').get_repo_analysis
get_artifact_by_revision = importlib.import_module('api.services').get_artifact_by_revision


class DocsEndpointsTests(TestCase):
    def test_schema_endpoint_returns_openapi_document(self):
        response = cast(HttpResponse, self.client.get('/api/schema/'))

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            'application/vnd.oai.openapi',
            response.headers.get('Content-Type', ''),
        )

        payload = cast(dict[str, object], yaml.safe_load(response.content))
        paths = cast(dict[str, object], payload['paths'])

        self.assertIn('openapi', payload)
        self.assertIn('paths', payload)
        self.assertIn('/api/repo/', paths)
        self.assertIn('/api/analysis/', paths)
        self.assertIn('/api/qa/', paths)

    def test_swagger_docs_endpoint_renders_ui(self):
        response = cast(HttpResponse, self.client.get('/api/docs/'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'SwaggerUIBundle')
        self.assertContains(response, '/api/schema/')


class WorkspaceSettingsTests(TestCase):
    def test_workspace_directories_stay_inside_repo(self):
        self.assertEqual(settings.TEMP_DIR, settings.BASE_DIR / 'temp')
        self.assertEqual(settings.PLAYGROUND_DIR, settings.BASE_DIR / 'playground')
        self.assertTrue(settings.TEMP_DIR.is_dir())
        self.assertTrue(settings.PLAYGROUND_DIR.is_dir())

    def test_test_runner_ignores_production_ssl_redirect_settings(self):
        self.assertFalse(settings.SECURE_SSL_REDIRECT)
        self.assertFalse(settings.SESSION_COOKIE_SECURE)
        self.assertFalse(settings.CSRF_COOKIE_SECURE)
        self.assertIn('testserver', settings.ALLOWED_HOSTS)

    def test_repo_ingestion_limits_are_configured(self):
        self.assertGreater(settings.GITHUB_REPO_GIT_TIMEOUT_SECONDS, 0)
        self.assertGreater(settings.GITHUB_REPO_MAX_FILES, 0)
        self.assertGreater(settings.GITHUB_REPO_MAX_PYTHON_FILES, 0)
        self.assertGreater(settings.GITHUB_REPO_MAX_SINGLE_FILE_BYTES, 0)
        self.assertGreater(settings.GITHUB_REPO_MAX_TOTAL_ANALYZED_BYTES, 0)
        self.assertGreater(settings.GITHUB_REPO_REVISION_FETCH_DEPTH, 0)


class RepoUrlSerializerTests(TestCase):
    def test_extract_repo_path_accepts_owner_repo_only(self):
        self.assertEqual(extract_repo_path('https://github.com/owner/repo'), 'owner/repo')

    def test_extract_repo_path_accepts_trailing_slash(self):
        self.assertEqual(extract_repo_path('https://github.com/owner/repo/'), 'owner/repo')

    def test_extract_repo_path_rejects_extra_segments(self):
        self.assertIsNone(extract_repo_path('https://github.com/owner/repo/issues/1'))
        self.assertIsNone(extract_repo_path('https://github.com/owner/repo/tree/main'))

    def test_extract_repo_path_rejects_non_github_hosts(self):
        self.assertIsNone(extract_repo_path('https://example.com/owner/repo'))

    def test_extract_repo_path_rejects_dot_segments(self):
        self.assertIsNone(extract_repo_path('https://github.com/../repo'))
        self.assertIsNone(extract_repo_path('https://github.com/owner/..'))
        self.assertIsNone(extract_repo_path('https://github.com/./repo'))

    def test_extract_repo_path_rejects_git_suffix(self):
        self.assertIsNone(extract_repo_path('https://github.com/owner/repo.git'))

    def test_revision_validator_rejects_path_escape_values(self):
        self.assertFalse(is_safe_revision('../../escaped'))
        self.assertFalse(is_safe_revision('a/b'))
        self.assertFalse(is_safe_revision('..'))
        self.assertFalse(is_safe_revision('-abc123'))
        self.assertTrue(is_safe_revision('main'))

    def test_ref_validator_allows_branch_like_refs_without_path_escape(self):
        self.assertTrue(is_safe_ref('main'))
        self.assertTrue(is_safe_ref('feature/safe-branch'))
        self.assertTrue(is_safe_ref('refs/heads/dev'))
        self.assertFalse(is_safe_ref('../main'))
        self.assertFalse(is_safe_ref('feature//branch'))
        self.assertFalse(is_safe_ref('feature branch'))
        self.assertFalse(is_safe_ref('refs/heads/main.lock'))
        self.assertFalse(is_safe_ref('main@{1}'))

    def test_graph_id_validator_allows_symbol_ids_but_rejects_path_escape(self):
        self.assertTrue(is_safe_graph_id('pkg/app.py::Worker::run'))
        self.assertTrue(is_safe_graph_id('module::pkg.app'))
        self.assertFalse(is_safe_graph_id('../pkg/app.py::run'))
        self.assertFalse(is_safe_graph_id('pkg/app.py::bad id'))

    def test_share_id_validator_accepts_urlsafe_tokens_only(self):
        self.assertTrue(is_safe_share_id('abcdefghijklmnopqrstuvwxyz_123456'))
        self.assertFalse(is_safe_share_id('short'))
        self.assertFalse(is_safe_share_id('../bad-share-token'))

    def test_repo_file_path_validator_rejects_unsafe_paths(self):
        self.assertTrue(is_safe_repo_file_path('pkg/app.py'))
        self.assertTrue(is_safe_repo_file_path('한글/모듈.py'))
        self.assertFalse(is_safe_repo_file_path('../pkg/app.py'))
        self.assertFalse(is_safe_repo_file_path('/pkg/app.py'))
        self.assertFalse(is_safe_repo_file_path('pkg\\app.py'))


class ServicePathValidationTests(TestCase):
    def test_get_file_tree_rejects_repo_paths_with_extra_segments(self):
        self.assertIsNone(get_file_tree('owner/repo/extra'))

    def test_get_repo_analysis_rejects_repo_paths_with_extra_segments(self):
        self.assertIsNone(get_repo_analysis('owner/repo/extra'))

    def test_get_file_content_rejects_repo_paths_with_extra_segments(self):
        self.assertIsNone(get_file_content('owner/repo/extra', 'a.py'))

    def test_get_file_tree_rejects_repo_paths_with_spaces(self):
        self.assertIsNone(get_file_tree('owner/re po'))

    def test_get_repo_analysis_rejects_repo_paths_with_spaces(self):
        self.assertIsNone(get_repo_analysis('owner/re po'))

    def test_repo_snapshot_or_raise_maps_invalid_repo_path(self):
        with self.assertRaisesRegex(RepoIngestionError, '올바른 repo 경로'):
            get_repo_snapshot_or_raise('owner/repo/extra')


class RepoLockCleanupTests(TestCase):
    def test_repo_lock_cleans_up_after_exception(self):
        lock_path = _repo_lock_path('owner/repo')

        with self.assertRaisesRegex(RuntimeError, 'boom'):
            with _repo_lock('owner/repo'):
                self.assertTrue(lock_path.is_file())
                raise RuntimeError('boom')

        self.assertFalse(lock_path.exists())


@override_settings(
    TEMP_DIR=settings.BASE_DIR / 'temp' / 'repo-workspace-tests',
    PLAYGROUND_DIR=settings.BASE_DIR / 'temp' / 'repo-workspace-tests' / 'playground',
)
class LocalRepoWorkspaceTests(TestCase):
    source_repo: Path = Path()

    def setUp(self):
        self.source_repo = settings.TEMP_DIR / 'source-repo'
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)
        shutil.rmtree(settings.PLAYGROUND_DIR, ignore_errors=True)
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        settings.PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)
        create_git_fixture_repo(
            self.source_repo,
            {
                'pkg/app.py': 'def greet():\n    return "hi"\n',
                'README.md': '# demo\n',
            },
        )

    def tearDown(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)

    @patch('github_repo.services._repo_clone_url')
    def test_get_file_tree_clones_repo_into_playground(self, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)

        files = get_file_tree('owner/repo')

        self.assertEqual(files, ['README.md', 'pkg/app.py'])
        self.assertTrue((settings.PLAYGROUND_DIR / 'owner' / 'repo').is_dir())

    @patch('github_repo.services._repo_clone_url')
    def test_get_file_content_reads_from_local_checkout(self, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)
        get_file_tree('owner/repo')

        content = get_file_content('owner/repo', 'pkg/app.py')

        self.assertEqual(content, 'def greet():\n    return "hi"\n')

    @patch('github_repo.services._repo_clone_url')
    def test_get_file_content_rejects_path_traversal(self, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)
        get_file_tree('owner/repo')

        content = get_file_content('owner/repo', '../pkg/app.py')

        self.assertIsNone(content)

    @patch('github_repo.services._repo_clone_url')
    def test_get_repo_revision_reads_local_checkout_head(self, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)
        get_file_tree('owner/repo')

        revision = get_repo_revision('owner/repo')
        expected_revision = run_git(self.source_repo, 'rev-parse', 'HEAD')

        self.assertEqual(revision, expected_revision)

    @patch('github_repo.services._repo_clone_url')
    def test_distinct_repo_paths_do_not_collide_in_playground(self, repo_clone_url):
        first_repo = settings.TEMP_DIR / 'source-repo-one'
        second_repo = settings.TEMP_DIR / 'source-repo-two'
        for repo_dir, message, content in (
            (first_repo, 'first repo', 'def first():\n    return 1\n'),
            (second_repo, 'second repo', 'def second():\n    return 2\n'),
        ):
            create_git_fixture_repo(repo_dir, {'pkg/app.py': content}, commit_message=message)

        repo_clone_url.side_effect = lambda repo_path: {
            'owner/repo__x': str(first_repo),
            'owner__repo/x': str(second_repo),
        }[repo_path]

        get_file_tree('owner/repo__x')
        get_file_tree('owner__repo/x')

        self.assertTrue((settings.PLAYGROUND_DIR / 'owner' / 'repo__x').is_dir())
        self.assertTrue((settings.PLAYGROUND_DIR / 'owner__repo' / 'x').is_dir())
        self.assertNotEqual(
            get_file_content('owner/repo__x', 'pkg/app.py'),
            get_file_content('owner__repo/x', 'pkg/app.py'),
        )
        shutil.rmtree(first_repo, ignore_errors=True)
        shutil.rmtree(second_repo, ignore_errors=True)


@override_settings(
    TEMP_DIR=settings.BASE_DIR / 'temp' / 'safe-ingestion-tests',
    PLAYGROUND_DIR=settings.BASE_DIR / 'temp' / 'safe-ingestion-tests' / 'playground',
)
class SafeRepoIngestionTests(TestCase):
    source_repo: Path = Path()

    def setUp(self):
        self.source_repo = settings.TEMP_DIR / 'source-repo'
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        settings.PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)

    def test_git_timeout_is_mapped_to_repo_ingestion_error(self):
        with patch('github_repo.services.subprocess.run') as run_mock:
            run_mock.side_effect = subprocess.TimeoutExpired(['git', 'clone'], timeout=30)

            with self.assertRaises(RepoIngestionError) as context:
                get_repo_snapshot_or_raise('owner/repo')

        self.assertEqual(context.exception.code, 'timeout')
        self.assertEqual(context.exception.command_category, 'clone')
        self.assertEqual(context.exception.metadata['timeout_seconds'], settings.GITHUB_REPO_GIT_TIMEOUT_SECONDS)

    def test_git_error_sanitizes_tokenized_stderr_and_preserves_category(self):
        with patch('github_repo.services.subprocess.run') as run_mock:
            run_mock.side_effect = subprocess.CalledProcessError(
                128,
                ['git', 'clone'],
                stderr='fatal: repository https://secret-token@github.com/owner/repo.git not found',
            )

            with self.assertRaises(RepoIngestionError) as context:
                _run_git('clone', 'https://github.com/owner/repo.git', '/tmp/repo')

        self.assertEqual(context.exception.code, 'repo_not_found')
        self.assertEqual(context.exception.command_category, 'clone')
        self.assertNotIn('secret-token', context.exception.stderr)

    @override_settings(GITHUB_REPO_MAX_FILES=1)
    @patch('github_repo.services._repo_clone_url')
    def test_repo_file_count_limit_raises_too_large(self, repo_clone_url):
        create_git_fixture_repo(
            self.source_repo,
            {
                'README.md': '# demo\n',
                'pkg/app.py': 'def app():\n    return "ok"\n',
            },
        )
        repo_clone_url.return_value = str(self.source_repo)

        with self.assertRaises(RepoIngestionError) as context:
            get_repo_snapshot_or_raise('owner/repo')

        self.assertEqual(context.exception.code, 'too_large')
        self.assertEqual(context.exception.metadata['limit_type'], 'max_files')

    @override_settings(GITHUB_REPO_MAX_PYTHON_FILES=1)
    @patch('github_repo.services._repo_clone_url')
    def test_python_file_count_limit_raises_too_large(self, repo_clone_url):
        create_git_fixture_repo(
            self.source_repo,
            {
                'pkg/one.py': 'def one():\n    return 1\n',
                'pkg/two.py': 'def two():\n    return 2\n',
            },
        )
        repo_clone_url.return_value = str(self.source_repo)

        with self.assertRaises(RepoIngestionError) as context:
            get_repo_snapshot_or_raise('owner/repo')

        self.assertEqual(context.exception.code, 'too_large')
        self.assertEqual(context.exception.metadata['limit_type'], 'max_python_files')

    @override_settings(GITHUB_REPO_MAX_SINGLE_FILE_BYTES=12)
    @patch('github_repo.services._repo_clone_url')
    def test_large_single_file_limit_raises_too_large(self, repo_clone_url):
        create_git_fixture_repo(
            self.source_repo,
            {'pkg/big.py': 'def big():\n    return "this is too large"\n'},
        )
        repo_clone_url.return_value = str(self.source_repo)
        get_repo_snapshot_or_raise('owner/repo')

        with self.assertRaises(RepoIngestionError) as context:
            get_file_content_or_raise('owner/repo', 'pkg/big.py')

        self.assertEqual(context.exception.code, 'too_large')
        self.assertEqual(context.exception.metadata['limit_type'], 'max_single_file_bytes')

    @override_settings(GITHUB_REPO_MAX_TOTAL_ANALYZED_BYTES=45)
    @patch('github_repo.services._repo_clone_url')
    def test_total_analyzed_python_bytes_limit_raises_too_large(self, repo_clone_url):
        create_git_fixture_repo(
            self.source_repo,
            {
                'pkg/one.py': 'def one():\n    return "1234567890"\n',
                'pkg/two.py': 'def two():\n    return "abcdefghij"\n',
            },
        )
        repo_clone_url.return_value = str(self.source_repo)

        with self.assertRaises(RepoIngestionError) as context:
            get_repo_analysis('owner/repo')

        self.assertEqual(context.exception.code, 'too_large')
        self.assertEqual(context.exception.metadata['limit_type'], 'max_total_analyzed_bytes')

    @patch('github_repo.services._repo_clone_url')
    def test_origin_mismatch_is_cleaned_before_clone(self, repo_clone_url):
        create_git_fixture_repo(self.source_repo, {'pkg/app.py': 'def app():\n    return "source"\n'})
        wrong_checkout = settings.PLAYGROUND_DIR / 'owner' / 'repo'
        create_git_fixture_repo(wrong_checkout, {'pkg/app.py': 'def app():\n    return "wrong"\n'})
        repo_clone_url.return_value = str(self.source_repo)

        files = get_file_tree('owner/repo')

        self.assertEqual(files, ['pkg/app.py'])
        self.assertEqual(get_file_content('owner/repo', 'pkg/app.py'), 'def app():\n    return "source"\n')


class ParserGraphTests(TestCase):
    def test_parse_repo_adds_contains_imports_and_call_edges(self):
        files = ['pkg/utils.py', 'pkg/models.py']
        file_contents = {
            'pkg/models.py': (
                'from pkg.utils import helper\n\n'
                'class Base:\n'
                '    pass\n\n'
                'class Child(Base):\n'
                '    def run(self):\n'
                '        self.step()\n\n'
                '    def step(self):\n'
                '        helper()\n\n'
            ),
            'pkg/utils.py': (
                'def helper():\n'
                '    return "ok"\n'
            ),
        }

        graph = parse_repo('owner/repo', list(reversed(files)), lambda _repo, path: file_contents[path])
        nodes_by_id = {node['id']: node for node in graph['nodes']}
        edges = {(edge['source'], edge['target'], edge['type']) for edge in graph['edges']}

        self.assertIn('module::pkg.utils', nodes_by_id)
        self.assertIn(('pkg/models.py', 'module::pkg.models', 'contains'), edges)
        self.assertIn(('module::pkg.models', 'pkg/models.py::Child', 'contains'), edges)
        self.assertIn(('pkg/models.py::Child', 'pkg/models.py::Child::run', 'contains'), edges)
        self.assertIn(('pkg/models.py', 'module::pkg.utils', 'imports'), edges)
        self.assertIn(('pkg/models.py::Child', 'pkg/models.py::Base', 'inherits'), edges)
        self.assertIn(('pkg/models.py::Child::run', 'pkg/models.py::Child::step', 'calls'), edges)
        self.assertIn(('pkg/models.py::Child::step', 'pkg/utils.py::helper', 'calls'), edges)

    def test_parse_repo_prefers_same_file_class_for_inheritance_resolution(self):
        files = ['a.py', 'b.py']
        file_contents = {
            'a.py': (
                'class Base:\n'
                '    pass\n\n'
                'class Child(Base):\n'
                '    pass\n'
            ),
            'b.py': (
                'class Base:\n'
                '    pass\n'
            ),
        }

        graph = parse_repo('owner/repo', files, lambda _repo, path: file_contents[path])
        edges = {(edge['source'], edge['target'], edge['type']) for edge in graph['edges']}

        self.assertIn(('a.py::Child', 'a.py::Base', 'inherits'), edges)

    def test_parse_repo_deduplicates_shared_module_nodes(self):
        files = ['a.py', 'b.py']
        file_contents = {
            'a.py': 'import os\n',
            'b.py': 'import os\n',
        }

        graph = parse_repo('owner/repo', files, lambda _repo, path: file_contents[path])
        module_nodes = [node for node in graph['nodes'] if node['id'] == 'module::os']

        self.assertEqual(len(module_nodes), 1)

    def test_parse_repo_includes_decorated_definitions(self):
        files = ['app.py']
        file_contents = {
            'app.py': (
                '@app.get("/")\n'
                'def route():\n'
                '    return "ok"\n\n'
                'class User:\n'
                '    @property\n'
                '    def name(self):\n'
                '        return "u"\n'
            ),
        }

        graph = parse_repo('owner/repo', files, lambda _repo, path: file_contents[path])
        node_ids = {node['id'] for node in graph['nodes']}

        self.assertIn('app.py::route', node_ids)
        self.assertIn('app.py::User::name', node_ids)

    def test_parse_repo_does_not_resolve_non_self_attribute_calls_to_local_functions(self):
        files = ['app.py']
        file_contents = {
            'app.py': (
                'def run():\n'
                '    return "top"\n\n'
                'def wrapper(other):\n'
                '    return other.run()\n'
            ),
        }

        graph = parse_repo('owner/repo', files, lambda _repo, path: file_contents[path])
        edges = {(edge['source'], edge['target'], edge['type']) for edge in graph['edges']}

        self.assertIn(('app.py::wrapper', 'attribute::run', 'calls'), edges)
        self.assertNotIn(('app.py::wrapper', 'app.py::run', 'calls'), edges)

    def test_parse_repo_does_not_pull_nested_function_calls_into_outer_function(self):
        files = ['app.py']
        file_contents = {
            'app.py': (
                'def helper():\n'
                '    return "ok"\n\n'
                'def outer():\n'
                '    def inner():\n'
                '        helper()\n'
                '    return inner\n'
            ),
        }

        graph = parse_repo('owner/repo', files, lambda _repo, path: file_contents[path])
        edges = {(edge['source'], edge['target'], edge['type']) for edge in graph['edges']}

        self.assertNotIn(('app.py::outer', 'app.py::helper', 'calls'), edges)


class ParserDirectorySymbolTests(TestCase):
    def test_parse_repo_builds_directory_tree_and_unsupported_file_nodes(self):
        files = ['README.md', 'pkg/app.py', 'pkg/nested/util.py']
        file_contents = {
            'pkg/app.py': 'def main():\n    return "ok"\n',
            'pkg/nested/util.py': 'def helper():\n    return "ok"\n',
        }

        graph = parse_repo('owner/repo', files, lambda _repo, path: file_contents.get(path))
        nodes_by_id = {node['id']: node for node in graph['nodes']}
        edges = {(edge['source'], edge['target'], edge['type']) for edge in graph['edges']}
        tree_ids = json.dumps(graph['tree'], ensure_ascii=False)

        self.assertIn('pkg', nodes_by_id)
        self.assertIn('pkg/nested', nodes_by_id)
        self.assertIn('README.md', nodes_by_id)
        self.assertIn('module::pkg.app', nodes_by_id)
        self.assertTrue(cast(dict[str, object], nodes_by_id['README.md']['metadata'])['unsupported'])
        self.assertIn(('pkg', 'pkg/app.py', 'contains'), edges)
        self.assertIn(('pkg/nested', 'pkg/nested/util.py', 'contains'), edges)
        self.assertIn('module::pkg.app', tree_ids)
        self.assertIn('README.md', tree_ids)

    def test_python_symbols_include_line_decorator_and_docstring_metadata(self):
        files = ['app.py']
        file_contents = {
            'app.py': (
                '@app.get("/users")\n'
                'def route():\n'
                '    """Serve users."""\n'
                '    return "ok"\n\n'
                'class User:\n'
                '    """User model."""\n'
                '    @property\n'
                '    def name(self):\n'
                '        """Display name."""\n'
                '        return "Ada"\n'
            ),
        }

        graph = parse_repo('owner/repo', files, lambda _repo, path: file_contents[path])
        nodes_by_id = {node['id']: node for node in graph['nodes']}

        route = nodes_by_id['app.py::route']
        user = nodes_by_id['app.py::User']
        method = nodes_by_id['app.py::User::name']

        self.assertEqual(route['type'], 'function')
        self.assertEqual(route['start_line'], 2)
        self.assertEqual(cast(dict[str, object], route['metadata'])['decorators'], ['app.get("/users")'])
        self.assertEqual(cast(dict[str, object], route['metadata'])['docstring'], 'Serve users.')
        self.assertEqual(user['type'], 'class')
        self.assertEqual(cast(dict[str, object], user['metadata'])['docstring'], 'User model.')
        self.assertEqual(method['type'], 'method')
        self.assertEqual(method['parent'], 'app.py::User')
        self.assertEqual(cast(dict[str, object], method['metadata'])['decorators'], ['property'])
        self.assertEqual(cast(dict[str, object], method['metadata'])['docstring'], 'Display name.')

    def test_syntax_error_file_adds_warning_and_does_not_stop_repo_parse(self):
        files = ['bad.py', 'good.py']
        file_contents = {
            'bad.py': 'def broken(:\n    pass\n',
            'good.py': 'def ok():\n    return True\n',
        }

        graph = parse_repo('owner/repo', files, lambda _repo, path: file_contents[path])
        node_ids = {node['id'] for node in graph['nodes']}
        nodes_by_id = {node['id']: node for node in graph['nodes']}
        warnings = graph['warnings']

        self.assertIn('bad.py', node_ids)
        self.assertIn('module::bad', node_ids)
        self.assertIn('good.py::ok', node_ids)
        self.assertFalse(cast(dict[str, object], nodes_by_id['bad.py']['metadata'])['analyzed'])
        self.assertEqual(warnings[0]['code'], 'syntax_error')
        self.assertEqual(warnings[0]['path'], 'bad.py')

    def test_parse_repo_ordering_is_deterministic(self):
        files = ['b.py', 'a.py']
        file_contents = {
            'a.py': 'def alpha():\n    return 1\n',
            'b.py': 'def beta():\n    return alpha()\n',
        }

        first = parse_repo('owner/repo', files, lambda _repo, path: file_contents[path])
        second = parse_repo('owner/repo', list(reversed(files)), lambda _repo, path: file_contents[path])

        self.assertEqual([node['id'] for node in first['nodes']], [node['id'] for node in second['nodes']])
        self.assertEqual(
            [(edge['id'], edge['source'], edge['target'], edge['type']) for edge in first['edges']],
            [(edge['id'], edge['source'], edge['target'], edge['type']) for edge in second['edges']],
        )
        self.assertEqual(first['tree'], second['tree'])


class ParserRelationshipEntrypointTests(TestCase):
    def test_relative_import_alias_call_resolves_to_local_symbol(self):
        files = ['pkg/app.py', 'pkg/utils.py']
        file_contents = {
            'pkg/app.py': (
                'from .utils import helper as run_helper\n\n'
                'def main():\n'
                '    return run_helper()\n'
            ),
            'pkg/utils.py': (
                'def helper():\n'
                '    return "ok"\n'
            ),
        }

        graph = parse_repo('owner/repo', files, lambda _repo, path: file_contents[path])
        edges = {(edge['source'], edge['target'], edge['type']) for edge in graph['edges']}

        self.assertIn(('pkg/app.py', 'module::pkg.utils', 'imports'), edges)
        self.assertIn(('pkg/app.py::main', 'pkg/utils.py::helper', 'calls'), edges)

    def test_module_alias_attribute_call_resolves_to_local_symbol(self):
        files = ['pkg/app.py', 'pkg/utils.py']
        file_contents = {
            'pkg/app.py': (
                'import pkg.utils as utils\n\n'
                'def run():\n'
                '    return utils.helper()\n'
            ),
            'pkg/utils.py': (
                'def helper():\n'
                '    return "ok"\n'
            ),
        }

        graph = parse_repo('owner/repo', files, lambda _repo, path: file_contents[path])
        edges = {(edge['source'], edge['target'], edge['type']) for edge in graph['edges']}

        self.assertIn(('pkg/app.py', 'module::pkg.utils', 'imports'), edges)
        self.assertIn(('pkg/app.py::run', 'pkg/utils.py::helper', 'calls'), edges)

    def test_imported_base_class_inheritance_resolves_to_local_class(self):
        files = ['pkg/base.py', 'pkg/child.py']
        file_contents = {
            'pkg/base.py': (
                'class BaseTask:\n'
                '    pass\n'
            ),
            'pkg/child.py': (
                'from .base import BaseTask\n\n'
                'class BuildTask(BaseTask):\n'
                '    pass\n'
            ),
        }

        graph = parse_repo('owner/repo', files, lambda _repo, path: file_contents[path])
        edges = {(edge['source'], edge['target'], edge['type']) for edge in graph['edges']}

        self.assertIn(('pkg/child.py', 'module::pkg.base', 'imports'), edges)
        self.assertIn(('pkg/child.py::BuildTask', 'pkg/base.py::BaseTask', 'inherits'), edges)

    def test_unresolved_attribute_call_gets_external_node_warning_and_low_confidence(self):
        files = ['worker.py']
        file_contents = {
            'worker.py': (
                'def run(client):\n'
                '    return client.execute()\n'
            ),
        }

        graph = parse_repo('owner/repo', files, lambda _repo, path: file_contents[path])
        nodes_by_id = {node['id']: node for node in graph['nodes']}
        call_edges = [edge for edge in graph['edges'] if edge['type'] == 'calls']

        self.assertEqual(nodes_by_id['attribute::execute']['type'], 'external')
        self.assertEqual(call_edges[0]['target'], 'attribute::execute')
        self.assertEqual(call_edges[0]['confidence'], 0.4)
        self.assertEqual(graph['warnings'][0]['code'], 'unresolved_call')

    def test_entrypoints_and_key_modules_are_reported(self):
        files = ['manage.py', 'service/api.py', 'service/core.py']
        file_contents = {
            'manage.py': (
                'from service.api import route\n\n'
                'def main():\n'
                '    route()\n\n'
                'if __name__ == "__main__":\n'
                '    main()\n'
            ),
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
        }

        graph = parse_repo('owner/repo', files, lambda _repo, path: file_contents[path])
        entrypoint_kinds = {entrypoint['kind'] for entrypoint in graph['entrypoints']}
        key_module_ids = {module['id'] for module in graph['key_modules']}
        entrypoint_edges = [edge for edge in graph['edges'] if edge['type'] == 'entrypoint']

        self.assertIn('python_main_guard', entrypoint_kinds)
        self.assertIn('main_function', entrypoint_kinds)
        self.assertIn('django_manage', entrypoint_kinds)
        self.assertIn('web_route', entrypoint_kinds)
        self.assertIn('module::service.api', key_module_ids)
        self.assertTrue(entrypoint_edges)


@override_settings(
    TEMP_DIR=settings.BASE_DIR / 'temp' / 'foundation-fixture-tests',
)
class FoundationEvalFixtureTests(TestCase):
    def setUp(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)

    def test_git_fixture_repo_helper_is_compatible_with_server_git_version(self):
        fixture_repo = create_named_fixture_repo('plain_python_package', settings.TEMP_DIR / 'plain-package')

        self.assertEqual(run_git(fixture_repo.path, 'branch', '--show-current'), 'main')
        self.assertEqual(run_git(fixture_repo.path, 'rev-parse', 'HEAD'), fixture_repo.revision)
        self.assertTrue((fixture_repo.path / 'sample_pkg' / 'main.py').is_file())

    def test_golden_fixture_catalog_covers_planned_foundation_cases(self):
        self.assertGreaterEqual(len(GOLDEN_FIXTURE_REPOS), 3)
        self.assertIn('plain_python_package', GOLDEN_FIXTURE_REPOS)
        self.assertIn('oop_inheritance_sample', GOLDEN_FIXTURE_REPOS)
        self.assertIn('cross_file_import_call_sample', GOLDEN_FIXTURE_REPOS)
        self.assertIn('django_like_mini_app', GOLDEN_FIXTURE_REPOS)
        self.assertIn('fastapi_like_mini_app', GOLDEN_FIXTURE_REPOS)
        self.assertIn('ambiguous_symbol_sample', GOLDEN_FIXTURE_REPOS)
        self.assertIn('korean_readme_sample', GOLDEN_FIXTURE_REPOS)

    def test_golden_fixture_expected_graph_fragments_parse(self):
        for fixture in GOLDEN_FIXTURE_REPOS.values():
            with self.subTest(fixture=fixture.name):
                graph = parse_repo('owner/repo', list(fixture.files), lambda _repo, path: fixture.files[path])
                node_ids = {node['id'] for node in graph['nodes']}
                edges = {(edge['source'], edge['target'], edge['type']) for edge in graph['edges']}

                for expected_node in fixture.expected_nodes:
                    self.assertIn(expected_node, node_ids)
                for expected_edge in fixture.expected_edges:
                    self.assertIn(expected_edge, edges)

    def test_eval_rubric_has_graph_entrypoint_and_qa_dimensions(self):
        self.assertEqual(
            set(EVAL_RUBRIC),
            {'graph_node_recall', 'edge_correctness', 'entrypoint_correctness', 'qa_citation_correctness'},
        )
        for fixture in GOLDEN_FIXTURE_REPOS.values():
            self.assertTrue(set(fixture.rubric_tags).issubset(EVAL_RUBRIC))
            self.assertTrue(fixture.rubric_tags)


class GraphArtifactContractTests(TestCase):
    def _legacy_graph(self):
        return {
            'tree': [{'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'children': []}],
            'nodes': [
                {'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'file': 'pkg/app.py'},
                {'id': 'pkg/app.py::Worker', 'type': 'class', 'label': 'Worker', 'file': 'pkg/app.py', 'parent': 'pkg/app.py'},
                {'id': 'pkg/app.py::Worker::run', 'type': 'function', 'label': 'run', 'file': 'pkg/app.py', 'parent': 'pkg/app.py::Worker'},
            ],
            'edges': [
                {'id': 'e1', 'source': 'pkg/app.py', 'target': 'pkg/app.py::Worker', 'type': 'contains', 'file': 'pkg/app.py'},
                {'id': 'e2', 'source': 'pkg/app.py::Worker', 'target': 'pkg/app.py::Worker::run', 'type': 'contains', 'file': 'pkg/app.py'},
            ],
        }

    def test_build_graph_artifact_adds_v1_contract_and_compatibility_aliases(self):
        artifact = build_graph_artifact(
            repo_path='owner/repo',
            revision='abc123',
            graph=self._legacy_graph(),
            file_contents={'pkg/app.py': 'class Worker:\n    def run(self):\n        pass\n'},
            generated_at='2026-01-01T00:00:00Z',
        )

        validate_graph_artifact(artifact)
        self.assertEqual(artifact['schema_version'], GRAPH_ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(artifact['provider'], 'github')
        self.assertEqual(artifact['owner'], 'owner')
        self.assertEqual(artifact['name'], 'repo')
        self.assertEqual(artifact['ref'], 'HEAD')
        self.assertEqual(artifact['status'], 'succeeded')
        self.assertEqual(artifact['entrypoints'], [])
        self.assertEqual(artifact['key_modules'], [])
        self.assertEqual(artifact['summaries'], {})
        self.assertEqual(artifact['warnings'], [])
        self.assertIn('max_python_files', artifact['limits'])

        nodes_by_id = {node['id']: node for node in artifact['nodes']}
        self.assertEqual(nodes_by_id['pkg/app.py::Worker::run']['kind'], 'method')
        self.assertEqual(nodes_by_id['pkg/app.py::Worker::run']['type'], 'function')
        self.assertEqual(nodes_by_id['pkg/app.py::Worker::run']['path'], 'pkg/app.py')
        self.assertEqual(nodes_by_id['pkg/app.py::Worker::run']['file'], 'pkg/app.py')
        self.assertEqual(nodes_by_id['pkg/app.py::Worker::run']['parent_id'], 'pkg/app.py::Worker')

        self.assertEqual(artifact['edges'][0]['kind'], 'contains')
        self.assertEqual(artifact['edges'][0]['type'], 'contains')
        self.assertEqual(artifact['edges'][0]['path'], 'pkg/app.py')
        self.assertEqual(artifact['edges'][0]['confidence'], 1.0)

    def test_graph_artifact_schema_field_sets_are_snapshotted(self):
        artifact = build_graph_artifact(
            repo_path='owner/repo',
            revision='abc123',
            graph=self._legacy_graph(),
            generated_at='2026-01-01T00:00:00Z',
        )
        method_node = next(node for node in artifact['nodes'] if node['id'] == 'pkg/app.py::Worker::run')

        self.assertEqual(
            set(artifact),
            {
                'schema_version',
                'repo',
                'provider',
                'owner',
                'name',
                'ref',
                'revision',
                'default_branch',
                'generated_at',
                'status',
                'limits',
                'file_contents',
                'tree',
                'nodes',
                'edges',
                'entrypoints',
                'key_modules',
                'summaries',
                'warnings',
            },
        )
        self.assertEqual(
            set(method_node),
            {
                'id',
                'kind',
                'label',
                'path',
                'parent_id',
                'symbol',
                'language',
                'start_line',
                'end_line',
                'metadata',
                'type',
                'file',
                'parent',
            },
        )
        self.assertEqual(
            set(artifact['edges'][0]),
            {'id', 'kind', 'source', 'target', 'path', 'confidence', 'metadata', 'type', 'file'},
        )

    def test_validate_graph_artifact_rejects_unknown_kinds(self):
        artifact = build_graph_artifact(
            repo_path='owner/repo',
            revision='abc123',
            graph=self._legacy_graph(),
            generated_at='2026-01-01T00:00:00Z',
        )

        bad_node_artifact = json.loads(json.dumps(artifact))
        bad_node_artifact['nodes'][0]['kind'] = 'unknown'
        with self.assertRaisesRegex(ArtifactValidationError, 'unknown node kind'):
            validate_graph_artifact(bad_node_artifact)

        bad_edge_artifact = json.loads(json.dumps(artifact))
        bad_edge_artifact['edges'][0]['kind'] = 'unknown'
        with self.assertRaisesRegex(ArtifactValidationError, 'unknown edge kind'):
            validate_graph_artifact(bad_edge_artifact)

    def test_validate_graph_artifact_requires_node_and_edge_fields(self):
        artifact = build_graph_artifact(
            repo_path='owner/repo',
            revision='abc123',
            graph=self._legacy_graph(),
            generated_at='2026-01-01T00:00:00Z',
        )

        missing_node_field = json.loads(json.dumps(artifact))
        del missing_node_field['nodes'][0]['metadata']
        with self.assertRaisesRegex(ArtifactValidationError, 'node missing required fields'):
            validate_graph_artifact(missing_node_field)

        missing_edge_field = json.loads(json.dumps(artifact))
        del missing_edge_field['edges'][0]['confidence']
        with self.assertRaisesRegex(ArtifactValidationError, 'edge missing required fields'):
            validate_graph_artifact(missing_edge_field)

    def test_artifact_ids_are_deterministic_for_same_fixture(self):
        fixture = GOLDEN_FIXTURE_REPOS['cross_file_import_call_sample']
        graph_one = parse_repo('owner/repo', list(fixture.files), lambda _repo, path: fixture.files[path])
        graph_two = parse_repo('owner/repo', list(reversed(fixture.files)), lambda _repo, path: fixture.files[path])

        artifact_one = build_graph_artifact(
            repo_path='owner/repo',
            revision='abc123',
            graph=graph_one,
            generated_at='2026-01-01T00:00:00Z',
        )
        artifact_two = build_graph_artifact(
            repo_path='owner/repo',
            revision='abc123',
            graph=graph_two,
            generated_at='2026-01-01T00:00:00Z',
        )

        self.assertEqual(
            [(node['id'], node['kind']) for node in artifact_one['nodes']],
            [(node['id'], node['kind']) for node in artifact_two['nodes']],
        )
        self.assertEqual(
            [(edge['id'], edge['kind'], edge['source'], edge['target']) for edge in artifact_one['edges']],
            [(edge['id'], edge['kind'], edge['source'], edge['target']) for edge in artifact_two['edges']],
        )

    def test_coerce_graph_artifact_upgrades_legacy_cache_payload(self):
        legacy_payload = {
            'repo': 'owner/repo',
            'revision': 'abc123',
            'file_contents': {'pkg/app.py': 'def main():\n    pass\n'},
            **self._legacy_graph(),
        }

        artifact = coerce_graph_artifact(legacy_payload)

        self.assertEqual(artifact['schema_version'], GRAPH_ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(artifact['repo'], 'owner/repo')
        self.assertEqual(artifact['revision'], 'abc123')
        self.assertEqual(artifact['nodes'][0]['kind'], 'file')


class GraphDiffContractTests(TestCase):
    def _artifact(self, revision: str, *, nodes=None, edges=None, entrypoints=None):
        return build_graph_artifact(
            repo_path='owner/repo',
            revision=revision,
            graph={
                'tree': [],
                'nodes': nodes if nodes is not None else [],
                'edges': edges if edges is not None else [],
            },
            entrypoints=entrypoints,
            generated_at='2026-01-01T00:00:00Z',
        )

    def test_same_revision_diff_is_empty(self):
        artifact = self._artifact(
            'abc123',
            nodes=[{'id': 'pkg/app.py::main', 'type': 'function', 'label': 'main', 'file': 'pkg/app.py'}],
            edges=[{'id': 'e1', 'source': 'pkg/app.py', 'target': 'pkg/app.py::main', 'type': 'contains', 'file': 'pkg/app.py'}],
        )

        diff = compare_graph_artifacts(artifact, artifact)

        self.assertEqual(
            diff['summary'],
            {
                'added_nodes': 0,
                'removed_nodes': 0,
                'changed_nodes': 0,
                'added_edges': 0,
                'removed_edges': 0,
                'changed_edges': 0,
                'changed_metadata': 0,
            },
        )

    def test_added_function_node_is_reported(self):
        base = self._artifact('abc123')
        head = self._artifact(
            'def456',
            nodes=[{'id': 'pkg/app.py::main', 'type': 'function', 'label': 'main', 'file': 'pkg/app.py'}],
        )

        diff = compare_graph_artifacts(base, head)

        self.assertEqual(diff['summary']['added_nodes'], 1)
        self.assertEqual(diff['nodes']['added'][0]['id'], 'pkg/app.py::main')

    def test_removed_class_node_is_reported(self):
        base = self._artifact(
            'abc123',
            nodes=[{'id': 'pkg/app.py::Worker', 'type': 'class', 'label': 'Worker', 'file': 'pkg/app.py'}],
        )
        head = self._artifact('def456')

        diff = compare_graph_artifacts(base, head)

        self.assertEqual(diff['summary']['removed_nodes'], 1)
        self.assertEqual(diff['nodes']['removed'][0]['id'], 'pkg/app.py::Worker')

    def test_added_call_and_removed_import_edges_are_reported(self):
        base = self._artifact(
            'abc123',
            edges=[
                {'id': 'e1', 'source': 'pkg/app.py', 'target': 'module::pkg.lib', 'type': 'imports', 'file': 'pkg/app.py'},
            ],
        )
        head = self._artifact(
            'def456',
            edges=[
                {'id': 'e2', 'source': 'pkg/app.py::main', 'target': 'pkg/lib.py::helper', 'type': 'calls', 'file': 'pkg/app.py'},
            ],
        )

        diff = compare_graph_artifacts(base, head)

        self.assertEqual(diff['summary']['added_edges'], 1)
        self.assertEqual(diff['summary']['removed_edges'], 1)
        self.assertEqual(diff['edges']['added'][0]['kind'], 'calls')
        self.assertEqual(diff['edges']['removed'][0]['kind'], 'imports')

    def test_changed_node_and_metadata_are_reported(self):
        base = self._artifact(
            'abc123',
            nodes=[{'id': 'pkg/app.py::main', 'type': 'function', 'label': 'main', 'file': 'pkg/app.py'}],
            entrypoints=[],
        )
        head = self._artifact(
            'def456',
            nodes=[{'id': 'pkg/app.py::main', 'type': 'function', 'label': 'run', 'file': 'pkg/app.py'}],
            entrypoints=[{'id': 'pkg/app.py::main', 'kind': 'main_function'}],
        )

        diff = compare_graph_artifacts(base, head)

        self.assertEqual(diff['summary']['changed_nodes'], 1)
        self.assertIn('label', diff['nodes']['changed'][0]['changed_fields'])
        self.assertEqual(diff['summary']['changed_metadata'], 1)
        self.assertEqual(diff['metadata']['changed'][0]['field'], 'entrypoints')


@override_settings(
    TEMP_DIR=settings.TEMP_DIR / 'analysis-service-tests',
)
class AnalysisArtifactServiceTests(TestCase):
    def setUp(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)

    @patch('api.services.parse_repo')
    @patch('api.services.get_repo_snapshot')
    @patch('api.services.get_file_content_or_raise', return_value='def main():\n    pass\n')
    def test_get_repo_analysis_caches_graph_payload(self, get_file_content_mock, get_repo_snapshot_mock, parse_repo_mock):
        get_repo_snapshot_mock.return_value = ('abc123', ['pkg/app.py'])
        parse_repo_mock.return_value = {
            'tree': [{'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'children': []}],
            'nodes': [{'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'file': 'pkg/app.py'}],
            'edges': [],
        }

        first = get_repo_analysis('owner/repo')
        second = get_repo_analysis('owner/repo')

        artifact_path = settings.TEMP_DIR / 'analysis' / 'owner' / 'repo@abc123' / 'graph.json'
        self.assertTrue(artifact_path.is_file())
        self.assertEqual(first, second)
        parse_repo_mock.assert_called_once()
        self.assertEqual(Repository.objects.count(), 1)
        self.assertEqual(AnalysisRun.objects.filter(status=AnalysisRun.STATUS_SUCCEEDED).count(), 1)
        self.assertEqual(AnalysisArtifact.objects.count(), 1)
        artifact = AnalysisArtifact.objects.select_related('analysis_run').get()
        self.assertEqual(artifact.analysis_run.repository.full_name, 'owner/repo')
        self.assertEqual(artifact.analysis_run.revision, 'abc123')
        self.assertEqual(artifact.node_count, 1)
        self.assertEqual(artifact.edge_count, 0)

    @patch('api.services.parse_repo')
    @patch('api.services.get_repo_snapshot')
    @patch('api.services.get_file_content_or_raise', return_value='def main():\n    pass\n')
    def test_get_repo_analysis_returns_v1_artifact_without_breaking_graph_aliases(self, get_file_content_mock, get_repo_snapshot_mock, parse_repo_mock):
        get_repo_snapshot_mock.return_value = ('abc123', ['pkg/app.py'])
        parse_repo_mock.return_value = {
            'tree': [{'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'children': []}],
            'nodes': [{'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'file': 'pkg/app.py'}],
            'edges': [{'id': 'e1', 'source': 'pkg/app.py', 'target': 'pkg/app.py::main', 'type': 'contains', 'file': 'pkg/app.py'}],
        }

        analysis = get_repo_analysis('owner/repo')

        self.assertIsNotNone(analysis)
        analysis_payload = cast(dict[str, object], analysis)
        self.assertEqual(analysis_payload['schema_version'], GRAPH_ARTIFACT_SCHEMA_VERSION)
        node = cast(list[dict[str, object]], analysis_payload['nodes'])[0]
        edge = cast(list[dict[str, object]], analysis_payload['edges'])[0]
        self.assertEqual(node['kind'], 'file')
        self.assertEqual(node['type'], 'file')
        self.assertEqual(node['path'], 'pkg/app.py')
        self.assertEqual(node['file'], 'pkg/app.py')
        self.assertEqual(edge['kind'], 'contains')
        self.assertEqual(edge['type'], 'contains')
        self.assertEqual(edge['path'], 'pkg/app.py')
        self.assertEqual(edge['file'], 'pkg/app.py')

    @patch('api.services.parse_repo')
    @patch('api.services.get_repo_snapshot')
    @patch('api.services.get_file_content_or_raise', return_value='def main():\n    pass\n')
    def test_get_repo_analysis_preserves_entrypoints_and_key_modules(self, get_file_content_mock, get_repo_snapshot_mock, parse_repo_mock):
        get_repo_snapshot_mock.return_value = ('abc123', ['pkg/app.py'])
        parse_repo_mock.return_value = {
            'tree': [],
            'nodes': [{'id': 'module::pkg.app', 'type': 'module', 'label': 'pkg.app', 'file': 'pkg/app.py'}],
            'edges': [],
            'entrypoints': [{'id': 'pkg/app.py::main', 'kind': 'main_function', 'path': 'pkg/app.py'}],
            'key_modules': [{'id': 'module::pkg.app', 'path': 'pkg/app.py', 'score': 5}],
            'warnings': [{'code': 'demo', 'path': 'pkg/app.py'}],
        }

        analysis = get_repo_analysis('owner/repo')

        self.assertIsNotNone(analysis)
        analysis_payload = cast(dict[str, object], analysis)
        self.assertEqual(analysis_payload['entrypoints'], [{'id': 'pkg/app.py::main', 'kind': 'main_function', 'path': 'pkg/app.py'}])
        self.assertEqual(analysis_payload['key_modules'], [{'id': 'module::pkg.app', 'path': 'pkg/app.py', 'score': 5}])
        self.assertEqual(analysis_payload['warnings'], [{'code': 'demo', 'path': 'pkg/app.py'}])

    @patch('api.services.parse_repo')
    @patch('api.services.get_repo_snapshot')
    @patch('api.services.get_file_content_or_raise', return_value='def main():\n    pass\n')
    def test_get_repo_analysis_refreshes_cache_for_new_revision(self, get_file_content_mock, get_repo_snapshot_mock, parse_repo_mock):
        get_repo_snapshot_mock.side_effect = [('abc123', ['pkg/app.py']), ('def456', ['pkg/app.py'])]
        parse_repo_mock.side_effect = [
            {
                'tree': [{'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'children': []}],
                'nodes': [{'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'file': 'pkg/app.py'}],
                'edges': [],
            },
            {
                'tree': [{'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'children': []}],
                'nodes': [{'id': 'pkg/app.py::main', 'type': 'function', 'label': 'main', 'file': 'pkg/app.py'}],
                'edges': [],
            },
        ]

        first = get_repo_analysis('owner/repo')
        second = get_repo_analysis('owner/repo')

        first_artifact = settings.TEMP_DIR / 'analysis' / 'owner' / 'repo@abc123' / 'graph.json'
        second_artifact = settings.TEMP_DIR / 'analysis' / 'owner' / 'repo@def456' / 'graph.json'
        self.assertTrue(first_artifact.is_file())
        self.assertTrue(second_artifact.is_file())
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        first_analysis = cast(dict[str, object], first)
        second_analysis = cast(dict[str, object], second)
        self.assertNotEqual(first_analysis['revision'], second_analysis['revision'])
        self.assertEqual(parse_repo_mock.call_count, 2)
        self.assertEqual(AnalysisRun.objects.filter(status=AnalysisRun.STATUS_SUCCEEDED).count(), 2)
        self.assertEqual(AnalysisArtifact.objects.count(), 2)

    @patch('api.services.parse_repo')
    @patch('api.services.get_repo_snapshot')
    @patch('api.services.get_file_content_or_raise', return_value='def main():\n    pass\n')
    def test_get_repo_analysis_uses_single_snapshot_per_call(self, get_file_content_mock, get_repo_snapshot_mock, parse_repo_mock):
        get_repo_snapshot_mock.return_value = ('abc123', ['pkg/app.py'])
        parse_repo_mock.return_value = {
            'tree': [],
            'nodes': [],
            'edges': [],
        }

        analysis = get_repo_analysis('owner/repo')

        self.assertIsNotNone(analysis)
        analysis_payload = cast(dict[str, object], analysis)
        self.assertEqual(analysis_payload['revision'], 'abc123')
        get_repo_snapshot_mock.assert_called_once_with('owner/repo')

    @patch('api.services.get_repo_snapshot')
    def test_get_repo_analysis_can_load_cached_revision_without_snapshot(self, get_repo_snapshot_mock):
        artifact_path = settings.TEMP_DIR / 'analysis' / 'owner' / 'repo@abc123' / 'graph.json'
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps({'repo': 'owner/repo', 'revision': 'abc123', 'file_contents': {}, 'tree': [], 'nodes': [], 'edges': []}), encoding='utf-8')

        analysis = get_repo_analysis('owner/repo', 'abc123')

        self.assertIsNotNone(analysis)
        get_repo_snapshot_mock.assert_not_called()
        cached_payload = json.loads(artifact_path.read_text(encoding='utf-8'))
        self.assertEqual(cached_payload['schema_version'], GRAPH_ARTIFACT_SCHEMA_VERSION)
        self.assertIsNotNone(get_artifact_by_revision('owner/repo', 'abc123'))

    @patch('api.services.parse_repo')
    @patch('api.services.get_repo_snapshot')
    @patch('api.services.get_file_content_or_raise')
    def test_failed_analysis_stores_failed_run(self, get_file_content_mock, get_repo_snapshot_mock, parse_repo_mock):
        get_repo_snapshot_mock.return_value = ('abc123', ['pkg/app.py'])
        get_file_content_mock.side_effect = RepoIngestionError('too_large', '분석 대상 파일이 허용 크기를 초과했습니다.')

        with self.assertRaises(RepoIngestionError):
            get_repo_analysis('owner/repo')

        parse_repo_mock.assert_not_called()
        failed_run = AnalysisRun.objects.get(status=AnalysisRun.STATUS_FAILED)
        self.assertEqual(failed_run.repository.full_name, 'owner/repo')
        self.assertEqual(failed_run.revision, 'abc123')
        self.assertEqual(failed_run.error_code, 'too_large')
        self.assertEqual(AnalysisArtifact.objects.count(), 0)

    @patch('api.services.get_repo_snapshot')
    def test_snapshot_failure_stores_failed_run_without_revision(self, get_repo_snapshot_mock):
        get_repo_snapshot_mock.side_effect = RepoIngestionError('timeout', 'Git 명령 시간이 초과되었습니다.')

        with self.assertRaises(RepoIngestionError):
            get_repo_analysis('owner/repo')

        failed_run = AnalysisRun.objects.get(status=AnalysisRun.STATUS_FAILED)
        self.assertEqual(failed_run.repository.full_name, 'owner/repo')
        self.assertEqual(failed_run.revision, '')
        self.assertEqual(failed_run.error_code, 'timeout')

    def test_get_repo_analysis_rejects_unsafe_cached_revision(self):
        self.assertIsNone(get_repo_analysis('owner/repo', '../../escaped'))

    def test_get_repo_analysis_rejects_dot_segment_repo_paths(self):
        self.assertIsNone(get_repo_analysis('../repo', 'abc123'))
        self.assertIsNone(get_repo_analysis('owner/..', 'abc123'))

    def test_get_repo_analysis_rejects_git_suffix_repo_paths(self):
        self.assertIsNone(get_repo_analysis('owner/repo.git', 'abc123'))


class AnalysisEndpointReuseTests(TestCase):
    @patch('api.views.get_file_tree_or_raise')
    def test_repo_endpoint_maps_ingestion_timeout_to_json_error(self, get_file_tree_mock):
        get_file_tree_mock.side_effect = RepoIngestionError('timeout', 'Git 명령 시간이 초과되었습니다.', command_category='clone')

        response = cast(HttpResponse, self.client.get('/api/repo/', {'url': 'https://github.com/owner/repo'}))
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 504)
        self.assertEqual(payload['code'], 'timeout')
        self.assertEqual(payload['error'], 'Git 명령 시간이 초과되었습니다.')

    @patch('api.views.get_repo_analysis')
    def test_graph_endpoint_maps_repo_too_large_to_json_error(self, get_repo_analysis_mock):
        get_repo_analysis_mock.side_effect = RepoIngestionError(
            'too_large',
            '레포 파일 수가 허용 한도를 초과했습니다.',
            metadata={'limit_type': 'max_files'},
        )

        response = cast(HttpResponse, self.client.get('/api/graph/', {'url': 'https://github.com/owner/repo'}))
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 413)
        self.assertEqual(payload['code'], 'too_large')
        detail = cast(dict[str, object], payload['detail'])
        metadata = cast(dict[str, object], detail['metadata'])
        self.assertEqual(metadata['limit_type'], 'max_files')

    @patch('api.views.get_repo_analysis')
    def test_tree_endpoint_uses_cached_analysis(self, get_repo_analysis_mock):
        get_repo_analysis_mock.return_value = {
            'repo': 'owner/repo',
            'revision': 'abc123',
            'tree': [{'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'children': []}],
            'nodes': [],
            'edges': [],
            'warnings': [{'code': 'demo', 'path': 'pkg/app.py'}],
        }

        response = cast(HttpResponse, self.client.get('/api/tree/', {'url': 'https://github.com/owner/repo'}))
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(payload['analysis_id'])
        self.assertEqual(payload['revision'], 'abc123')
        self.assertEqual(cast(list[dict[str, object]], payload['tree'])[0]['id'], 'pkg/app.py')
        self.assertEqual(payload['warnings'], [{'code': 'demo', 'path': 'pkg/app.py'}])

    @patch('api.views.get_repo_analysis')
    def test_graph_endpoint_uses_cached_analysis(self, get_repo_analysis_mock):
        get_repo_analysis_mock.return_value = {
            'repo': 'owner/repo',
            'revision': 'abc123',
            'tree': [],
            'nodes': [{'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'file': 'pkg/app.py'}],
            'edges': [{'id': 'e1', 'source': 'pkg/app.py', 'target': 'pkg/app.py::main', 'type': 'contains', 'file': 'pkg/app.py'}],
            'entrypoints': [{'id': 'pkg/app.py::main', 'kind': 'main_function', 'path': 'pkg/app.py'}],
            'key_modules': [{'id': 'module::pkg.app', 'path': 'pkg/app.py', 'score': 3}],
            'warnings': [{'code': 'demo', 'path': 'pkg/app.py'}],
        }

        response = cast(HttpResponse, self.client.get('/api/graph/', {'url': 'https://github.com/owner/repo'}))
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(payload['analysis_id'])
        self.assertEqual(payload['revision'], 'abc123')
        self.assertEqual(cast(list[dict[str, object]], payload['nodes'])[0]['id'], 'pkg/app.py')
        self.assertEqual(cast(list[dict[str, object]], payload['edges'])[0]['id'], 'e1')
        self.assertEqual(payload['entrypoints'], [{'id': 'pkg/app.py::main', 'kind': 'main_function', 'path': 'pkg/app.py'}])
        self.assertEqual(payload['key_modules'], [{'id': 'module::pkg.app', 'path': 'pkg/app.py', 'score': 3}])
        self.assertEqual(payload['warnings'], [{'code': 'demo', 'path': 'pkg/app.py'}])

    @patch('api.views.answer_question')
    @patch('api.views.get_repo_analysis')
    def test_qa_endpoint_returns_answer_with_citations(self, get_repo_analysis_mock, answer_question_mock):
        get_repo_analysis_mock.return_value = {
            'repo': 'owner/repo',
            'revision': 'abc123',
            'tree': [],
            'nodes': [],
            'edges': [],
        }
        answer_question_mock.return_value = {
            'answer': 'builder.py에서 처리합니다.',
            'citations': ['sample_pkg/factory.py'],
        }

        response = cast(
            HttpResponse,
            self.client.post(
                '/api/qa/',
                data={'repo_url': 'https://github.com/owner/repo', 'question': 'Where is load_component defined?'},
                content_type='application/json',
            ),
        )
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload['answer'], 'builder.py에서 처리합니다.')
        self.assertEqual(payload['citations'], ['sample_pkg/factory.py'])

    @patch('api.views.answer_question')
    @patch('api.views.get_analysis_response_by_id')
    def test_qa_endpoint_accepts_analysis_id_and_context_options(self, get_analysis_response_by_id_mock, answer_question_mock):
        get_analysis_response_by_id_mock.return_value = {
            'analysis_id': 7,
            'repo': 'owner/repo',
            'revision': 'abc123',
            'status': AnalysisRun.STATUS_SUCCEEDED,
            'artifact': {
                'repo': 'owner/repo',
                'revision': 'abc123',
                'tree': [],
                'nodes': [{'id': 'pkg/app.py::main', 'path': 'pkg/app.py', 'label': 'main'}],
                'edges': [],
                'file_contents': {'pkg/app.py': 'def main():\n    return "ok"\n'},
            },
            'warnings': [],
        }
        answer_question_mock.return_value = {
            'answer': 'main입니다.',
            'citations': ['pkg/app.py'],
            'selected_nodes': ['pkg/app.py::main'],
            'context_files': ['pkg/app.py'],
            'context_summary': {'strategy': 'selected_node'},
            'warnings': [],
        }

        response = cast(
            HttpResponse,
            self.client.post(
                '/api/qa/',
                data={
                    'analysis_id': 7,
                    'question': 'main은 어디인가요?',
                    'selected_node_id': 'pkg/app.py::main',
                    'selected_file_path': 'pkg/app.py',
                    'max_context_files': 2,
                },
                content_type='application/json',
            ),
        )
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload['selected_nodes'], ['pkg/app.py::main'])
        answer_question_mock.assert_called_once()
        self.assertEqual(answer_question_mock.call_args.args[0], 'owner/repo')
        self.assertEqual(answer_question_mock.call_args.kwargs['selected_node_id'], 'pkg/app.py::main')
        self.assertEqual(answer_question_mock.call_args.kwargs['selected_file_path'], 'pkg/app.py')
        self.assertEqual(answer_question_mock.call_args.kwargs['max_context_files'], 2)

    def test_qa_endpoint_requires_repo_url_or_analysis_id(self):
        response = cast(
            HttpResponse,
            self.client.post(
                '/api/qa/',
                data={'question': '무엇을 하나요?'},
                content_type='application/json',
            ),
        )
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 400)
        self.assertIn('repo_url', payload)

    def test_qa_endpoint_rejects_unsafe_selected_file_path(self):
        response = cast(
            HttpResponse,
            self.client.post(
                '/api/qa/',
                data={
                    'repo_url': 'https://github.com/owner/repo',
                    'question': '무엇을 하나요?',
                    'selected_file_path': '../settings.py',
                },
                content_type='application/json',
            ),
        )
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 400)
        self.assertIn('selected_file_path', payload)


@override_settings(
    TEMP_DIR=settings.BASE_DIR / 'temp' / 'analysis-endpoint-tests',
    PLAYGROUND_DIR=settings.BASE_DIR / 'temp' / 'analysis-endpoint-tests' / 'playground',
)
class AnalysisEndpointVerticalSliceTests(TestCase):
    source_repo: Path = Path()

    def setUp(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        settings.PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)
        self.source_repo = settings.TEMP_DIR / 'source-repo'
        create_git_fixture_repo(
            self.source_repo,
            {
                'pkg/app.py': (
                    'def main():\n'
                    '    return "ok"\n'
                ),
                'README.md': '# demo\n',
            },
            commit_message='init',
        )

    def tearDown(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)

    @patch('github_repo.services._repo_clone_url')
    def test_post_analysis_returns_artifact_with_analysis_id(self, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)

        response = cast(
            HttpResponse,
            self.client.post(
                '/api/analysis/',
                data={'repo_url': 'https://github.com/owner/repo'},
                content_type='application/json',
            ),
        )
        payload = cast(dict[str, object], json.loads(response.content))
        artifact = cast(dict[str, object], payload['artifact'])
        node_ids = {node['id'] for node in cast(list[dict[str, object]], artifact['nodes'])}

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(payload['analysis_id'], int)
        self.assertEqual(payload['repo'], 'owner/repo')
        self.assertEqual(payload['status'], AnalysisRun.STATUS_SUCCEEDED)
        self.assertEqual(artifact['repo'], 'owner/repo')
        self.assertIn('module::pkg.app', node_ids)
        self.assertIn('pkg/app.py::main', node_ids)
        self.assertEqual(Repository.objects.count(), 1)
        self.assertEqual(AnalysisRun.objects.filter(status=AnalysisRun.STATUS_SUCCEEDED).count(), 1)
        self.assertEqual(AnalysisArtifact.objects.count(), 1)

    @patch('github_repo.services._repo_clone_url')
    def test_same_revision_returns_cached_analysis_run(self, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)

        first = cast(
            HttpResponse,
            self.client.post(
                '/api/analysis/',
                data={'repo_url': 'https://github.com/owner/repo'},
                content_type='application/json',
            ),
        )
        second = cast(
            HttpResponse,
            self.client.post(
                '/api/analysis/',
                data={'repo_url': 'https://github.com/owner/repo'},
                content_type='application/json',
            ),
        )
        first_payload = cast(dict[str, object], json.loads(first.content))
        second_payload = cast(dict[str, object], json.loads(second.content))

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first_payload['analysis_id'], second_payload['analysis_id'])
        self.assertEqual(first_payload['revision'], second_payload['revision'])
        self.assertEqual(AnalysisRun.objects.filter(status=AnalysisRun.STATUS_SUCCEEDED).count(), 1)
        self.assertEqual(AnalysisArtifact.objects.count(), 1)

    @patch('github_repo.services._repo_clone_url')
    def test_get_analysis_can_pin_cached_revision(self, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)
        created = cast(
            HttpResponse,
            self.client.post(
                '/api/analysis/',
                data={'repo_url': 'https://github.com/owner/repo'},
                content_type='application/json',
            ),
        )
        created_payload = cast(dict[str, object], json.loads(created.content))

        response = cast(
            HttpResponse,
            self.client.get(
                '/api/analysis/',
                {
                    'url': 'https://github.com/owner/repo',
                    'revision': created_payload['revision'],
                },
            ),
        )
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload['analysis_id'], created_payload['analysis_id'])
        self.assertEqual(payload['revision'], created_payload['revision'])

    @patch('github_repo.services._repo_clone_url')
    def test_tree_and_graph_endpoints_reuse_analysis_artifact(self, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)
        created = cast(
            HttpResponse,
            self.client.post(
                '/api/analysis/',
                data={'repo_url': 'https://github.com/owner/repo'},
                content_type='application/json',
            ),
        )
        created_payload = cast(dict[str, object], json.loads(created.content))
        artifact = cast(dict[str, object], created_payload['artifact'])
        query = {
            'url': 'https://github.com/owner/repo',
            'revision': created_payload['revision'],
        }

        tree_response = cast(HttpResponse, self.client.get('/api/tree/', query))
        graph_response = cast(HttpResponse, self.client.get('/api/graph/', query))
        tree_payload = cast(dict[str, object], json.loads(tree_response.content))
        graph_payload = cast(dict[str, object], json.loads(graph_response.content))

        self.assertEqual(tree_response.status_code, 200)
        self.assertEqual(graph_response.status_code, 200)
        self.assertEqual(tree_payload['analysis_id'], created_payload['analysis_id'])
        self.assertEqual(graph_payload['analysis_id'], created_payload['analysis_id'])
        self.assertEqual(tree_payload['tree'], artifact['tree'])
        self.assertEqual(graph_payload['nodes'], artifact['nodes'])
        self.assertEqual(graph_payload['edges'], artifact['edges'])

    @patch('github_repo.services._repo_clone_url')
    def test_get_analysis_detail_returns_stored_artifact(self, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)
        created = cast(
            HttpResponse,
            self.client.post(
                '/api/analysis/',
                data={'repo_url': 'https://github.com/owner/repo'},
                content_type='application/json',
            ),
        )
        created_payload = cast(dict[str, object], json.loads(created.content))
        analysis_id = created_payload['analysis_id']

        response = cast(HttpResponse, self.client.get(f'/api/analysis/{analysis_id}/'))
        payload = cast(dict[str, object], json.loads(response.content))
        artifact = cast(dict[str, object], payload['artifact'])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload['analysis_id'], analysis_id)
        self.assertEqual(artifact['revision'], created_payload['revision'])

    def test_analysis_endpoint_rejects_invalid_url(self):
        response = cast(
            HttpResponse,
            self.client.post(
                '/api/analysis/',
                data={'repo_url': 'https://example.com/owner/repo'},
                content_type='application/json',
            ),
        )
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 400)
        self.assertIn('repo_url', payload)

    @patch('api.views.get_analysis_response')
    def test_analysis_endpoint_maps_repo_too_large_error(self, get_analysis_response_mock):
        get_analysis_response_mock.side_effect = RepoIngestionError(
            'too_large',
            '레포 파일 수가 허용 한도를 초과했습니다.',
            metadata={'limit_type': 'max_files'},
        )

        response = cast(
            HttpResponse,
            self.client.post(
                '/api/analysis/',
                data={'repo_url': 'https://github.com/owner/repo'},
                content_type='application/json',
            ),
        )
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 413)
        self.assertEqual(payload['code'], 'too_large')


class SummaryEndpointTests(TestCase):
    def setUp(self):
        repository = Repository.objects.create(
            provider='github',
            owner='owner',
            name='repo',
            full_name='owner/repo',
            clone_url='https://github.com/owner/repo.git',
        )
        self.analysis_run = AnalysisRun.objects.create(
            repository=repository,
            revision='abc123',
            status=AnalysisRun.STATUS_SUCCEEDED,
        )
        payload = build_graph_artifact(
            repo_path='owner/repo',
            revision='abc123',
            graph={
                'tree': [],
                'nodes': [
                    {'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'file': 'pkg/app.py'},
                    {'id': 'module::pkg.app', 'type': 'module', 'label': 'pkg.app', 'file': 'pkg/app.py'},
                    {'id': 'pkg/app.py::main', 'type': 'function', 'label': 'main', 'file': 'pkg/app.py', 'start_line': 1, 'end_line': 2},
                ],
                'edges': [
                    {'id': 'e1', 'type': 'contains', 'source': 'module::pkg.app', 'target': 'pkg/app.py::main', 'file': 'pkg/app.py'},
                ],
            },
            file_contents={'pkg/app.py': 'def main():\n    return "ok"\n'},
            entrypoints=[{'id': 'pkg/app.py::main', 'kind': 'main_function', 'path': 'pkg/app.py'}],
            key_modules=[{'id': 'module::pkg.app', 'path': 'pkg/app.py', 'score': 10}],
        )
        AnalysisArtifact.objects.create(
            analysis_run=self.analysis_run,
            schema_version=GRAPH_ARTIFACT_SCHEMA_VERSION,
            payload=payload,
            node_count=len(payload['nodes']),
            edge_count=len(payload['edges']),
        )

    @patch('llm.summaries._generate_answer')
    def test_summary_endpoint_generates_and_caches_repo_summary(self, generate_answer):
        generate_answer.return_value = '레포 요약입니다.'

        first = cast(HttpResponse, self.client.get('/api/summary/', {'analysis_id': self.analysis_run.id}))
        second = cast(HttpResponse, self.client.get('/api/summary/', {'analysis_id': self.analysis_run.id}))
        first_payload = cast(dict[str, object], json.loads(first.content))
        second_payload = cast(dict[str, object], json.loads(second.content))
        summary = cast(dict[str, object], first_payload['summary'])

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertFalse(first_payload['cached'])
        self.assertTrue(second_payload['cached'])
        self.assertEqual(summary['text'], '레포 요약입니다.')
        self.assertIn('pkg/app.py', summary['source_files'])
        generate_answer.assert_called_once()

    @patch('llm.summaries._generate_answer')
    def test_summary_prompt_version_change_invalidates_cache(self, generate_answer):
        generate_answer.side_effect = ['v1 summary', 'v2 summary']

        first = cast(HttpResponse, self.client.get('/api/summary/', {'analysis_id': self.analysis_run.id}))
        with patch('llm.summaries.SUMMARY_PROMPT_VERSION', 'summary.v2'):
            second = cast(HttpResponse, self.client.get('/api/summary/', {'analysis_id': self.analysis_run.id}))
        first_payload = cast(dict[str, object], json.loads(first.content))
        second_payload = cast(dict[str, object], json.loads(second.content))

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(cast(dict[str, object], first_payload['summary'])['prompt_version'], 'summary.v1')
        self.assertEqual(cast(dict[str, object], second_payload['summary'])['prompt_version'], 'summary.v2')
        self.assertEqual(generate_answer.call_count, 2)
        artifact = AnalysisArtifact.objects.get(analysis_run=self.analysis_run)
        self.assertEqual(len(cast(dict[str, object], artifact.payload['summaries'])), 2)

    @patch('llm.summaries._generate_answer', side_effect=RuntimeError('사용 가능한 AI API 키가 없습니다.'))
    def test_summary_endpoint_returns_503_when_model_unavailable(self, generate_answer):
        response = cast(HttpResponse, self.client.get('/api/summary/', {'analysis_id': self.analysis_run.id}))
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload['code'], 'summary_unavailable')
        self.assertEqual(AnalysisArtifact.objects.get(analysis_run=self.analysis_run).payload['summaries'], {})

    @patch('llm.summaries._generate_answer')
    def test_node_summary_endpoint_returns_source_citations(self, generate_answer):
        generate_answer.return_value = 'main 함수 설명입니다.'

        response = cast(
            HttpResponse,
            self.client.get(
                '/api/node-summary/',
                {
                    'analysis_id': self.analysis_run.id,
                    'node_id': 'pkg/app.py::main',
                },
            ),
        )
        payload = cast(dict[str, object], json.loads(response.content))
        summary = cast(dict[str, object], payload['summary'])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(summary['kind'], 'node')
        self.assertEqual(summary['target_id'], 'pkg/app.py::main')
        self.assertIn('pkg/app.py::main', summary['source_nodes'])
        self.assertIn('pkg/app.py', summary['source_files'])

    @patch('llm.summaries._generate_answer')
    def test_node_summary_missing_node_returns_400(self, generate_answer):
        response = cast(
            HttpResponse,
            self.client.get(
                '/api/node-summary/',
                {
                    'analysis_id': self.analysis_run.id,
                    'node_id': 'pkg/app.py::missing',
                },
            ),
        )
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload['code'], 'summary_input_error')
        generate_answer.assert_not_called()


@override_settings(
    TEMP_DIR=settings.BASE_DIR / 'temp' / 'revision-api-tests',
    PLAYGROUND_DIR=settings.BASE_DIR / 'temp' / 'revision-api-tests' / 'playground',
)
class RevisionPinnedApiTests(TestCase):
    source_repo: Path = Path()

    def setUp(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        settings.PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)
        self.source_repo = settings.TEMP_DIR / 'source-repo'
        create_git_fixture_repo(self.source_repo, {'pkg/app.py': 'def greet():\n    return "v1"\n'}, commit_message='v1')

    def tearDown(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)

    @patch('github_repo.services._repo_clone_url')
    @patch('api.views.answer_question')
    def test_qa_can_pin_cached_revision_after_upstream_changes(self, answer_question_mock, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)
        analysis = get_repo_analysis('owner/repo')
        self.assertIsNotNone(analysis)
        revision = cast(dict[str, object], analysis)['revision']

        write_files(self.source_repo, {'pkg/app.py': 'def greet():\n    return "v2"\n'})
        commit_all(self.source_repo, 'v2')

        answer_question_mock.return_value = {'answer': 'v1', 'citations': ['pkg/app.py']}
        response = cast(
            HttpResponse,
            self.client.post(
                '/api/qa/',
                data={'repo_url': 'https://github.com/owner/repo', 'question': 'Where is greet defined?', 'revision': revision},
                content_type='application/json',
            ),
        )

        self.assertEqual(response.status_code, 200)
        called_analysis = answer_question_mock.call_args.args[1]
        self.assertEqual(called_analysis['revision'], revision)

    @patch('github_repo.services._repo_clone_url')
    def test_pinned_artifact_loads_from_db_after_temp_and_playground_cleanup(self, repo_clone_url):
        repo_clone_url.return_value = str(self.source_repo)
        analysis = get_repo_analysis('owner/repo')
        self.assertIsNotNone(analysis)
        revision = cast(dict[str, object], analysis)['revision']
        self.assertIsNotNone(get_artifact_by_revision('owner/repo', cast(str, revision)))

        shutil.rmtree(settings.TEMP_DIR / 'analysis', ignore_errors=True)
        shutil.rmtree(settings.PLAYGROUND_DIR, ignore_errors=True)

        cached_analysis = get_repo_analysis('owner/repo', cast(str, revision))

        self.assertIsNotNone(cached_analysis)
        cached_payload = cast(dict[str, object], cached_analysis)
        self.assertEqual(cached_payload['revision'], revision)
        self.assertEqual(cached_payload['file_contents'], {'pkg/app.py': 'def greet():\n    return "v1"\n'})

    def test_tree_endpoint_rejects_unsafe_revision(self):
        response = cast(HttpResponse, self.client.get('/api/tree/', {'url': 'https://github.com/owner/repo', 'revision': '../../escaped'}))
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload['revision'], ['올바른 revision이 아닙니다'])

    def test_graph_endpoint_rejects_unsafe_revision(self):
        response = cast(HttpResponse, self.client.get('/api/graph/', {'url': 'https://github.com/owner/repo', 'revision': 'a/b'}))
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload['revision'], ['올바른 revision이 아닙니다'])

    def test_qa_endpoint_rejects_unsafe_revision(self):
        response = cast(
            HttpResponse,
            self.client.post(
                '/api/qa/',
                data={'repo_url': 'https://github.com/owner/repo', 'question': 'Where is greet defined?', 'revision': '../../escaped'},
                content_type='application/json',
            ),
        )
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload['revision'], ['올바른 revision이 아닙니다'])


@override_settings(
    TEMP_DIR=settings.BASE_DIR / 'temp' / 'diff-api-tests',
    PLAYGROUND_DIR=settings.BASE_DIR / 'temp' / 'diff-api-tests' / 'playground',
)
class DiffLatestRefreshApiTests(TestCase):
    source_repo: Path = Path()

    def setUp(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        settings.PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)
        self.source_repo = settings.TEMP_DIR / 'source-repo'
        create_git_fixture_repo(self.source_repo, {'pkg/app.py': 'def greet():\n    return "v1"\n'}, commit_message='v1')

    def tearDown(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)

    def _clone_url(self) -> str:
        return f'file://{self.source_repo}'

    @patch('github_repo.services._repo_clone_url')
    def test_diff_without_head_uses_latest_refresh_and_reuses_same_revision(self, repo_clone_url):
        repo_clone_url.return_value = self._clone_url()
        base_analysis = get_repo_analysis('owner/repo')
        self.assertIsNotNone(base_analysis)
        base_revision = cast(dict[str, object], base_analysis)['revision']

        response = cast(HttpResponse, self.client.get('/api/diff/', {'url': 'https://github.com/owner/repo', 'base': base_revision}))
        payload = cast(dict[str, object], json.loads(response.content))
        diff = cast(dict[str, object], payload['diff'])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(cast(dict[str, object], payload['base'])['revision'], base_revision)
        self.assertEqual(cast(dict[str, object], payload['head'])['revision'], base_revision)
        self.assertEqual(cast(dict[str, object], diff['summary'])['added_nodes'], 0)
        self.assertEqual(AnalysisRun.objects.filter(status=AnalysisRun.STATUS_SUCCEEDED).count(), 1)

    @patch('github_repo.services._repo_clone_url')
    def test_diff_without_head_refreshes_latest_when_remote_head_changes(self, repo_clone_url):
        repo_clone_url.return_value = self._clone_url()
        base_analysis = get_repo_analysis('owner/repo')
        self.assertIsNotNone(base_analysis)
        base_revision = cast(dict[str, object], base_analysis)['revision']
        write_files(
            self.source_repo,
            {'pkg/app.py': 'def greet():\n    return "v2"\n\ndef helper():\n    return greet()\n'},
        )
        head_revision = commit_all(self.source_repo, 'v2')

        response = cast(HttpResponse, self.client.get('/api/diff/', {'url': 'https://github.com/owner/repo', 'base': base_revision}))
        payload = cast(dict[str, object], json.loads(response.content))
        diff = cast(dict[str, object], payload['diff'])
        summary = cast(dict[str, object], diff['summary'])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(cast(dict[str, object], payload['head'])['revision'], head_revision)
        self.assertGreaterEqual(cast(int, summary['added_nodes']), 1)
        self.assertEqual(AnalysisRun.objects.filter(status=AnalysisRun.STATUS_SUCCEEDED).count(), 2)
        self.assertEqual(AnalysisArtifact.objects.count(), 2)

    @patch('github_repo.services._repo_clone_url')
    def test_diff_endpoint_fetches_missing_revision_from_shallow_history(self, repo_clone_url):
        repo_clone_url.return_value = self._clone_url()
        base_revision = run_git(self.source_repo, 'rev-parse', 'HEAD')
        write_files(
            self.source_repo,
            {'pkg/app.py': 'def greet():\n    return "v2"\n\ndef helper():\n    return greet()\n'},
        )
        head_revision = commit_all(self.source_repo, 'v2')
        self.assertIsNotNone(get_repo_analysis('owner/repo', head_revision))

        response = cast(
            HttpResponse,
            self.client.get(
                '/api/diff/',
                {'url': 'https://github.com/owner/repo', 'base': base_revision, 'head': head_revision},
            ),
        )
        payload = cast(dict[str, object], json.loads(response.content))
        diff = cast(dict[str, object], payload['diff'])
        summary = cast(dict[str, object], diff['summary'])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(cast(dict[str, object], payload['base'])['revision'], base_revision)
        self.assertEqual(cast(dict[str, object], payload['head'])['revision'], head_revision)
        self.assertGreaterEqual(cast(int, summary['added_nodes']), 1)

    @patch('github_repo.services._repo_clone_url')
    def test_analysis_id_diff_compares_stored_artifacts(self, repo_clone_url):
        repo_clone_url.return_value = self._clone_url()
        base_analysis = get_repo_analysis('owner/repo')
        self.assertIsNotNone(base_analysis)
        write_files(
            self.source_repo,
            {'pkg/app.py': 'def greet():\n    return "v2"\n\ndef helper():\n    return greet()\n'},
        )
        commit_all(self.source_repo, 'v2')
        head_analysis = get_repo_analysis('owner/repo')
        self.assertIsNotNone(head_analysis)
        base_run = get_artifact_by_revision('owner/repo', cast(str, cast(dict[str, object], base_analysis)['revision']))
        head_run = get_artifact_by_revision('owner/repo', cast(str, cast(dict[str, object], head_analysis)['revision']))
        self.assertIsNotNone(base_run)
        self.assertIsNotNone(head_run)
        base_analysis_run = AnalysisRun.objects.get(revision=cast(dict[str, object], base_analysis)['revision'])
        head_analysis_run = AnalysisRun.objects.get(revision=cast(dict[str, object], head_analysis)['revision'])

        response = cast(HttpResponse, self.client.get(f'/api/analysis/{head_analysis_run.id}/diff/', {'base': base_analysis_run.id}))
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(cast(dict[str, object], payload['base'])['analysis_id'], base_analysis_run.id)
        self.assertEqual(cast(dict[str, object], payload['head'])['analysis_id'], head_analysis_run.id)

    def test_diff_endpoint_rejects_invalid_revision(self):
        response = cast(HttpResponse, self.client.get('/api/diff/', {'url': 'https://github.com/owner/repo', 'base': '-abc123'}))
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 400)
        self.assertIn('base', payload)

    @patch('github_repo.services._repo_clone_url')
    def test_diff_endpoint_returns_404_for_missing_revision(self, repo_clone_url):
        repo_clone_url.return_value = self._clone_url()

        response = cast(HttpResponse, self.client.get('/api/diff/', {'url': 'https://github.com/owner/repo', 'base': 'deadbeef'}))
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 404)
        self.assertEqual(payload['code'], 'revision_not_found')


@override_settings(
    TEMP_DIR=settings.BASE_DIR / 'temp' / 'share-api-tests',
    PLAYGROUND_DIR=settings.BASE_DIR / 'temp' / 'share-api-tests' / 'playground',
)
class ShareEmbedPublicApiTests(TestCase):
    source_repo: Path = Path()

    def setUp(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        settings.PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)
        self.source_repo = settings.TEMP_DIR / 'source-repo'
        create_git_fixture_repo(self.source_repo, {'pkg/app.py': 'def greet():\n    return "v1"\n'}, commit_message='v1')

    def tearDown(self):
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)

    def _clone_url(self) -> str:
        return f'file://{self.source_repo}'

    def _create_share(self, *, mode='fixed', revision=None):
        data = {'repo_url': 'https://github.com/owner/repo', 'mode': mode}
        if revision is not None:
            data['revision'] = revision
        return cast(
            HttpResponse,
            self.client.post('/api/share/', data=data, content_type='application/json'),
        )

    @patch('github_repo.services._repo_clone_url')
    def test_create_fixed_share_returns_token_and_public_graph(self, repo_clone_url):
        repo_clone_url.return_value = self._clone_url()

        response = self._create_share()
        payload = cast(dict[str, object], json.loads(response.content))
        graph = cast(dict[str, object], payload['graph'])

        self.assertEqual(response.status_code, 201)
        self.assertTrue(is_safe_share_id(cast(str, payload['share_id'])))
        self.assertEqual(payload['mode'], ShareLink.MODE_FIXED)
        self.assertEqual(payload['repo'], 'owner/repo')
        self.assertIn('nodes', graph)
        self.assertNotIn('file_contents', graph)
        self.assertIn('markdown_link', cast(dict[str, object], payload['snippets']))
        self.assertIn('/api/embed/', cast(dict[str, object], payload['urls'])['embed'])
        self.assertEqual(ShareLink.objects.count(), 1)

    @patch('github_repo.services._repo_clone_url')
    def test_create_latest_share_and_retrieve(self, repo_clone_url):
        repo_clone_url.return_value = self._clone_url()

        create_response = self._create_share(mode='latest')
        create_payload = cast(dict[str, object], json.loads(create_response.content))
        share_id = cast(str, create_payload['share_id'])
        retrieve_response = cast(HttpResponse, self.client.get(f'/api/share/{share_id}/'))
        retrieve_payload = cast(dict[str, object], json.loads(retrieve_response.content))

        self.assertEqual(create_response.status_code, 201)
        self.assertEqual(retrieve_response.status_code, 200)
        self.assertEqual(retrieve_payload['mode'], ShareLink.MODE_LATEST)
        self.assertEqual(retrieve_payload['share_id'], share_id)

    def test_invalid_share_id_returns_404(self):
        response = cast(HttpResponse, self.client.get('/api/share/short/'))

        self.assertEqual(response.status_code, 404)

    @patch('github_repo.services._repo_clone_url')
    def test_expired_share_returns_404(self, repo_clone_url):
        repo_clone_url.return_value = self._clone_url()
        create_response = self._create_share()
        create_payload = cast(dict[str, object], json.loads(create_response.content))
        share_link = ShareLink.objects.get(token=create_payload['share_id'])
        share_link.expires_at = timezone.now() - timedelta(minutes=1)
        share_link.save(update_fields=['expires_at'])

        response = cast(HttpResponse, self.client.get(f'/api/share/{create_payload["share_id"]}/'))

        self.assertEqual(response.status_code, 404)

    @patch('github_repo.services._repo_clone_url')
    def test_fixed_share_stays_on_original_revision_after_remote_changes(self, repo_clone_url):
        repo_clone_url.return_value = self._clone_url()
        create_response = self._create_share(mode='fixed')
        create_payload = cast(dict[str, object], json.loads(create_response.content))
        fixed_revision = create_payload['revision']
        write_files(
            self.source_repo,
            {'pkg/app.py': 'def greet():\n    return "v2"\n\ndef helper():\n    return greet()\n'},
        )
        commit_all(self.source_repo, 'v2')

        retrieve_response = cast(HttpResponse, self.client.get(f'/api/share/{create_payload["share_id"]}/'))
        retrieve_payload = cast(dict[str, object], json.loads(retrieve_response.content))

        self.assertEqual(retrieve_response.status_code, 200)
        self.assertEqual(retrieve_payload['revision'], fixed_revision)
        self.assertEqual(AnalysisRun.objects.filter(status=AnalysisRun.STATUS_SUCCEEDED).count(), 1)

    @patch('github_repo.services._repo_clone_url')
    def test_latest_share_resolves_new_revision(self, repo_clone_url):
        repo_clone_url.return_value = self._clone_url()
        create_response = self._create_share(mode='latest')
        create_payload = cast(dict[str, object], json.loads(create_response.content))
        write_files(
            self.source_repo,
            {'pkg/app.py': 'def greet():\n    return "v2"\n\ndef helper():\n    return greet()\n'},
        )
        head_revision = commit_all(self.source_repo, 'v2')

        retrieve_response = cast(HttpResponse, self.client.get(f'/api/share/{create_payload["share_id"]}/'))
        retrieve_payload = cast(dict[str, object], json.loads(retrieve_response.content))
        share_link = ShareLink.objects.get(token=create_payload['share_id'])

        self.assertEqual(retrieve_response.status_code, 200)
        self.assertEqual(retrieve_payload['revision'], head_revision)
        self.assertEqual(share_link.analysis_run.revision, head_revision)
        self.assertEqual(AnalysisArtifact.objects.count(), 2)

    @patch('github_repo.services._repo_clone_url')
    def test_embed_endpoint_returns_html_without_global_frame_header(self, repo_clone_url):
        repo_clone_url.return_value = self._clone_url()
        create_response = self._create_share()
        create_payload = cast(dict[str, object], json.loads(create_response.content))

        response = cast(HttpResponse, self.client.get(f'/api/embed/{create_payload["share_id"]}/'))

        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.headers.get('Content-Type', ''))
        self.assertContains(response, 'graph-data')
        self.assertNotIn('X-Frame-Options', response.headers)

    @patch('github_repo.services._repo_clone_url')
    def test_share_id_is_not_sequential(self, repo_clone_url):
        repo_clone_url.return_value = self._clone_url()

        first = cast(dict[str, object], json.loads(self._create_share().content))
        second = cast(dict[str, object], json.loads(self._create_share().content))

        self.assertNotEqual(first['share_id'], second['share_id'])
        self.assertFalse(cast(str, first['share_id']).isdigit())
        self.assertFalse(cast(str, second['share_id']).isdigit())

    @patch('github_repo.services._repo_clone_url')
    def test_latest_share_rejects_revision(self, repo_clone_url):
        repo_clone_url.return_value = self._clone_url()

        response = self._create_share(mode='latest', revision='abc123')
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 400)
        self.assertIn('revision', payload)

    def test_share_endpoint_rejects_unsupported_repo_url(self):
        response = cast(
            HttpResponse,
            self.client.post(
                '/api/share/',
                data={'repo_url': 'https://example.com/owner/repo', 'mode': 'fixed'},
                content_type='application/json',
            ),
        )
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 400)
        self.assertIn('repo_url', payload)

    @patch('api.views.create_share_response')
    def test_share_endpoint_forbids_private_repo(self, create_share_response_mock):
        create_share_response_mock.side_effect = RepoIngestionError('private_repo', '레포 접근 권한이 없거나 private repository입니다.')

        response = self._create_share(mode='fixed')
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 404)
        self.assertEqual(payload['code'], 'private_repo')
        self.assertEqual(ShareLink.objects.count(), 0)


class SchemaRevisionDocumentationTests(TestCase):
    def test_schema_documents_revision_for_tree_graph_and_qa(self):
        response = cast(HttpResponse, self.client.get('/api/schema/'))
        schema_text = response.content.decode('utf-8')

        self.assertIn('/api/analysis/', schema_text)
        self.assertIn('/api/analysis/{analysis_id}/', schema_text)
        self.assertIn('/api/analysis/{analysis_id}/diff/', schema_text)
        self.assertIn('/api/diff/', schema_text)
        self.assertIn('/api/share/', schema_text)
        self.assertIn('/api/share/{share_id}/', schema_text)
        self.assertIn('/api/embed/{share_id}/', schema_text)
        self.assertIn('/api/tree/', schema_text)
        self.assertIn('/api/graph/', schema_text)
        self.assertIn('/api/summary/', schema_text)
        self.assertIn('/api/node-summary/', schema_text)
        self.assertIn('/api/qa/', schema_text)
        self.assertIn('revision', schema_text)
        self.assertIn('analysis_id', schema_text)
        self.assertIn('selected_node_id', schema_text)
        self.assertIn('context_summary', schema_text)
        self.assertIn('tool_trace', schema_text)
