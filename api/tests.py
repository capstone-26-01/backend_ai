from typing import cast
from pathlib import Path
from unittest.mock import patch
import importlib
import json
import shutil

from django.conf import settings
from django.http import HttpResponse
from django.test import TestCase, override_settings
from github_repo.services import get_file_content, get_file_tree, get_repo_revision, _repo_lock, _repo_lock_path
from api.artifacts import (
    GRAPH_ARTIFACT_SCHEMA_VERSION,
    ArtifactValidationError,
    build_graph_artifact,
    coerce_graph_artifact,
    validate_graph_artifact,
)
from api.serializers import extract_repo_path, is_safe_revision
from api.services import get_repo_analysis
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


class RepoUrlSerializerTests(TestCase):
    def test_extract_repo_path_accepts_owner_repo_only(self):
        self.assertEqual(extract_repo_path('https://github.com/owner/repo'), 'owner/repo')

    def test_extract_repo_path_rejects_extra_segments(self):
        self.assertIsNone(extract_repo_path('https://github.com/owner/repo/issues/1'))

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
        self.assertTrue(is_safe_revision('main'))


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
        self.assertIn(('pkg/models.py', 'pkg/models.py::Child', 'contains'), edges)
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
    def test_get_repo_analysis_caches_graph_payload(self, get_repo_snapshot_mock, parse_repo_mock):
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

    @patch('api.services.parse_repo')
    @patch('api.services.get_repo_snapshot')
    def test_get_repo_analysis_returns_v1_artifact_without_breaking_graph_aliases(self, get_repo_snapshot_mock, parse_repo_mock):
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
    def test_get_repo_analysis_refreshes_cache_for_new_revision(self, get_repo_snapshot_mock, parse_repo_mock):
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

    @patch('api.services.parse_repo')
    @patch('api.services.get_repo_snapshot')
    def test_get_repo_analysis_uses_single_snapshot_per_call(self, get_repo_snapshot_mock, parse_repo_mock):
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

    def test_get_repo_analysis_rejects_unsafe_cached_revision(self):
        self.assertIsNone(get_repo_analysis('owner/repo', '../../escaped'))

    def test_get_repo_analysis_rejects_dot_segment_repo_paths(self):
        self.assertIsNone(get_repo_analysis('../repo', 'abc123'))
        self.assertIsNone(get_repo_analysis('owner/..', 'abc123'))

    def test_get_repo_analysis_rejects_git_suffix_repo_paths(self):
        self.assertIsNone(get_repo_analysis('owner/repo.git', 'abc123'))


class AnalysisEndpointReuseTests(TestCase):
    @patch('api.views.get_repo_analysis')
    def test_tree_endpoint_uses_cached_analysis(self, get_repo_analysis_mock):
        get_repo_analysis_mock.return_value = {
            'repo': 'owner/repo',
            'revision': 'abc123',
            'tree': [{'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'children': []}],
            'nodes': [],
            'edges': [],
        }

        response = cast(HttpResponse, self.client.get('/api/tree/', {'url': 'https://github.com/owner/repo'}))
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload['revision'], 'abc123')
        self.assertEqual(cast(list[dict[str, object]], payload['tree'])[0]['id'], 'pkg/app.py')

    @patch('api.views.get_repo_analysis')
    def test_graph_endpoint_uses_cached_analysis(self, get_repo_analysis_mock):
        get_repo_analysis_mock.return_value = {
            'repo': 'owner/repo',
            'revision': 'abc123',
            'tree': [],
            'nodes': [{'id': 'pkg/app.py', 'type': 'file', 'label': 'app.py', 'file': 'pkg/app.py'}],
            'edges': [{'id': 'e1', 'source': 'pkg/app.py', 'target': 'pkg/app.py::main', 'type': 'contains', 'file': 'pkg/app.py'}],
        }

        response = cast(HttpResponse, self.client.get('/api/graph/', {'url': 'https://github.com/owner/repo'}))
        payload = cast(dict[str, object], json.loads(response.content))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload['revision'], 'abc123')
        self.assertEqual(cast(list[dict[str, object]], payload['nodes'])[0]['id'], 'pkg/app.py')
        self.assertEqual(cast(list[dict[str, object]], payload['edges'])[0]['id'], 'e1')

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


class SchemaRevisionDocumentationTests(TestCase):
    def test_schema_documents_revision_for_tree_graph_and_qa(self):
        response = cast(HttpResponse, self.client.get('/api/schema/'))
        schema_text = response.content.decode('utf-8')

        self.assertIn('/api/tree/', schema_text)
        self.assertIn('/api/graph/', schema_text)
        self.assertIn('/api/qa/', schema_text)
        self.assertIn('revision', schema_text)
