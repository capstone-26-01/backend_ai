from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Mapping
import shutil
import subprocess
from unittest.mock import patch
from urllib.parse import urlencode

import requests

from api.artifacts import build_graph_artifact


GIT_USER_NAME = 'Test User'
GIT_USER_EMAIL = 'test@example.com'


@dataclass(frozen=True)
class GitFixtureRepo:
    path: Path
    revision: str
    files: Mapping[str, str]


@dataclass(frozen=True)
class GoldenFixtureRepo:
    name: str
    files: Mapping[str, str]
    expected_nodes: tuple[str, ...]
    expected_edges: tuple[tuple[str, str, str], ...]
    rubric_tags: tuple[str, ...]


@dataclass(frozen=True)
class IssueMapFixture:
    repo_path: str
    revision: str
    files: Mapping[str, str]
    expected_node_ids: tuple[str, ...]
    expected_entrypoint_ids: tuple[str, ...]
    expected_key_module_ids: tuple[str, ...]


@dataclass(frozen=True)
class IssueMapGoldenCase:
    name: str
    issue: Mapping[str, Any]
    comments: tuple[Mapping[str, Any], ...]
    expected_node_ids: tuple[str, ...]
    expected_file_paths: tuple[str, ...]
    expected_top_node_id: str | None = None


@dataclass(frozen=True)
class IssueRankingCaseResult:
    name: str
    ranked_node_ids: tuple[str, ...]
    expected_node_ids: tuple[str, ...]
    expected_file_paths: tuple[str, ...]
    hits_at_k: Mapping[int, int]
    recall_at_k: Mapping[int, float]


@dataclass(frozen=True)
class MockGithubHttpResponse:
    payload: Any
    status_code: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)
    url: str = 'https://api.github.com/repos/owner/repo/issues'

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400

    @property
    def text(self) -> str:
        return '' if self.payload is None else str(self.payload)

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(f'{self.status_code} response for {self.url}', response=self)


@dataclass
class IssueLlmCallRecorder:
    response: Mapping[str, Any]
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def __call__(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        self.calls.append((args, kwargs))
        return dict(self.response)


EVAL_RUBRIC = {
    'graph_node_recall': 'Expected modules, files, classes, and functions from a fixture must appear in the graph.',
    'edge_correctness': 'Expected contains, imports, inherits, and calls edges must point to the intended source and target.',
    'entrypoint_correctness': 'Entrypoint-like symbols should be identifiable by graph metadata or deterministic hints.',
    'qa_citation_correctness': 'Q&A should cite the files that contain the code evidence used for the answer.',
}


ISSUE_MAP_FIXTURE = IssueMapFixture(
    repo_path='owner/repo',
    revision='abc123',
    files={
        'api/views.py': (
            'from api.services import get_repo_analysis\n\n'
            'def analysis(request):\n'
            '    return get_repo_analysis("owner/repo")\n'
        ),
        'api/services.py': (
            'from parser.services import parse_repo\n\n'
            'def get_repo_analysis(repo_path):\n'
            '    return _build_and_store_analysis(repo_path)\n\n'
            'def _build_and_store_analysis(repo_path):\n'
            '    return parse_repo({})\n'
        ),
        'parser/services.py': (
            'def parse_repo(files):\n'
            '    return {"nodes": [], "edges": []}\n'
        ),
    },
    expected_node_ids=(
        'api/views.py',
        'api/views.py::analysis',
        'api/services.py',
        'api/services.py::get_repo_analysis',
        'api/services.py::_build_and_store_analysis',
        'parser/services.py::parse_repo',
    ),
    expected_entrypoint_ids=('api/views.py::analysis',),
    expected_key_module_ids=('api/services.py::get_repo_analysis', 'parser/services.py::parse_repo'),
)


def build_issue_map_analysis_artifact(
    *,
    repo_path: str = ISSUE_MAP_FIXTURE.repo_path,
    revision: str = ISSUE_MAP_FIXTURE.revision,
) -> dict[str, Any]:
    graph = {
        'tree': [],
        'nodes': [
            {'id': 'api/views.py', 'type': 'file', 'label': 'views.py', 'file': 'api/views.py'},
            {
                'id': 'api/views.py::analysis',
                'type': 'function',
                'label': 'analysis',
                'file': 'api/views.py',
                'start_line': 3,
                'end_line': 4,
            },
            {'id': 'api/services.py', 'type': 'file', 'label': 'services.py', 'file': 'api/services.py'},
            {
                'id': 'api/services.py::get_repo_analysis',
                'type': 'function',
                'label': 'get_repo_analysis',
                'file': 'api/services.py',
                'start_line': 3,
                'end_line': 4,
            },
            {
                'id': 'api/services.py::_build_and_store_analysis',
                'type': 'function',
                'label': '_build_and_store_analysis',
                'file': 'api/services.py',
                'start_line': 6,
                'end_line': 7,
            },
            {
                'id': 'parser/services.py::parse_repo',
                'type': 'function',
                'label': 'parse_repo',
                'file': 'parser/services.py',
                'start_line': 1,
                'end_line': 2,
            },
        ],
        'edges': [
            {'source': 'api/views.py', 'target': 'api/views.py::analysis', 'type': 'contains', 'file': 'api/views.py'},
            {'source': 'api/services.py', 'target': 'api/services.py::get_repo_analysis', 'type': 'contains', 'file': 'api/services.py'},
            {'source': 'api/services.py', 'target': 'api/services.py::_build_and_store_analysis', 'type': 'contains', 'file': 'api/services.py'},
            {
                'source': 'api/views.py::analysis',
                'target': 'api/services.py::get_repo_analysis',
                'type': 'calls',
                'file': 'api/views.py',
            },
            {
                'source': 'api/services.py::get_repo_analysis',
                'target': 'api/services.py::_build_and_store_analysis',
                'type': 'calls',
                'file': 'api/services.py',
            },
            {
                'source': 'api/services.py::_build_and_store_analysis',
                'target': 'parser/services.py::parse_repo',
                'type': 'calls',
                'file': 'api/services.py',
            },
        ],
    }
    return build_graph_artifact(
        repo_path=repo_path,
        revision=revision,
        graph=graph,
        file_contents=ISSUE_MAP_FIXTURE.files,
        entrypoints=[
            {
                'id': 'api/views.py::analysis',
                'kind': 'django_view',
                'label': 'analysis',
                'path': 'api/views.py',
                'confidence': 0.9,
                'reason': 'API view fixture',
            }
        ],
        key_modules=[
            {'id': 'api/services.py::get_repo_analysis', 'path': 'api/services.py', 'score': 20},
            {'id': 'parser/services.py::parse_repo', 'path': 'parser/services.py', 'score': 15},
        ],
    )


def create_issue_map_fixture_repo(repo_dir: Path) -> GitFixtureRepo:
    return create_git_fixture_repo(repo_dir, ISSUE_MAP_FIXTURE.files, commit_message='issue-map fixture')


def github_issue_user(login: str = 'octocat') -> dict[str, str]:
    return {
        'login': login,
        'avatar_url': f'https://github.com/{login}.png',
        'html_url': f'https://github.com/{login}',
    }


def github_issue_label(name: str = 'bug', color: str = 'd73a4a', description: str | None = 'Something is not working') -> dict[str, str | None]:
    return {'name': name, 'color': color, 'description': description}


def github_issue_payload(
    *,
    repo_path: str = ISSUE_MAP_FIXTURE.repo_path,
    number: int = 42,
    title: str = 'Repository analysis fails on parser limits',
    body: str = 'Trace points at api/services.py::_build_and_store_analysis and parser/services.py.',
    labels: list[dict[str, Any]] | None = None,
    comments: int = 2,
    pull_request: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'number': number,
        'title': title,
        'state': 'open',
        'html_url': f'https://github.com/{repo_path}/issues/{number}',
        'user': github_issue_user(),
        'labels': labels if labels is not None else [github_issue_label(), github_issue_label('analysis', '1d76db', 'Analysis flow')],
        'assignees': [github_issue_user('backend-owner')],
        'comments': comments,
        'created_at': '2026-05-20T10:00:00Z',
        'updated_at': '2026-05-23T12:30:00Z',
        'body': body,
        'locked': False,
    }
    if pull_request:
        payload['pull_request'] = {'url': f'https://api.github.com/repos/{repo_path}/pulls/{number}'}
    return payload


def github_issue_comments_payload(*, count: int = 2) -> list[dict[str, Any]]:
    return [
        {
            'id': index,
            'user': github_issue_user(f'commenter-{index}'),
            'body': f'Comment {index}: check api/services.py and parse_repo.',
            'created_at': f'2026-05-2{index}T10:00:00Z',
            'updated_at': f'2026-05-2{index}T10:30:00Z',
            'html_url': f'https://github.com/owner/repo/issues/42#issuecomment-{index}',
        }
        for index in range(1, count + 1)
    ]


def github_issue_link_header(repo_path: str = ISSUE_MAP_FIXTURE.repo_path, *, page: int = 1, per_page: int = 30, state: str = 'open') -> str:
    query = urlencode({'state': state, 'page': page + 1, 'per_page': per_page})
    return f'<https://api.github.com/repos/{repo_path}/issues?{query}>; rel="next"'


def mock_github_issue_list_response(
    *,
    repo_path: str = ISSUE_MAP_FIXTURE.repo_path,
    include_pull_request: bool = True,
    has_next_page: bool = True,
) -> MockGithubHttpResponse:
    payload = [
        github_issue_payload(repo_path=repo_path, number=42),
        github_issue_payload(repo_path=repo_path, number=77, title='Graph view misses call relationships'),
    ]
    if include_pull_request:
        payload.append(github_issue_payload(repo_path=repo_path, number=88, title='Draft PR should be filtered', pull_request=True))
    headers = {'Link': github_issue_link_header(repo_path)} if has_next_page else {}
    return MockGithubHttpResponse(payload=payload, headers=headers, url=f'https://api.github.com/repos/{repo_path}/issues')


def mock_github_issue_detail_response(*, repo_path: str = ISSUE_MAP_FIXTURE.repo_path, issue_number: int = 42) -> MockGithubHttpResponse:
    return MockGithubHttpResponse(
        payload=github_issue_payload(repo_path=repo_path, number=issue_number),
        url=f'https://api.github.com/repos/{repo_path}/issues/{issue_number}',
    )


def mock_github_issue_comments_response(*, repo_path: str = ISSUE_MAP_FIXTURE.repo_path, issue_number: int = 42, count: int = 2) -> MockGithubHttpResponse:
    return MockGithubHttpResponse(
        payload=github_issue_comments_payload(count=count),
        url=f'https://api.github.com/repos/{repo_path}/issues/{issue_number}/comments',
    )


ISSUE_MAP_GOLDEN_CASES: tuple[IssueMapGoldenCase, ...] = (
    IssueMapGoldenCase(
        name='stack_trace_build_failure',
        issue=github_issue_payload(
            title='Repository analysis crashes during build',
            body='Traceback: File "api/services.py", line 6, in _build_and_store_analysis. parse_repo() raises a parser timeout error.',
            labels=[github_issue_label('analysis', '1d76db', 'Analysis flow')],
        ),
        comments=(
            {
                'id': 1,
                'author': 'maintainer',
                'body': 'The failing call path goes through parser/services.py::parse_repo.',
            },
        ),
        expected_node_ids=('api/services.py::_build_and_store_analysis', 'parser/services.py::parse_repo'),
        expected_file_paths=('api/services.py', 'parser/services.py'),
        expected_top_node_id='api/services.py::_build_and_store_analysis',
    ),
    IssueMapGoldenCase(
        name='parser_entry_comment',
        issue=github_issue_payload(
            title='Parser returns empty graph',
            body='The graph has no nodes after parser/services.py runs. parse_repo() returns an empty payload.',
            labels=[github_issue_label('parser', '5319e7', 'Parser area')],
        ),
        comments=(
            {
                'id': 2,
                'author': 'reviewer',
                'body': 'parser/services.py:1:in parse_repo is the smallest repro.',
            },
        ),
        expected_node_ids=('parser/services.py::parse_repo',),
        expected_file_paths=('parser/services.py',),
        expected_top_node_id='parser/services.py::parse_repo',
    ),
    IssueMapGoldenCase(
        name='view_to_service_handoff',
        issue=github_issue_payload(
            title='Analysis endpoint does not return graph',
            body='The api/views.py analysis view calls get_repo_analysis(), but the response is empty for users.',
            labels=[github_issue_label('api', '0e8a16', 'API view')],
        ),
        comments=(),
        expected_node_ids=('api/views.py::analysis', 'api/services.py::get_repo_analysis'),
        expected_file_paths=('api/views.py', 'api/services.py'),
        expected_top_node_id='api/views.py::analysis',
    ),
    IssueMapGoldenCase(
        name='bare_filename_mentions',
        issue=github_issue_payload(
            title='services.py issue is hard to trace',
            body='The user only mentioned services.py and get_repo_analysis without a package path.',
            labels=[github_issue_label('analysis', '1d76db', 'Analysis flow')],
        ),
        comments=(
            {
                'id': 3,
                'author': 'newcomer',
                'body': 'I think _build_and_store_analysis is also involved.',
            },
        ),
        expected_node_ids=('api/services.py::get_repo_analysis', 'api/services.py::_build_and_store_analysis'),
        expected_file_paths=('api/services.py',),
    ),
)


ISSUE_MAP_RANKING_BASELINE = {
    'case_count': len(ISSUE_MAP_GOLDEN_CASES),
    'recall_at_1': 0.625,
    'recall_at_3': 1.0,
    'recall_at_5': 1.0,
}


def issue_ranking_case_result(
    case: IssueMapGoldenCase,
    ranked_node_ids: list[str],
    *,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> IssueRankingCaseResult:
    expected = set(case.expected_node_ids)
    hits_at_k = {}
    recall_at_k = {}
    for k in k_values:
        hits = len(set(ranked_node_ids[:k]) & expected)
        hits_at_k[k] = hits
        recall_at_k[k] = round(hits / len(expected), 3) if expected else 1.0
    return IssueRankingCaseResult(
        name=case.name,
        ranked_node_ids=tuple(ranked_node_ids),
        expected_node_ids=case.expected_node_ids,
        expected_file_paths=case.expected_file_paths,
        hits_at_k=hits_at_k,
        recall_at_k=recall_at_k,
    )


def issue_ranking_recall_report(results: tuple[IssueRankingCaseResult, ...], *, k_values: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
    return {
        'case_count': len(results),
        **{
            f'recall_at_{k}': round(
                sum(result.recall_at_k[k] for result in results) / len(results),
                3,
            )
            if results
            else 0.0
            for k in k_values
        },
        'cases': [
            {
                'name': result.name,
                'ranked_node_ids': result.ranked_node_ids,
                'expected_node_ids': result.expected_node_ids,
                'expected_file_paths': result.expected_file_paths,
                'recall_at_k': dict(result.recall_at_k),
            }
            for result in results
        ],
    }


def make_issue_llm_stub(response: Mapping[str, Any] | None = None) -> IssueLlmCallRecorder:
    return IssueLlmCallRecorder(
        response=response
        or {
            'hypotheses': [
                {
                    'kind': 'likely_origin',
                    'node_id': 'api/services.py::_build_and_store_analysis',
                    'confidence': 0.8,
                    'rationale': 'Deterministic issue LLM test response.',
                }
            ],
            'investigation_path': [
                {
                    'step': 1,
                    'node_id': 'api/services.py::_build_and_store_analysis',
                    'path': 'api/services.py',
                    'action': 'inspect',
                    'why': 'Fixture response for tests.',
                }
            ],
            'confidence': {'level': 'medium', 'score': 0.8, 'reasons': ['fixture']},
        }
    )


class ExternalHttpBlockedMixin:
    """Blocks requests-based HTTP calls in tests that must stay offline."""

    _external_http_patcher: Any

    def setUp(self) -> None:
        super().setUp()
        self._external_http_patcher = patch('requests.sessions.Session.request', side_effect=self._blocked_external_http_request)
        self._external_http_patcher.start()

    def tearDown(self) -> None:
        self._external_http_patcher.stop()
        super().tearDown()

    @staticmethod
    def _blocked_external_http_request(*args: Any, **kwargs: Any) -> None:
        method = args[1] if len(args) > 1 else kwargs.get('method', 'GET')
        url = args[2] if len(args) > 2 else kwargs.get('url', '<unknown>')
        raise AssertionError(f'External HTTP request blocked in offline issue test: {method} {url}')


def assert_uses_issue_llm_stub(stub: Callable[..., Mapping[str, Any]]) -> Mapping[str, Any]:
    return stub({'issue': {'number': 42}}, candidates=['api/services.py::_build_and_store_analysis'])


GOLDEN_FIXTURE_REPOS: dict[str, GoldenFixtureRepo] = {
    'plain_python_package': GoldenFixtureRepo(
        name='plain_python_package',
        files={
            'sample_pkg/__init__.py': '',
            'sample_pkg/main.py': (
                'from sample_pkg.utils import normalize_name\n\n'
                'def run():\n'
                '    return normalize_name("Ada")\n'
            ),
            'sample_pkg/utils.py': (
                'def normalize_name(value):\n'
                '    return value.strip().lower()\n'
            ),
        },
        expected_nodes=(
            'sample_pkg/main.py',
            'sample_pkg/main.py::run',
            'sample_pkg/utils.py::normalize_name',
            'module::sample_pkg.utils',
        ),
        expected_edges=(
            ('sample_pkg/main.py', 'module::sample_pkg.utils', 'imports'),
            ('sample_pkg/main.py::run', 'sample_pkg/utils.py::normalize_name', 'calls'),
        ),
        rubric_tags=('graph_node_recall', 'edge_correctness', 'qa_citation_correctness'),
    ),
    'oop_inheritance_sample': GoldenFixtureRepo(
        name='oop_inheritance_sample',
        files={
            'domain/models.py': (
                'class BaseTask:\n'
                '    pass\n\n'
                'class BuildTask(BaseTask):\n'
                '    def run(self):\n'
                '        return self.step()\n\n'
                '    def step(self):\n'
                '        return "built"\n'
            ),
        },
        expected_nodes=(
            'domain/models.py::BaseTask',
            'domain/models.py::BuildTask',
            'domain/models.py::BuildTask::run',
            'domain/models.py::BuildTask::step',
        ),
        expected_edges=(
            ('domain/models.py::BuildTask', 'domain/models.py::BaseTask', 'inherits'),
            ('domain/models.py::BuildTask::run', 'domain/models.py::BuildTask::step', 'calls'),
        ),
        rubric_tags=('graph_node_recall', 'edge_correctness'),
    ),
    'cross_file_import_call_sample': GoldenFixtureRepo(
        name='cross_file_import_call_sample',
        files={
            'app/factory.py': (
                'def load_component():\n'
                '    return "component"\n'
            ),
            'app/runner.py': (
                'from app.factory import load_component\n\n'
                'def main():\n'
                '    return load_component()\n'
            ),
        },
        expected_nodes=(
            'app/factory.py::load_component',
            'app/runner.py::main',
            'module::app.factory',
        ),
        expected_edges=(
            ('app/runner.py', 'module::app.factory', 'imports'),
            ('app/runner.py::main', 'app/factory.py::load_component', 'calls'),
        ),
        rubric_tags=('graph_node_recall', 'edge_correctness', 'entrypoint_correctness', 'qa_citation_correctness'),
    ),
    'django_like_mini_app': GoldenFixtureRepo(
        name='django_like_mini_app',
        files={
            'manage.py': (
                'from webapp.urls import urlpatterns\n\n'
                'def main():\n'
                '    return urlpatterns\n'
            ),
            'webapp/views.py': (
                'def index(request):\n'
                '    return "ok"\n'
            ),
            'webapp/urls.py': (
                'from webapp.views import index\n\n'
                'urlpatterns = [index]\n'
            ),
        },
        expected_nodes=('manage.py::main', 'webapp/views.py::index', 'module::webapp.urls', 'module::webapp.views'),
        expected_edges=(
            ('manage.py', 'module::webapp.urls', 'imports'),
            ('webapp/urls.py', 'module::webapp.views', 'imports'),
        ),
        rubric_tags=('graph_node_recall', 'entrypoint_correctness', 'qa_citation_correctness'),
    ),
    'fastapi_like_mini_app': GoldenFixtureRepo(
        name='fastapi_like_mini_app',
        files={
            'service/api.py': (
                'from service.core import build_payload\n\n'
                '@app.get("/")\n'
                'def read_root():\n'
                '    return build_payload()\n'
            ),
            'service/core.py': (
                'def build_payload():\n'
                '    return {"status": "ok"}\n'
            ),
        },
        expected_nodes=('service/api.py::read_root', 'service/core.py::build_payload', 'module::service.core'),
        expected_edges=(
            ('service/api.py', 'module::service.core', 'imports'),
            ('service/api.py::read_root', 'service/core.py::build_payload', 'calls'),
        ),
        rubric_tags=('graph_node_recall', 'edge_correctness', 'qa_citation_correctness'),
    ),
    'ambiguous_symbol_sample': GoldenFixtureRepo(
        name='ambiguous_symbol_sample',
        files={
            'alpha/tasks.py': (
                'def run():\n'
                '    return "alpha"\n'
            ),
            'beta/tasks.py': (
                'def run():\n'
                '    return "beta"\n'
            ),
            'orchestrator.py': (
                'def dispatch(worker):\n'
                '    return worker.run()\n'
            ),
        },
        expected_nodes=('alpha/tasks.py::run', 'beta/tasks.py::run', 'orchestrator.py::dispatch'),
        expected_edges=(('orchestrator.py::dispatch', 'attribute::run', 'calls'),),
        rubric_tags=('graph_node_recall', 'edge_correctness'),
    ),
    'korean_readme_sample': GoldenFixtureRepo(
        name='korean_readme_sample',
        files={
            'README.md': '# 샘플 저장소\n\n사용자 온보딩 설명입니다.\n',
            'src/사용자.py': (
                'def build_user():\n'
                '    return {"name": "홍길동"}\n'
            ),
        },
        expected_nodes=('src/사용자.py::build_user',),
        expected_edges=(('module::src.사용자', 'src/사용자.py::build_user', 'contains'),),
        rubric_tags=('graph_node_recall', 'qa_citation_correctness'),
    ),
}


def run_git(repo_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ['git', *args],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def init_git_repo(repo_dir: Path, branch: str = 'main') -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    run_git(repo_dir, 'init')
    run_git(repo_dir, 'checkout', '-b', branch)


def write_files(repo_dir: Path, files: Mapping[str, str]) -> None:
    for relative_path, content in files.items():
        file_path = repo_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')


def commit_all(repo_dir: Path, message: str) -> str:
    run_git(repo_dir, 'add', '.')
    run_git(
        repo_dir,
        '-c',
        f'user.name={GIT_USER_NAME}',
        '-c',
        f'user.email={GIT_USER_EMAIL}',
        'commit',
        '-m',
        message,
    )
    return run_git(repo_dir, 'rev-parse', 'HEAD')


def create_git_fixture_repo(
    repo_dir: Path,
    files: Mapping[str, str],
    *,
    branch: str = 'main',
    commit_message: str = 'initial repo',
) -> GitFixtureRepo:
    shutil.rmtree(repo_dir, ignore_errors=True)
    init_git_repo(repo_dir, branch=branch)
    write_files(repo_dir, files)
    revision = commit_all(repo_dir, commit_message)
    return GitFixtureRepo(path=repo_dir, revision=revision, files=dict(files))


def create_named_fixture_repo(name: str, repo_dir: Path) -> GitFixtureRepo:
    fixture = GOLDEN_FIXTURE_REPOS[name]
    return create_git_fixture_repo(repo_dir, fixture.files, commit_message=f'{name} fixture')
