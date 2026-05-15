from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import shutil
import subprocess


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


EVAL_RUBRIC = {
    'graph_node_recall': 'Expected modules, files, classes, and functions from a fixture must appear in the graph.',
    'edge_correctness': 'Expected contains, imports, inherits, and calls edges must point to the intended source and target.',
    'entrypoint_correctness': 'Entrypoint-like symbols should be identifiable by graph metadata or deterministic hints.',
    'qa_citation_correctness': 'Q&A should cite the files that contain the code evidence used for the answer.',
}


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
        expected_edges=(('src/사용자.py', 'src/사용자.py::build_user', 'contains'),),
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
