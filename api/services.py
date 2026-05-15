from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from django.conf import settings
from django.db import IntegrityError, transaction
from django.utils import timezone

from api.artifacts import GRAPH_ARTIFACT_SCHEMA_VERSION, build_graph_artifact, coerce_graph_artifact
from api.models import AnalysisArtifact, AnalysisRun, Repository
from api.serializers import is_safe_revision, _is_safe_repo_segment
from github_repo.services import RepoIngestionError, get_file_content_or_raise, get_repo_snapshot_or_raise as get_repo_snapshot
from parser.services import parse_repo


def _analysis_parts(repo_path: str) -> tuple[str, str]:
    parts = repo_path.split('/')
    if len(parts) != 2:
        raise ValueError('Unsafe repo path')
    owner, repo = parts
    if not _is_safe_repo_segment(owner) or not _is_safe_repo_segment(repo) or repo.endswith('.git'):
        raise ValueError('Unsafe repo path')
    return owner, repo


def _analysis_key(repo_path: str, revision: str) -> str:
    if not is_safe_revision(revision):
        raise ValueError('Unsafe revision')
    owner, repo = _analysis_parts(repo_path)
    return f'{owner}/{repo}@{revision}'


def _analysis_path(repo_path: str, revision: str):
    analysis_dir = settings.TEMP_DIR / 'analysis' / _analysis_key(repo_path, revision)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir / 'graph.json'


def _repo_clone_url(repo_path: str) -> str:
    return f'https://github.com/{repo_path}.git'


def get_or_create_repository(repo_path: str, *, provider: str = 'github', default_branch: str | None = None, clone_url: str | None = None) -> Repository:
    owner, name = _analysis_parts(repo_path)
    repository, _created = Repository.objects.get_or_create(
        provider=provider,
        full_name=repo_path,
        defaults={
            'owner': owner,
            'name': name,
            'default_branch': default_branch,
            'clone_url': clone_url or _repo_clone_url(repo_path),
        },
    )

    update_fields = []
    if repository.owner != owner:
        repository.owner = owner
        update_fields.append('owner')
    if repository.name != name:
        repository.name = name
        update_fields.append('name')
    if default_branch is not None and repository.default_branch != default_branch:
        repository.default_branch = default_branch
        update_fields.append('default_branch')
    if clone_url is not None and repository.clone_url != clone_url:
        repository.clone_url = clone_url
        update_fields.append('clone_url')
    if update_fields:
        repository.save(update_fields=[*update_fields, 'updated_at'])
    return repository


def start_analysis_run(repository: Repository, *, ref: str = 'HEAD', revision: str = '') -> AnalysisRun:
    return AnalysisRun.objects.create(
        repository=repository,
        ref=ref,
        revision=revision,
        status=AnalysisRun.STATUS_STARTED,
    )


def _artifact_counts(payload: dict[str, Any]) -> tuple[int, int, int]:
    return (
        len(payload.get('nodes', [])),
        len(payload.get('edges', [])),
        len(payload.get('warnings', [])),
    )


def store_artifact(analysis_run: AnalysisRun, payload: dict[str, Any]) -> dict[str, Any]:
    node_count, edge_count, warning_count = _artifact_counts(payload)
    with transaction.atomic():
        analysis_run.status = AnalysisRun.STATUS_SUCCEEDED
        analysis_run.finished_at = timezone.now()
        analysis_run.error_code = ''
        analysis_run.error_message = ''
        analysis_run.save(update_fields=['status', 'finished_at', 'error_code', 'error_message'])
        AnalysisArtifact.objects.update_or_create(
            analysis_run=analysis_run,
            defaults={
                'schema_version': payload['schema_version'],
                'payload': payload,
                'node_count': node_count,
                'edge_count': edge_count,
                'warning_count': warning_count,
            },
        )
    return payload


def _error_code_and_message(error: Exception) -> tuple[str, str]:
    if isinstance(error, RepoIngestionError):
        return error.code, error.message
    return error.__class__.__name__, str(error)


def store_failed_run(analysis_run: AnalysisRun, error: Exception) -> AnalysisRun:
    error_code, error_message = _error_code_and_message(error)
    analysis_run.status = AnalysisRun.STATUS_FAILED
    analysis_run.finished_at = timezone.now()
    analysis_run.error_code = error_code
    analysis_run.error_message = error_message
    analysis_run.save(update_fields=['status', 'finished_at', 'error_code', 'error_message'])
    return analysis_run


def get_artifact_by_revision(repo_path: str, revision: str) -> dict[str, Any] | None:
    try:
        _analysis_parts(repo_path)
    except ValueError:
        return None
    if not is_safe_revision(revision):
        return None

    artifact = (
        AnalysisArtifact.objects
        .select_related('analysis_run', 'analysis_run__repository')
        .filter(
            analysis_run__repository__provider='github',
            analysis_run__repository__full_name=repo_path,
            analysis_run__revision=revision,
            analysis_run__status=AnalysisRun.STATUS_SUCCEEDED,
        )
        .order_by('-created_at')
        .first()
    )
    if artifact is None:
        return None
    return artifact.payload


def _persist_succeeded_artifact(repo_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    revision = str(payload['revision'])
    existing_payload = get_artifact_by_revision(repo_path, revision)
    if existing_payload is not None:
        return existing_payload

    repository = get_or_create_repository(
        repo_path,
        provider=str(payload.get('provider') or 'github'),
        default_branch=payload.get('default_branch'),
        clone_url=_repo_clone_url(repo_path),
    )
    analysis_run = start_analysis_run(repository, ref=str(payload.get('ref') or 'HEAD'), revision=revision)
    try:
        return store_artifact(analysis_run, payload)
    except IntegrityError:
        existing_payload = get_artifact_by_revision(repo_path, revision)
        if existing_payload is not None:
            analysis_run.delete()
            return existing_payload
        raise


def _write_analysis_artifact(artifact_path, analysis: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=artifact_path.parent, prefix='graph.', suffix='.tmp', delete=False) as temporary_file:
        temporary_file.write(json.dumps(analysis, ensure_ascii=False, indent=2))
        temporary_path = temporary_file.name
    os.replace(temporary_path, artifact_path)


def _read_analysis_artifact(artifact_path) -> dict[str, Any]:
    payload = json.loads(artifact_path.read_text(encoding='utf-8'))
    analysis = coerce_graph_artifact(payload)
    if payload.get('schema_version') != GRAPH_ARTIFACT_SCHEMA_VERSION:
        _write_analysis_artifact(artifact_path, analysis)
    return analysis


def _max_total_analyzed_bytes() -> int:
    return int(getattr(settings, 'GITHUB_REPO_MAX_TOTAL_ANALYZED_BYTES', 5_000_000))


def get_repo_analysis(repo_path: str, revision: str | None = None) -> dict[str, Any] | None:
    try:
        _analysis_parts(repo_path)
    except ValueError:
        return None

    if revision is not None:
        if not is_safe_revision(revision):
            return None
        artifact = get_artifact_by_revision(repo_path, revision)
        if artifact is not None:
            return artifact
        artifact_path = _analysis_path(repo_path, revision)
        if artifact_path.exists():
            return _persist_succeeded_artifact(repo_path, _read_analysis_artifact(artifact_path))
        return None

    repository = get_or_create_repository(repo_path)
    try:
        snapshot = get_repo_snapshot(repo_path)
    except Exception as error:
        failed_run = start_analysis_run(repository, revision='')
        store_failed_run(failed_run, error)
        raise

    if snapshot is None:
        return None
    revision, files = snapshot
    if not files:
        return None

    artifact = get_artifact_by_revision(repo_path, revision)
    if artifact is not None:
        return artifact

    artifact_path = _analysis_path(repo_path, revision)
    if artifact_path.exists():
        return _persist_succeeded_artifact(repo_path, _read_analysis_artifact(artifact_path))

    analysis_run = start_analysis_run(repository, revision=revision)
    try:
        python_files = [file_path for file_path in files if file_path.endswith('.py')]
        file_contents = {}
        total_analyzed_bytes = 0
        max_total_analyzed_bytes = _max_total_analyzed_bytes()
        for file_path in python_files:
            content = get_file_content_or_raise(repo_path, file_path, revision)
            if content is None:
                continue
            total_analyzed_bytes += len(content.encode('utf-8'))
            if total_analyzed_bytes > max_total_analyzed_bytes:
                raise RepoIngestionError(
                    'too_large',
                    '분석 대상 Python 코드 총량이 허용 한도를 초과했습니다.',
                    metadata={
                        'limit': max_total_analyzed_bytes,
                        'actual': total_analyzed_bytes,
                        'limit_type': 'max_total_analyzed_bytes',
                    },
                )
            file_contents[file_path] = content

        graph = parse_repo(repo_path, files, lambda _repo_path, file_path: file_contents.get(file_path))
        analysis = build_graph_artifact(
            repo_path=repo_path,
            revision=revision,
            graph=graph,
            file_contents=file_contents,
            entrypoints=graph.get('entrypoints', []),
            key_modules=graph.get('key_modules', []),
            warnings=graph.get('warnings', []),
        )
        _write_analysis_artifact(artifact_path, analysis)
        return store_artifact(analysis_run, analysis)
    except Exception as error:
        store_failed_run(analysis_run, error)
        raise
