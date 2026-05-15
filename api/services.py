from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from django.conf import settings
from django.db import IntegrityError, transaction
from django.utils import timezone

from api.artifacts import GRAPH_ARTIFACT_SCHEMA_VERSION, build_graph_artifact, coerce_graph_artifact
from api.diff import GraphDiffInputError, compare_graph_artifacts
from api.models import AnalysisArtifact, AnalysisRun, Repository
from api.serializers import is_safe_revision, _is_safe_repo_segment
from github_repo.services import (
    RepoIngestionError,
    get_file_content_or_raise,
    get_repo_snapshot_at_revision_or_raise as get_repo_snapshot_at_revision,
    get_repo_snapshot_or_raise as get_repo_snapshot,
)
from llm.summaries import (
    SUMMARY_KIND_NODE,
    SUMMARY_KIND_ONBOARDING,
    SUMMARY_KIND_REPO_OVERVIEW,
    SummaryInputError,
    SummaryUnavailable,
    generate_summary,
    summary_cache_key,
)
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


def get_analysis_run_by_revision(repo_path: str, revision: str) -> AnalysisRun | None:
    try:
        _analysis_parts(repo_path)
    except ValueError:
        return None
    if not is_safe_revision(revision):
        return None

    return (
        AnalysisRun.objects
        .select_related('repository')
        .filter(
            repository__provider='github',
            repository__full_name=repo_path,
            revision=revision,
            status=AnalysisRun.STATUS_SUCCEEDED,
        )
        .order_by('-finished_at', '-started_at')
        .first()
    )


def build_analysis_response(payload: dict[str, Any], analysis_run: AnalysisRun | None = None) -> dict[str, Any]:
    return {
        'analysis_id': analysis_run.id if analysis_run is not None else None,
        'repo': payload['repo'],
        'revision': payload['revision'],
        'status': analysis_run.status if analysis_run is not None else payload.get('status', 'succeeded'),
        'artifact': payload,
        'warnings': payload.get('warnings', []),
    }


def build_tree_response(payload: dict[str, Any], analysis_run: AnalysisRun | None = None) -> dict[str, Any]:
    return {
        'analysis_id': analysis_run.id if analysis_run is not None else None,
        'repo': payload['repo'],
        'revision': payload['revision'],
        'tree': payload['tree'],
        'warnings': payload.get('warnings', []),
    }


def build_graph_response(payload: dict[str, Any], analysis_run: AnalysisRun | None = None) -> dict[str, Any]:
    return {
        'analysis_id': analysis_run.id if analysis_run is not None else None,
        'repo': payload['repo'],
        'revision': payload['revision'],
        'nodes': payload['nodes'],
        'edges': payload['edges'],
        'entrypoints': payload.get('entrypoints', []),
        'key_modules': payload.get('key_modules', []),
        'warnings': payload.get('warnings', []),
    }


def get_analysis_response(repo_path: str, revision: str | None = None) -> dict[str, Any] | None:
    payload = get_repo_analysis(repo_path, revision)
    if payload is None:
        return None
    analysis_run = get_analysis_run_by_revision(repo_path, str(payload['revision']))
    return build_analysis_response(payload, analysis_run)


def get_analysis_response_by_id(analysis_id: int) -> dict[str, Any] | None:
    analysis_run = (
        AnalysisRun.objects
        .select_related('repository')
        .filter(id=analysis_id)
        .first()
    )
    if analysis_run is None:
        return None
    if analysis_run.status != AnalysisRun.STATUS_SUCCEEDED:
        return {
            'analysis_id': analysis_run.id,
            'repo': analysis_run.repository.full_name,
            'revision': analysis_run.revision,
            'status': analysis_run.status,
            'artifact': None,
            'warnings': [],
            'error': {
                'code': analysis_run.error_code,
                'message': analysis_run.error_message,
            },
        }
    try:
        artifact = analysis_run.artifact
    except AnalysisArtifact.DoesNotExist:
        return None
    return build_analysis_response(artifact.payload, analysis_run)


def _get_succeeded_artifact_record_by_id(analysis_id: int) -> tuple[AnalysisRun, AnalysisArtifact] | None:
    analysis_run = (
        AnalysisRun.objects
        .select_related('repository')
        .filter(id=analysis_id, status=AnalysisRun.STATUS_SUCCEEDED)
        .first()
    )
    if analysis_run is None:
        return None
    try:
        return analysis_run, analysis_run.artifact
    except AnalysisArtifact.DoesNotExist:
        return None


def _summary_response(analysis_run: AnalysisRun, summary: dict[str, Any], *, cached: bool) -> dict[str, Any]:
    return {
        'analysis_id': analysis_run.id,
        'repo': analysis_run.repository.full_name,
        'revision': analysis_run.revision,
        'summary': summary,
        'cached': cached,
    }


def get_or_create_summary_response(analysis_id: int, kind: str = SUMMARY_KIND_REPO_OVERVIEW) -> dict[str, Any] | None:
    if kind not in {SUMMARY_KIND_REPO_OVERVIEW, SUMMARY_KIND_ONBOARDING}:
        raise SummaryInputError('unsupported summary kind')
    record = _get_succeeded_artifact_record_by_id(analysis_id)
    if record is None:
        return None
    analysis_run, artifact = record
    payload = dict(artifact.payload)
    summaries = dict(payload.get('summaries') or {})
    cache_key = summary_cache_key(kind)
    cached_summary = summaries.get(cache_key)
    if isinstance(cached_summary, dict):
        return _summary_response(analysis_run, cached_summary, cached=True)

    summary = generate_summary(payload, kind)
    summaries[cache_key] = summary
    payload['summaries'] = summaries
    artifact.payload = payload
    artifact.save(update_fields=['payload'])
    return _summary_response(analysis_run, summary, cached=False)


def get_or_create_node_summary_response(analysis_id: int, node_id: str) -> dict[str, Any] | None:
    record = _get_succeeded_artifact_record_by_id(analysis_id)
    if record is None:
        return None
    analysis_run, artifact = record
    payload = dict(artifact.payload)
    summaries = dict(payload.get('summaries') or {})
    cache_key = summary_cache_key(SUMMARY_KIND_NODE, node_id=node_id)
    cached_summary = summaries.get(cache_key)
    if isinstance(cached_summary, dict):
        return _summary_response(analysis_run, cached_summary, cached=True)

    summary = generate_summary(payload, SUMMARY_KIND_NODE, node_id=node_id)
    summaries[cache_key] = summary
    payload['summaries'] = summaries
    artifact.payload = payload
    artifact.save(update_fields=['payload'])
    return _summary_response(analysis_run, summary, cached=False)


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


def _analysis_from_cache(repo_path: str, revision: str) -> dict[str, Any] | None:
    artifact = get_artifact_by_revision(repo_path, revision)
    if artifact is not None:
        return artifact

    artifact_path = _analysis_path(repo_path, revision)
    if artifact_path.exists():
        return _persist_succeeded_artifact(repo_path, _read_analysis_artifact(artifact_path))
    return None


def _build_and_store_analysis(
    repo_path: str,
    repository: Repository,
    revision: str,
    files: list[str],
    *,
    ref: str = 'HEAD',
) -> dict[str, Any]:
    cached_analysis = _analysis_from_cache(repo_path, revision)
    if cached_analysis is not None:
        return cached_analysis

    artifact_path = _analysis_path(repo_path, revision)
    analysis_run = start_analysis_run(repository, ref=ref, revision=revision)
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
            ref=ref,
            entrypoints=graph.get('entrypoints', []),
            key_modules=graph.get('key_modules', []),
            warnings=graph.get('warnings', []),
        )
        _write_analysis_artifact(artifact_path, analysis)
        return store_artifact(analysis_run, analysis)
    except Exception as error:
        store_failed_run(analysis_run, error)
        raise


def get_repo_analysis(repo_path: str, revision: str | None = None) -> dict[str, Any] | None:
    try:
        _analysis_parts(repo_path)
    except ValueError:
        return None

    repository = get_or_create_repository(repo_path)

    if revision is not None:
        if not is_safe_revision(revision):
            return None
        cached_analysis = _analysis_from_cache(repo_path, revision)
        if cached_analysis is not None:
            return cached_analysis

        try:
            snapshot = get_repo_snapshot_at_revision(repo_path, revision)
        except Exception as error:
            failed_run = start_analysis_run(repository, ref=revision, revision=revision)
            store_failed_run(failed_run, error)
            raise

        if snapshot is None:
            return None
        target_revision, files = snapshot
        if not files:
            return None
        return _build_and_store_analysis(repo_path, repository, target_revision, files, ref=revision)

    try:
        snapshot = get_repo_snapshot(repo_path)
    except Exception as error:
        failed_run = start_analysis_run(repository, revision='')
        store_failed_run(failed_run, error)
        raise

    if snapshot is None:
        return None
    target_revision, files = snapshot
    if not files:
        return None

    return _build_and_store_analysis(repo_path, repository, target_revision, files)


def _analysis_ref(run: AnalysisRun) -> dict[str, Any]:
    return {
        'analysis_id': run.id,
        'revision': run.revision,
        'ref': run.ref,
    }


def _build_diff_response(base_run: AnalysisRun, head_run: AnalysisRun) -> dict[str, Any]:
    base_artifact = base_run.artifact.payload
    head_artifact = head_run.artifact.payload
    diff = compare_graph_artifacts(base_artifact, head_artifact)
    return {
        'repo': head_run.repository.full_name,
        'base': _analysis_ref(base_run),
        'head': _analysis_ref(head_run),
        'diff': diff,
        'warnings': diff.get('warnings', []),
    }


def get_diff_response(repo_path: str, base_revision: str, head_revision: str | None = None) -> dict[str, Any] | None:
    try:
        _analysis_parts(repo_path)
    except ValueError:
        return None
    if not is_safe_revision(base_revision):
        return None
    if head_revision is not None and not is_safe_revision(head_revision):
        return None

    base_payload = get_repo_analysis(repo_path, base_revision)
    head_payload = get_repo_analysis(repo_path, head_revision) if head_revision is not None else get_repo_analysis(repo_path)
    if base_payload is None or head_payload is None:
        return None

    base_run = get_analysis_run_by_revision(repo_path, str(base_payload['revision']))
    head_run = get_analysis_run_by_revision(repo_path, str(head_payload['revision']))
    if base_run is None or head_run is None:
        return None
    return _build_diff_response(base_run, head_run)


def get_diff_response_by_analysis_id(head_analysis_id: int, base_analysis_id: int) -> dict[str, Any] | None:
    head_record = _get_succeeded_artifact_record_by_id(head_analysis_id)
    base_record = _get_succeeded_artifact_record_by_id(base_analysis_id)
    if head_record is None or base_record is None:
        return None

    base_run, _base_artifact = base_record
    head_run, _head_artifact = head_record
    if base_run.repository_id != head_run.repository_id:
        raise GraphDiffInputError('diff analyses must belong to the same repo')
    return _build_diff_response(base_run, head_run)
