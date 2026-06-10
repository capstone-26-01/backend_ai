from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from typing import Any

from django.conf import settings
from django.db import IntegrityError, transaction
from django.utils import timezone

from api.artifacts import GRAPH_ARTIFACT_SCHEMA_VERSION, build_graph_artifact, coerce_graph_artifact
from api.diff import GraphDiffInputError, compare_graph_artifacts
from api.issue_map import (
    build_code_context,
    build_focus_graph_projection,
    build_overview_graph_projection,
    extract_issue_evidence,
    rank_issue_candidates,
    sanitize_issue_explanation_output,
)
from api.models import AnalysisArtifact, AnalysisRun, Repository, ShareLink
from api.serializers import is_safe_revision, _is_safe_repo_segment
from github_repo.services import (
    GithubIssueApiError,
    RepoIngestionError,
    get_file_content_or_raise,
    get_github_issue_list_response,
    get_github_issue_comments_response,
    get_github_issue_detail_response,
    get_repo_snapshot_at_revision_or_raise as get_repo_snapshot_at_revision,
    get_repo_snapshot_or_raise as get_repo_snapshot,
)
from llm.context_selection import rank_nodes
from llm.issue_harness import (
    IssueHarnessUnavailable,
    build_issue_harness_job,
    command_from_string,
    run_issue_harness,
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


class ShareInputError(ValueError):
    pass


class IssueMapResponseError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int, metadata: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.metadata = metadata or {}

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {'code': self.code, 'message': self.message}
        if self.metadata:
            payload['metadata'] = self.metadata
        return payload


MOCK_ISSUE_TEMPLATES: list[dict[str, Any]] = [
    {
        'number': 42,
        'title': 'Repository analysis fails on large Python projects',
        'author': 'octocat',
        'assignees': ['hubot'],
        'labels': [
            {'name': 'bug', 'color': 'd73a4a', 'description': 'Something is not working'},
            {'name': 'analysis', 'color': '1d76db', 'description': 'Repository analysis flow'},
        ],
        'comments_count': 3,
        'created_at': '2026-05-20T10:00:00Z',
        'updated_at': '2026-05-23T12:30:00Z',
        'body_excerpt': 'Repository analysis fails when the project has many Python files or the parser exceeds configured limits.',
        'body_truncated': True,
        'locked': False,
        'search_text': 'repository analysis parse parser python files limits timeout ingestion graph artifact',
    },
    {
        'number': 77,
        'title': 'Graph view misses function call relationships',
        'author': 'graph-reviewer',
        'assignees': [],
        'labels': [
            {'name': 'graph', 'color': '5319e7', 'description': 'Code graph and relationship rendering'},
            {'name': 'enhancement', 'color': 'a2eeef', 'description': 'New feature or request'},
        ],
        'comments_count': 5,
        'created_at': '2026-05-18T08:15:00Z',
        'updated_at': '2026-05-22T16:45:00Z',
        'body_excerpt': 'The graph should expose function calls and imports clearly enough for the frontend to highlight related code.',
        'body_truncated': False,
        'locked': False,
        'search_text': 'graph nodes edges function calls imports relationships parser services highlight',
    },
    {
        'number': 103,
        'title': 'Swagger docs are unclear for frontend integration',
        'author': 'frontend-dev',
        'assignees': ['api-maintainer'],
        'labels': [
            {'name': 'documentation', 'color': '0075ca', 'description': 'Improvements or additions to documentation'},
            {'name': 'frontend', 'color': 'fbca04', 'description': 'Frontend integration support'},
        ],
        'comments_count': 2,
        'created_at': '2026-05-17T13:20:00Z',
        'updated_at': '2026-05-21T09:10:00Z',
        'body_excerpt': 'Frontend developers need concise request and response examples for analysis, graph, summary, and issue-related APIs.',
        'body_truncated': False,
        'locked': False,
        'search_text': 'swagger docs api views serializers frontend schema request response examples',
    },
    {
        'number': 128,
        'title': 'QA should focus on the selected graph node',
        'author': 'qa-user',
        'assignees': ['llm-owner', 'backend-owner'],
        'labels': [
            {'name': 'qa', 'color': '0e8a16', 'description': 'Question answering behavior'},
            {'name': 'llm', 'color': 'bfdadc', 'description': 'LLM-backed workflow'},
        ],
        'comments_count': 4,
        'created_at': '2026-05-16T11:40:00Z',
        'updated_at': '2026-05-22T18:05:00Z',
        'body_excerpt': 'When the user selects a node in the graph, QA should prioritize that node and nearby code context.',
        'body_truncated': False,
        'locked': False,
        'search_text': 'qa selected graph node context files llm answer question neighbors',
    },
    {
        'number': 156,
        'title': 'Empty state should work when an issue has no labels',
        'author': 'minimal-reporter',
        'assignees': [],
        'labels': [],
        'comments_count': 0,
        'created_at': '2026-05-14T07:30:00Z',
        'updated_at': '2026-05-14T07:30:00Z',
        'body_excerpt': '',
        'body_truncated': False,
        'locked': False,
        'search_text': 'empty state labels issue list frontend card',
    },
    {
        'number': 164,
        'title': 'Long issue title should wrap without breaking the issue picker layout on narrow mobile screens',
        'author': 'mobile-tester',
        'assignees': ['frontend-dev'],
        'labels': [
            {'name': 'ui', 'color': 'c5def5', 'description': None},
        ],
        'comments_count': 11,
        'created_at': '2026-05-12T22:10:00Z',
        'updated_at': '2026-05-23T21:45:00Z',
        'body_excerpt': 'A deliberately longer preview helps frontend developers verify wrapping, truncation, and spacing in the issue selection UI.',
        'body_truncated': True,
        'locked': False,
        'search_text': 'frontend mobile layout issue picker long title wrap truncation',
    },
    {
        'number': 181,
        'title': 'Deleted author issue should not crash rendering',
        'author': None,
        'assignees': [],
        'labels': [
            {'name': 'edge-case', 'color': 'ededed', 'description': 'Mock data for nullable GitHub fields'},
        ],
        'comments_count': 1,
        'created_at': '2026-05-10T03:05:00Z',
        'updated_at': '2026-05-15T19:25:00Z',
        'body_excerpt': 'GitHub can return nullable user-like data in some historical or deleted-user cases.',
        'body_truncated': False,
        'locked': False,
        'search_text': 'nullable author deleted user issue rendering edge case',
    },
    {
        'number': 209,
        'title': 'Locked conversation still needs related node suggestions',
        'author': 'security-reviewer',
        'assignees': ['backend-owner'],
        'labels': [
            {'name': 'security', 'color': 'ee0701', 'description': 'Security-sensitive behavior'},
            {'name': 'backend', 'color': '0052cc', 'description': 'Backend implementation'},
        ],
        'comments_count': 8,
        'created_at': '2026-05-08T14:00:00Z',
        'updated_at': '2026-05-24T06:55:00Z',
        'body_excerpt': 'Locked issues should remain selectable, but the frontend may show a lock badge while still requesting related nodes.',
        'body_truncated': False,
        'locked': True,
        'search_text': 'locked security backend issue related nodes permissions validation',
    },
]

PREFERRED_RELATED_NODE_KINDS = {'function', 'method', 'class', 'module'}


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


def _get_issue_map_artifact_record(analysis_id: int) -> tuple[AnalysisRun, dict[str, Any]]:
    analysis_run = (
        AnalysisRun.objects
        .select_related('repository')
        .filter(id=analysis_id)
        .first()
    )
    if analysis_run is None:
        raise IssueMapResponseError('analysis_not_found', '분석 결과를 찾을 수 없습니다.', status_code=404, metadata={'analysis_id': analysis_id})
    if analysis_run.status == AnalysisRun.STATUS_STARTED:
        raise IssueMapResponseError('analysis_not_ready', '분석이 아직 완료되지 않았습니다.', status_code=409, metadata={'analysis_id': analysis_id, 'status': analysis_run.status})
    if analysis_run.status == AnalysisRun.STATUS_FAILED:
        raise IssueMapResponseError(
            'analysis_failed',
            '분석이 실패해 issue map을 만들 수 없습니다.',
            status_code=409,
            metadata={'analysis_id': analysis_id, 'status': analysis_run.status, 'error_code': analysis_run.error_code},
        )
    try:
        artifact = analysis_run.artifact
    except AnalysisArtifact.DoesNotExist as exc:
        raise IssueMapResponseError('analysis_artifact_missing', '분석 artifact를 찾을 수 없습니다.', status_code=500, metadata={'analysis_id': analysis_id}) from exc
    payload = artifact.payload
    if not isinstance(payload, dict) or not isinstance(payload.get('nodes'), list) or not isinstance(payload.get('edges'), list):
        raise IssueMapResponseError('analysis_artifact_invalid', '분석 artifact 형식이 올바르지 않습니다.', status_code=500, metadata={'analysis_id': analysis_id})
    return analysis_run, payload


def _mock_github_user(login: str) -> dict[str, str]:
    return {
        'login': login,
        'avatar_url': f'https://github.com/{login}.png',
        'html_url': f'https://github.com/{login}',
    }


def _mock_github_user_or_none(login: str | None) -> dict[str, str] | None:
    if login is None:
        return None
    return _mock_github_user(login)


def _issue_key(repo_path: str, issue_number: int) -> str:
    return f'github:{repo_path}#{issue_number}'


def _mock_issue_payload(repo_path: str, template: dict[str, Any]) -> dict[str, Any]:
    issue_number = int(template['number'])
    return {
        'key': _issue_key(repo_path, issue_number),
        'number': issue_number,
        'title': template['title'],
        'state': 'open',
        'html_url': f'https://github.com/{repo_path}/issues/{issue_number}',
        'author': _mock_github_user_or_none(template.get('author')),
        'labels': template['labels'],
        'assignees': [_mock_github_user(str(login)) for login in template.get('assignees', [])],
        'comments_count': template['comments_count'],
        'created_at': template['created_at'],
        'updated_at': template['updated_at'],
        'body_excerpt': template['body_excerpt'],
        'body_truncated': template['body_truncated'],
        'locked': template['locked'],
        'is_pull_request': False,
    }


def _mock_issue_template(issue_number: int) -> dict[str, Any] | None:
    for template in MOCK_ISSUE_TEMPLATES:
        if int(template['number']) == issue_number:
            return template
    return None


def get_mock_issue_list_response(repo_path: str, *, page: int = 1, per_page: int = 30) -> dict[str, Any]:
    page = max(1, page)
    per_page = max(1, min(per_page, 100))
    start = (page - 1) * per_page
    end = start + per_page
    issues = [_mock_issue_payload(repo_path, template) for template in MOCK_ISSUE_TEMPLATES]
    paged_issues = issues[start:end]
    has_next_page = end < len(issues)
    return {
        'repo': repo_path,
        'provider': 'github',
        'source': 'mock',
        'mock': True,
        'state': 'open',
        'page': page,
        'per_page': per_page,
        'has_next_page': has_next_page,
        'next_page': page + 1 if has_next_page else None,
        'issues': paged_issues,
    }


def get_live_issue_list_response(repo_path: str, *, page: int = 1, per_page: int = 30, state: str = 'open') -> dict[str, Any]:
    try:
        return get_github_issue_list_response(repo_path, page=page, per_page=per_page, state=state)
    except ValueError as exc:
        raise GithubIssueApiError(
            'invalid_repo_path',
            '올바른 repo 경로가 아닙니다.',
            status_code=400,
            metadata={'repo': repo_path},
        ) from exc


def _node_display_payload(node: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': str(node.get('id', '')),
        'kind': str(node.get('kind') or node.get('type') or ''),
        'label': str(node.get('label') or node.get('symbol') or node.get('id') or ''),
        'path': node.get('path') or node.get('file'),
        'start_line': node.get('start_line'),
        'end_line': node.get('end_line'),
        'metadata': dict(node.get('metadata') or {}),
    }


def _node_kind(node: dict[str, Any]) -> str:
    return str(node.get('kind') or node.get('type') or '')


def _prioritize_related_node_ids(ranked_node_ids: list[str], nodes_by_id: dict[str, dict[str, Any]], *, max_nodes: int) -> list[str]:
    preferred_nodes = []
    fallback_nodes = []
    seen = set()
    for original_rank, node_id in enumerate(ranked_node_ids):
        if node_id in seen:
            continue
        seen.add(node_id)
        node = nodes_by_id.get(node_id)
        if node is None:
            continue
        node_kind = _node_kind(node)
        item = (original_rank, node_id)
        if node_kind in PREFERRED_RELATED_NODE_KINDS:
            preferred_nodes.append(item)
        else:
            fallback_nodes.append(item)

    ordered_nodes = [node_id for _rank, node_id in preferred_nodes]
    ordered_nodes.extend(node_id for _rank, node_id in fallback_nodes)
    return ordered_nodes[:max_nodes]


def _fallback_ranked_node_ids(analysis: dict[str, Any], *, max_nodes: int) -> list[str]:
    nodes = [
        node
        for node in analysis.get('nodes', [])
        if isinstance(node, dict) and node.get('id')
    ]
    nodes.sort(key=lambda node: (str(node.get('path') or node.get('file') or ''), str(node.get('id'))))
    return [str(node['id']) for node in nodes[:max_nodes]]


def get_mock_issue_related_nodes_response(analysis_id: int, issue_number: int, *, max_nodes: int = 8) -> dict[str, Any] | None:
    record = _get_succeeded_artifact_record_by_id(analysis_id)
    if record is None:
        return None

    issue_template = _mock_issue_template(issue_number)
    if issue_template is None:
        return None

    analysis_run, artifact = record
    analysis = dict(artifact.payload)
    repo_path = analysis_run.repository.full_name
    issue = _mock_issue_payload(repo_path, issue_template)
    issue_query = f'{issue["title"]} {issue["body_excerpt"]} {issue_template["search_text"]}'
    max_nodes = max(1, min(max_nodes, 20))
    ranking_pool_size = max(max_nodes * 6, 50)
    ranked_node_ids, warnings = rank_nodes(analysis, issue_query, max_nodes=ranking_pool_size)
    nodes_by_id = {
        str(node.get('id')): dict(node)
        for node in analysis.get('nodes', [])
        if isinstance(node, dict) and node.get('id')
    }
    ranked_node_ids = _prioritize_related_node_ids(ranked_node_ids, nodes_by_id, max_nodes=max_nodes)

    if not ranked_node_ids:
        ranked_node_ids = _fallback_ranked_node_ids(analysis, max_nodes=max_nodes)
        warnings.append({'code': 'mock_related_nodes_fallback', 'message': 'Issue text와 일치하는 graph node가 없어 앞쪽 graph node를 mock으로 반환했습니다.'})

    candidates = []
    for index, node_id in enumerate(ranked_node_ids[:max_nodes], start=1):
        node = nodes_by_id.get(node_id)
        if node is None:
            continue
        display_node = _node_display_payload(node)
        evidence = [
            {
                'type': 'mock',
                'message': '프런트엔드 graph highlight 연동을 위한 임시 추천입니다.',
            }
        ]
        if display_node['path']:
            evidence.append(
                {
                    'type': 'graph_metadata',
                    'message': f'Graph node path: {display_node["path"]}',
                }
            )
        if display_node['kind'] in PREFERRED_RELATED_NODE_KINDS:
            evidence.append(
                {
                    'type': 'node_kind_priority',
                    'message': f'{display_node["kind"]} node를 file node보다 우선 추천했습니다.',
                }
            )
        candidates.append(
            {
                'rank': index,
                'score': round(max(0.1, 1.0 - ((index - 1) * 0.08)), 2),
                'node_id': node_id,
                'node': display_node,
                'reason': 'Mock candidate based on issue title/body tokens and graph node metadata. 실제 구현에서는 GitHub issue 본문/comment와 deterministic graph ranking 및 bounded issue harness investigation을 사용합니다.',
                'evidence': evidence,
            }
        )
    hypotheses = _issue_hypotheses(candidates)
    investigation_path = _issue_investigation_path(candidates)
    confidence = _issue_confidence(candidates, warnings)
    guide = build_issue_navigation_guide(
        candidates=candidates,
        hypotheses=hypotheses,
        investigation_path=investigation_path,
        confidence=confidence,
        warnings=warnings,
    )

    return {
        'analysis_id': analysis_run.id,
        'repo': repo_path,
        'revision': analysis_run.revision,
        'provider': 'github',
        'source': 'mock',
        'mock': True,
        'issue': {
            'key': issue['key'],
            'number': issue['number'],
            'title': issue['title'],
            'state': issue['state'],
            'html_url': issue['html_url'],
            'labels': issue['labels'],
            'comments_count': issue['comments_count'],
            'updated_at': issue['updated_at'],
            'body_excerpt': issue['body_excerpt'],
        },
        'selected_node_ids': [candidate['node_id'] for candidate in candidates],
        'candidates': candidates,
        **guide,
        'limits': {'max_nodes': max_nodes},
        'warnings': warnings,
    }


def _issue_ref_payload(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        'key': issue['key'],
        'number': issue['number'],
        'title': issue['title'],
        'state': issue['state'],
        'html_url': issue['html_url'],
        'labels': issue['labels'],
        'comments_count': issue['comments_count'],
        'updated_at': issue['updated_at'],
        'body_excerpt': issue['body_excerpt'],
    }


def _issue_hypotheses(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'kind': 'likely_origin' if index == 0 else 'related_area',
            'node_id': candidate['node_id'],
            'confidence': candidate['score'],
            'rationale': candidate['reason'],
        }
        for index, candidate in enumerate(candidates[:5])
    ]


def _issue_investigation_path(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'step': index,
            'node_id': candidate['node_id'],
            'path': candidate['node'].get('path'),
            'action': 'inspect',
            'why': candidate['reason'],
        }
        for index, candidate in enumerate(candidates[:8], start=1)
    ]


def _issue_confidence(candidates: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> dict[str, Any]:
    score = float(candidates[0]['score']) if candidates else 0.0
    warning_codes = {str(warning.get('code')) for warning in warnings}
    if not candidates or 'no_ranked_issue_nodes' in warning_codes:
        level = 'low'
    elif score >= 0.75 and 'low_confidence_issue_ranking' not in warning_codes:
        level = 'high'
    elif score >= 0.35:
        level = 'medium'
    else:
        level = 'low'
    return {
        'level': level,
        'score': round(score, 3),
        'reasons': [candidate['reason'] for candidate in candidates[:3]],
        'warning_codes': sorted(code for code in warning_codes if code),
    }


STRONG_ISSUE_EVIDENCE_TYPES = {
    'stack_frame',
    'file_path',
    'file_symbol',
    'file_line',
    'symbol',
    'route',
    'config',
    'exception_message',
    'exception_class',
    'quoted_string',
    'test',
    'harness',
}
WEAK_ISSUE_EVIDENCE_TYPES = {'fallback', 'label', 'lexical', 'test_penalty'}
LOW_CONFIDENCE_WARNING_CODES = {'no_ranked_issue_nodes', 'low_confidence_issue_ranking', 'weak_issue_evidence'}


def issue_candidates_are_low_confidence(candidates: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> bool:
    warning_codes = {str(warning.get('code')) for warning in warnings if isinstance(warning, dict)}
    if warning_codes & LOW_CONFIDENCE_WARNING_CODES:
        return True
    if not candidates:
        return True
    top = candidates[0]
    try:
        raw_score = float(top.get('raw_score'))
    except (TypeError, ValueError):
        raw_score = float(top.get('score') or 0.0) * 100
    evidence_types = {
        str(item.get('type'))
        for item in top.get('evidence') or []
        if isinstance(item, dict) and item.get('type')
    }
    if raw_score < 40 and 'harness' not in evidence_types:
        return True
    if evidence_types and evidence_types.issubset(WEAK_ISSUE_EVIDENCE_TYPES):
        return True
    if evidence_types and not (evidence_types & STRONG_ISSUE_EVIDENCE_TYPES):
        return True
    top_kinds = [
        str((candidate.get('node') or {}).get('kind') or (candidate.get('node') or {}).get('type') or '')
        for candidate in candidates[:3]
        if isinstance(candidate.get('node'), dict)
    ]
    return bool(top_kinds) and all(kind in {'directory', 'file'} for kind in top_kinds)


def _low_confidence_confidence(confidence: dict[str, Any], warnings: list[dict[str, Any]]) -> dict[str, Any]:
    warning_codes = {str(warning.get('code')) for warning in warnings if isinstance(warning, dict) and warning.get('code')}
    reasons = [str(reason) for reason in confidence.get('reasons') or [] if reason]
    reasons.insert(0, 'Issue evidence is too weak to identify an exact starting node.')
    return {
        **confidence,
        'level': 'low',
        'score': round(min(float(confidence.get('score') or 0.0), 0.34), 3),
        'reasons': reasons[:5],
        'warning_codes': sorted({*warning_codes, *[str(code) for code in confidence.get('warning_codes') or [] if code]}),
    }


def _issue_search_terms(evidence: dict[str, Any] | None) -> list[str]:
    if not isinstance(evidence, dict):
        return []
    raw_parts = [
        str(evidence.get('query') or ''),
        ' '.join(str(item.get('symbol') or '') for item in evidence.get('symbol_mentions') or [] if isinstance(item, dict)),
        ' '.join(str(item.get('path') or '') for item in evidence.get('file_mentions') or [] if isinstance(item, dict)),
        ' '.join(str(item.get('route') or '') for item in evidence.get('route_mentions') or [] if isinstance(item, dict)),
        ' '.join(str(item.get('name') or '') for item in evidence.get('config_mentions') or [] if isinstance(item, dict)),
        ' '.join(str(item.get('class') or '') for item in evidence.get('exception_mentions') or [] if isinstance(item, dict)),
        ' '.join(str(item.get('text') or '') for item in evidence.get('quoted_strings') or [] if isinstance(item, dict)),
    ]
    terms: list[str] = []
    generic = {'issue', 'error', 'failed', 'failure', 'bug', 'please', 'investigate', 'github', 'python'}
    for token in re.findall(r'[A-Za-z_][A-Za-z0-9_./:-]{2,}|[가-힣]{2,}', ' '.join(raw_parts)):
        normalized = token.strip('.,:;()[]{}').lower()
        if len(normalized) < 3 or normalized in generic or normalized not in terms:
            if len(normalized) >= 3 and normalized not in generic:
                terms.append(normalized)
        if len(terms) >= 8:
            break
    return terms


def _guide_candidate_by_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate.get('node_id')): candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get('node_id')
    }


def _guide_node_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    node = candidate.get('node')
    return dict(node) if isinstance(node, dict) else {}


def _guide_entry(
    candidate: dict[str, Any],
    *,
    why: Any = None,
    confidence: Any = None,
    action: str | None = None,
    path: Any = None,
) -> dict[str, Any]:
    node = _guide_node_from_candidate(candidate)
    node_id = str(candidate.get('node_id') or node.get('id') or '')
    entry: dict[str, Any] = {
        'node_id': node_id,
        'path': path or node.get('path'),
        'start_line': node.get('start_line'),
        'end_line': node.get('end_line'),
        'label': node.get('label') or node_id,
        'kind': node.get('kind') or node.get('type') or '',
        'why': str(why or candidate.get('reason') or 'Issue evidence points to this node.'),
    }
    if action:
        entry['action'] = action
    if confidence is not None:
        try:
            entry['confidence'] = round(max(0.0, min(1.0, float(confidence))), 3)
        except (TypeError, ValueError):
            entry['confidence'] = round(max(0.0, min(1.0, float(candidate.get('score') or 0.0))), 3)
    return entry


def _guide_start_candidate(
    candidates: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    investigation_path: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    candidates_by_id = _guide_candidate_by_id(candidates)
    for step in investigation_path:
        if not isinstance(step, dict) or not step.get('path'):
            continue
        candidate = candidates_by_id.get(str(step.get('node_id')))
        if candidate:
            return candidate, step
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            continue
        candidate = candidates_by_id.get(str(hypothesis.get('node_id')))
        if candidate:
            return candidate, hypothesis
    for candidate in candidates:
        node = _guide_node_from_candidate(candidate)
        if str(node.get('kind') or node.get('type') or '') in {'function', 'method', 'class', 'module'}:
            return candidate, None
    for candidate in candidates:
        node = _guide_node_from_candidate(candidate)
        if str(node.get('kind') or node.get('type') or '') == 'file':
            return candidate, None
    return None, None


def build_issue_navigation_guide(
    *,
    candidates: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    investigation_path: list[dict[str, Any]],
    confidence: dict[str, Any],
    warnings: list[dict[str, Any]],
    evidence: dict[str, Any] | None = None,
    low_confidence: bool = False,
) -> dict[str, Any]:
    warning_codes = {str(warning.get('code')) for warning in warnings if isinstance(warning, dict)}
    start_candidate, start_source = _guide_start_candidate(candidates, hypotheses, investigation_path)
    mode = 'harness' if confidence.get('source') == 'harness' else 'deterministic'
    if low_confidence or start_candidate is None:
        search_terms = _issue_search_terms(evidence)
        exploratory_steps: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        term_text = ', '.join(search_terms[:3]) if search_terms else 'the reproduced error'
        for candidate in candidates:
            if len(exploratory_steps) >= 3:
                break
            node = _guide_node_from_candidate(candidate)
            path = node.get('path')
            if not path or path in seen_paths:
                continue
            seen_paths.add(str(path))
            exploratory_steps.append(
                {
                    'node_id': str(candidate.get('node_id') or node.get('id') or ''),
                    'path': path,
                    'start_line': node.get('start_line'),
                    'end_line': node.get('end_line'),
                    'label': node.get('label') or candidate.get('node_id'),
                    'kind': node.get('kind') or node.get('type') or '',
                    'action': 'search',
                    'why': f'Issue evidence is weak; reproduce the issue and search this file for {term_text}.',
                }
            )
        message = 'No exact origin was found. Reproduce the issue and search these terms first.'
        return {
            'start_here': None,
            'next_steps': exploratory_steps,
            'avoid': [],
            'guidance_summary': {
                'mode': 'low_confidence',
                'message': message,
                'search_terms': search_terms,
                'warning_codes': sorted(code for code in warning_codes if code),
                'source_mode': mode,
            },
        }

    start_node_id = str(start_candidate.get('node_id'))
    start_why = start_source.get('why') if isinstance(start_source, dict) else None
    start_confidence = start_source.get('confidence') if isinstance(start_source, dict) else None
    start_here = _guide_entry(
        start_candidate,
        why=start_why or (start_source.get('rationale') if isinstance(start_source, dict) else None),
        confidence=start_confidence if start_confidence is not None else (confidence.get('score') if 'score' in confidence else start_candidate.get('score')),
        path=start_source.get('path') if isinstance(start_source, dict) else None,
    )
    start_path = start_here.get('path')
    candidates_by_id = _guide_candidate_by_id(candidates)
    next_steps: list[dict[str, Any]] = []
    seen = {start_node_id}

    def add_step(candidate: dict[str, Any], *, source: dict[str, Any] | None = None) -> None:
        if len(next_steps) >= 3:
            return
        node_id = str(candidate.get('node_id') or '')
        if not node_id or node_id in seen:
            return
        seen.add(node_id)
        action = str((source or {}).get('action') or 'inspect')
        why = (source or {}).get('why') or candidate.get('reason')
        next_steps.append(_guide_entry(candidate, why=why, action=action, path=(source or {}).get('path')))

    for step in investigation_path:
        if not isinstance(step, dict):
            continue
        candidate = candidates_by_id.get(str(step.get('node_id')))
        if candidate:
            add_step(candidate, source=step)
    for candidate in candidates:
        node = _guide_node_from_candidate(candidate)
        if start_path and node.get('path') == start_path:
            continue
        add_step(candidate)
    for candidate in candidates:
        add_step(candidate)

    summary_target = start_here['node_id']
    if next_steps:
        message = f'Start with {summary_target}, then inspect {next_steps[0]["node_id"]}.'
    else:
        message = f'Start with {summary_target}.'
    return {
        'start_here': start_here,
        'next_steps': next_steps,
        'avoid': [],
        'guidance_summary': {
            'mode': mode,
            'message': message,
            'warning_codes': sorted(code for code in warning_codes if code),
        },
    }


def _issue_harness_enabled() -> bool:
    return bool(getattr(settings, 'ISSUE_HARNESS_ENABLED', False))


def _issue_harness_command() -> list[str] | None:
    command = str(getattr(settings, 'ISSUE_HARNESS_COMMAND', '') or '').strip()
    if not command:
        return None
    return command_from_string(command)


def _issue_harness_warning(error: IssueHarnessUnavailable) -> dict[str, Any]:
    return {
        'code': error.code,
        'message': error.message or 'Issue harness를 사용할 수 없어 deterministic issue ranking을 반환했습니다.',
    }


def _issue_harness_node_ids(output: dict[str, Any]) -> list[str]:
    node_ids: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value and value not in node_ids:
            node_ids.append(value)

    for hypothesis in output.get('hypotheses') or []:
        if isinstance(hypothesis, dict):
            add(hypothesis.get('node_id'))
    for step in output.get('investigation_path') or []:
        if isinstance(step, dict):
            add(step.get('node_id'))
    return node_ids


def _issue_harness_candidates(
    analysis: dict[str, Any],
    harness_output: dict[str, Any],
    seed_candidates: list[dict[str, Any]],
    *,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes_by_id = {
        str(node.get('id')): dict(node)
        for node in analysis.get('nodes', [])
        if isinstance(node, dict) and node.get('id')
    }
    seed_by_id = {str(candidate.get('node_id')): candidate for candidate in seed_candidates}
    warnings: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for rank, node_id in enumerate(_issue_harness_node_ids(harness_output)[:max_candidates], start=1):
        node = nodes_by_id.get(node_id)
        if node is None:
            warnings.append({'code': 'harness_node_not_in_graph', 'message': 'Issue harness returned a node_id that is not in the analysis graph.', 'node_id': node_id})
            continue
        seed = seed_by_id.get(node_id, {})
        score = max(0.05, min(1.0, 1.0 - ((rank - 1) * 0.08)))
        candidates.append(
            {
                'rank': rank,
                'score': round(score, 3),
                'raw_score': seed.get('raw_score', round(score * 100, 3)),
                'node_id': node_id,
                'node': _node_display_payload(node),
                'reason': 'Issue harness inspected bounded repo tools and selected this node.',
                'evidence': [
                    {'type': 'harness', 'message': 'Selected by bounded issue investigation harness.'},
                    *list(seed.get('evidence') or [])[:3],
                ],
            }
        )
    return candidates, warnings


def _issue_harness_summary(harness_result) -> dict[str, Any]:
    return {
        'enabled': True,
        'source': 'pi_harness',
        'tool_calls': harness_result.tool_calls,
        'metadata': harness_result.metadata,
    }


def _comments_unavailable_warning(error: GithubIssueApiError) -> dict[str, Any]:
    return {
        'code': 'github_comments_unavailable',
        'message': 'GitHub issue comment를 읽지 못해 issue 본문만으로 후보를 계산했습니다.',
        'detail': error.as_dict(),
    }


def get_issue_map_response(
    analysis_id: int,
    issue_number: int,
    *,
    max_nodes: int = 8,
    include_comments: bool = True,
    max_context_files: int = 4,
) -> dict[str, Any]:
    analysis_run, analysis = _get_issue_map_artifact_record(analysis_id)
    repo_path = analysis_run.repository.full_name
    max_nodes = max(1, min(max_nodes, 20))
    max_context_files = max(1, min(max_context_files, 10))

    issue = get_github_issue_detail_response(repo_path, issue_number)
    comments: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if include_comments:
        try:
            comments, comment_warnings = get_github_issue_comments_response(repo_path, issue_number, max_comments=20)
            warnings.extend(comment_warnings)
        except GithubIssueApiError as error:
            warnings.append(_comments_unavailable_warning(error))

    evidence = extract_issue_evidence(issue, comments)
    candidates, ranking_warnings = rank_issue_candidates(analysis, evidence, max_candidates=max(12, max_nodes * 3))
    warnings.extend(ranking_warnings)
    overview_graph = build_overview_graph_projection(analysis)
    focus_graph, selected_node_ids, focus_warnings = build_focus_graph_projection(
        analysis,
        candidates,
        max_focus_nodes=max(24, max_nodes * 6),
        max_selected_nodes=max_nodes,
    )
    warnings.extend(focus_warnings)
    code_context, code_warnings = build_code_context(
        analysis,
        candidates,
        max_context_files=max_context_files,
    )
    warnings.extend(code_warnings)
    hypotheses = _issue_hypotheses(candidates)
    investigation_path = _issue_investigation_path(candidates)
    confidence = _issue_confidence(candidates, warnings)
    limits = {
        'max_nodes': max_nodes,
        'include_comments': include_comments,
        'max_context_files': max_context_files,
        'max_candidates': max(12, max_nodes * 3),
        'llm_enabled': _issue_harness_enabled(),
        'harness_enabled': _issue_harness_enabled(),
    }
    harness: dict[str, Any] = {'enabled': False, 'source': 'deterministic'}

    if _issue_harness_enabled():
        try:
            harness_job = build_issue_harness_job(
                repo_path=repo_path,
                revision=analysis_run.revision,
                issue=issue,
                comments=comments,
                evidence=evidence,
                candidates=candidates,
                analysis=analysis,
            )
            harness_result = run_issue_harness(
                harness_job,
                command=_issue_harness_command(),
                timeout_seconds=int(getattr(settings, 'ISSUE_HARNESS_TIMEOUT_SECONDS', 180)),
            )
            harness = _issue_harness_summary(harness_result)
            harness_candidates, harness_warnings = _issue_harness_candidates(
                analysis,
                harness_result.output,
                candidates,
                max_candidates=max(12, max_nodes * 3),
            )
            warnings.extend(harness_warnings)
            if _issue_harness_node_ids(harness_result.output) and not harness_candidates:
                raise IssueHarnessUnavailable('harness_no_valid_nodes', 'Issue harness did not return any node_id present in the analysis graph.')
            candidates = harness_candidates
            focus_graph, selected_node_ids, focus_warnings = build_focus_graph_projection(
                analysis,
                candidates,
                max_focus_nodes=max(24, max_nodes * 6),
                max_selected_nodes=max_nodes,
            )
            warnings.extend(focus_warnings)
            code_context, code_warnings = build_code_context(
                analysis,
                candidates,
                max_context_files=max_context_files,
            )
            warnings.extend(code_warnings)
            fallback_hypotheses = _issue_hypotheses(candidates)
            fallback_investigation_path = _issue_investigation_path(candidates)
            fallback_confidence = _issue_confidence(candidates, warnings)
            llm_fields, llm_warnings = sanitize_issue_explanation_output(
                harness_result.output,
                focus_graph=focus_graph,
                code_context=code_context,
                fallback_hypotheses=fallback_hypotheses,
                fallback_investigation_path=fallback_investigation_path,
                fallback_confidence=fallback_confidence,
                source='harness',
            )
            hypotheses = llm_fields['hypotheses']
            investigation_path = llm_fields['investigation_path']
            confidence = llm_fields['confidence']
            warnings.extend(llm_warnings)
        except IssueHarnessUnavailable as error:
            harness = {'enabled': True, 'source': 'deterministic', 'fallback_reason': error.code}
            warnings.append(_issue_harness_warning(error))
    else:
        warnings.append({'code': 'harness_disabled', 'message': 'Issue harness flag가 꺼져 있어 deterministic issue ranking을 반환했습니다.'})
    low_confidence = confidence.get('source') != 'harness' and issue_candidates_are_low_confidence(candidates, warnings)
    if low_confidence:
        confidence = _low_confidence_confidence(confidence, warnings)
    guide = build_issue_navigation_guide(
        candidates=candidates,
        hypotheses=hypotheses,
        investigation_path=investigation_path,
        confidence=confidence,
        warnings=warnings,
        evidence=evidence,
        low_confidence=low_confidence,
    )

    return {
        'analysis_id': analysis_run.id,
        'repo': repo_path,
        'revision': analysis_run.revision,
        'provider': 'github',
        'source': 'github',
        'mock': False,
        'issue': _issue_ref_payload(issue),
        'selected_node_ids': selected_node_ids,
        'candidates': candidates,
        'overview_graph': overview_graph,
        'focus_graph': focus_graph,
        'hypotheses': hypotheses,
        'investigation_path': investigation_path,
        **guide,
        'code_context': code_context,
        'confidence': confidence,
        'harness': harness,
        'limits': limits,
        'warnings': warnings,
    }


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


def _generate_share_token() -> str:
    for _attempt in range(10):
        token = secrets.token_urlsafe(24)
        if not ShareLink.objects.filter(token=token).exists():
            return token
    raise ShareInputError('share token을 생성할 수 없습니다')


def _share_is_expired(share_link: ShareLink) -> bool:
    return share_link.expires_at is not None and share_link.expires_at <= timezone.now()


def _repository_payload(repository: Repository) -> dict[str, Any]:
    return {
        'provider': repository.provider,
        'owner': repository.owner,
        'name': repository.name,
        'full_name': repository.full_name,
        'default_branch': repository.default_branch,
    }


def _build_share_response(share_link: ShareLink, analysis_run: AnalysisRun) -> dict[str, Any]:
    try:
        artifact = analysis_run.artifact
    except AnalysisArtifact.DoesNotExist:
        raise ShareInputError('share에 연결된 분석 artifact가 없습니다')

    graph = build_graph_response(artifact.payload, analysis_run)
    return {
        'share_id': share_link.token,
        'mode': share_link.mode,
        'title': share_link.title,
        'repo': analysis_run.repository.full_name,
        'repository': _repository_payload(analysis_run.repository),
        'ref': share_link.ref,
        'revision': analysis_run.revision,
        'analysis_id': analysis_run.id,
        'graph': graph,
        'is_active': share_link.is_active,
        'created_at': share_link.created_at.isoformat(),
        'expires_at': share_link.expires_at.isoformat() if share_link.expires_at else None,
        'warnings': graph.get('warnings', []),
    }


def create_share_response(
    repo_path: str,
    *,
    mode: str = ShareLink.MODE_FIXED,
    revision: str | None = None,
    title: str = '',
    expires_at=None,
) -> dict[str, Any] | None:
    if mode not in {ShareLink.MODE_FIXED, ShareLink.MODE_LATEST}:
        raise ShareInputError('unsupported share mode')
    if mode == ShareLink.MODE_LATEST and revision is not None:
        raise ShareInputError('latest share에는 revision을 지정할 수 없습니다')

    analysis = get_repo_analysis(repo_path, revision if mode == ShareLink.MODE_FIXED else None)
    if analysis is None:
        return None
    analysis_run = get_analysis_run_by_revision(repo_path, str(analysis['revision']))
    if analysis_run is None:
        return None

    share_link = ShareLink.objects.create(
        token=_generate_share_token(),
        repository=analysis_run.repository,
        analysis_run=analysis_run,
        mode=mode,
        ref='HEAD' if mode == ShareLink.MODE_LATEST else str(revision or analysis_run.revision),
        title=title,
        expires_at=expires_at,
    )
    return _build_share_response(share_link, analysis_run)


def _resolve_share_analysis_run(share_link: ShareLink) -> AnalysisRun | None:
    if not share_link.is_active or _share_is_expired(share_link):
        return None

    if share_link.mode == ShareLink.MODE_FIXED:
        return share_link.analysis_run

    analysis = get_repo_analysis(share_link.repository.full_name)
    if analysis is None:
        return share_link.analysis_run
    analysis_run = get_analysis_run_by_revision(share_link.repository.full_name, str(analysis['revision']))
    if analysis_run is None:
        return share_link.analysis_run
    if share_link.analysis_run_id != analysis_run.id:
        share_link.analysis_run = analysis_run
        share_link.save(update_fields=['analysis_run', 'updated_at'])
    return analysis_run


def get_share_response(share_id: str) -> dict[str, Any] | None:
    share_link = (
        ShareLink.objects
        .select_related('repository', 'analysis_run', 'analysis_run__repository')
        .filter(token=share_id)
        .first()
    )
    if share_link is None:
        return None
    analysis_run = _resolve_share_analysis_run(share_link)
    if analysis_run is None:
        return None
    return _build_share_response(share_link, analysis_run)
