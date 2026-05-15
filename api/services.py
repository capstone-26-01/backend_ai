from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from django.conf import settings

from api.artifacts import GRAPH_ARTIFACT_SCHEMA_VERSION, build_graph_artifact, coerce_graph_artifact
from api.serializers import is_safe_revision, _is_safe_repo_segment
from github_repo.services import get_file_content, get_repo_snapshot
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


def get_repo_analysis(repo_path: str, revision: str | None = None) -> dict[str, Any] | None:
    try:
        _analysis_parts(repo_path)
    except ValueError:
        return None

    if revision is not None:
        if not is_safe_revision(revision):
            return None
        artifact_path = _analysis_path(repo_path, revision)
        if artifact_path.exists():
            return _read_analysis_artifact(artifact_path)
        return None

    snapshot = get_repo_snapshot(repo_path)
    if snapshot is None:
        return None
    revision, files = snapshot
    if not files:
        return None

    artifact_path = _analysis_path(repo_path, revision)
    if artifact_path.exists():
        return _read_analysis_artifact(artifact_path)

    python_files = [file_path for file_path in files if file_path.endswith('.py')]
    file_contents = {
        file_path: content
        for file_path in python_files
        if (content := get_file_content(repo_path, file_path, revision)) is not None
    }

    graph = parse_repo(repo_path, python_files, lambda _repo_path, file_path: file_contents.get(file_path))
    analysis = build_graph_artifact(
        repo_path=repo_path,
        revision=revision,
        graph=graph,
        file_contents=file_contents,
    )
    _write_analysis_artifact(artifact_path, analysis)
    return analysis
