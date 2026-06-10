from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from parser.language_registry import LANGUAGE_BY_ID, language_for_path


GRAPH_ARTIFACT_SCHEMA_VERSION = 'graph-artifact.v1'

ALLOWED_NODE_KINDS = frozenset({'directory', 'file', 'module', 'class', 'function', 'method', 'external'})
ALLOWED_EDGE_KINDS = frozenset({'contains', 'imports', 'inherits', 'calls', 'references', 'entrypoint'})
ALLOWED_STATUSES = frozenset({'succeeded', 'failed', 'partial'})

REQUIRED_TOP_LEVEL_FIELDS = frozenset({
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
    'tree',
    'nodes',
    'edges',
    'entrypoints',
    'key_modules',
    'summaries',
    'warnings',
})
REQUIRED_NODE_FIELDS = frozenset({
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
})
REQUIRED_EDGE_FIELDS = frozenset({
    'id',
    'kind',
    'source',
    'target',
    'path',
    'confidence',
    'metadata',
})

DEFAULT_LIMITS = {
    'max_files': None,
    'max_python_files': None,
    'max_js_ts_files': None,
    'max_single_file_bytes': None,
    'max_total_analyzed_bytes': None,
}


class ArtifactValidationError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _repo_parts(repo_path: str) -> tuple[str, str]:
    parts = repo_path.split('/')
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ArtifactValidationError('repo must use owner/name format')
    return parts[0], parts[1]


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _line_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactValidationError('line fields must be integers or null')
    return value


def _language_from_path(path: str | None) -> str | None:
    if not path:
        return None
    spec = language_for_path(path, enabled_languages=LANGUAGE_BY_ID.keys())
    return spec.id if spec is not None else None


def _legacy_node_kind(node: Mapping[str, Any], nodes_by_id: Mapping[str, Mapping[str, Any]]) -> str:
    existing_kind = node.get('kind')
    if isinstance(existing_kind, str) and existing_kind:
        return existing_kind

    legacy_type = str(node.get('type', 'external'))
    if legacy_type == 'function':
        parent_id = node.get('parent_id') or node.get('parent')
        parent = nodes_by_id.get(str(parent_id)) if parent_id else None
        if parent and parent.get('type') == 'class':
            return 'method'
    if legacy_type in ALLOWED_NODE_KINDS:
        return legacy_type
    return 'external'


def _normalize_node(node: Mapping[str, Any], nodes_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    node_id = str(node.get('id', ''))
    if not node_id:
        raise ArtifactValidationError('node id is required')

    kind = _legacy_node_kind(node, nodes_by_id)
    path = _string_or_none(node.get('path', node.get('file')))
    parent_id = _string_or_none(node.get('parent_id', node.get('parent')))
    label = str(node.get('label', node_id))
    metadata = dict(node.get('metadata') or {})
    legacy_type = node.get('type')
    if legacy_type is not None:
        metadata.setdefault('legacy_type', legacy_type)

    normalized = {
        'id': node_id,
        'kind': kind,
        'label': label,
        'path': path,
        'parent_id': parent_id,
        'symbol': _string_or_none(node.get('symbol', label if kind in {'class', 'function', 'method'} else None)),
        'language': _string_or_none(node.get('language', _language_from_path(path))),
        'start_line': _line_or_none(node.get('start_line')),
        'end_line': _line_or_none(node.get('end_line')),
        'metadata': metadata,
        'type': str(legacy_type or kind),
    }
    if path is not None:
        normalized['file'] = path
    if parent_id is not None:
        normalized['parent'] = parent_id
    return normalized


def _normalize_edge(edge: Mapping[str, Any], index: int) -> dict[str, Any]:
    legacy_kind = edge.get('type')
    kind = str(edge.get('kind') or legacy_kind or '')
    path = _string_or_none(edge.get('path', edge.get('file')))
    metadata = dict(edge.get('metadata') or {})
    if legacy_kind is not None:
        metadata.setdefault('legacy_type', legacy_kind)

    normalized = {
        'id': str(edge.get('id') or f'e{index}'),
        'kind': kind,
        'source': str(edge.get('source', '')),
        'target': str(edge.get('target', '')),
        'path': path,
        'confidence': float(edge.get('confidence', 1.0)),
        'metadata': metadata,
        'type': kind,
    }
    if path is not None:
        normalized['file'] = path
    return normalized


def build_graph_artifact(
    *,
    repo_path: str,
    revision: str,
    graph: Mapping[str, Any],
    file_contents: Mapping[str, str] | None = None,
    provider: str = 'github',
    ref: str = 'HEAD',
    default_branch: str | None = None,
    generated_at: str | None = None,
    status: str = 'succeeded',
    limits: Mapping[str, Any] | None = None,
    entrypoints: list[dict[str, Any]] | None = None,
    key_modules: list[dict[str, Any]] | None = None,
    summaries: Mapping[str, Any] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    analysis_profile: str = 'python-v1',
    languages: list[str] | tuple[str, ...] | None = None,
    file_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    owner, name = _repo_parts(repo_path)
    raw_nodes = list(graph.get('nodes', []))
    nodes_by_id = {str(node.get('id')): node for node in raw_nodes if node.get('id')}
    nodes = [_normalize_node(node, nodes_by_id) for node in raw_nodes]
    edges = [_normalize_edge(edge, index) for index, edge in enumerate(graph.get('edges', []), start=1)]

    artifact = {
        'schema_version': GRAPH_ARTIFACT_SCHEMA_VERSION,
        'repo': repo_path,
        'provider': provider,
        'owner': owner,
        'name': name,
        'ref': ref,
        'revision': revision,
        'default_branch': default_branch,
        'generated_at': generated_at or _utc_now(),
        'status': status,
        'analysis_profile': analysis_profile,
        'limits': {**DEFAULT_LIMITS, **dict(limits or {})},
        'languages': list(languages or []),
        'file_manifest': dict(file_manifest or {}),
        'file_contents': dict(file_contents or {}),
        'tree': list(graph.get('tree', [])),
        'nodes': nodes,
        'edges': edges,
        'entrypoints': list(entrypoints or []),
        'key_modules': list(key_modules or []),
        'summaries': dict(summaries or {}),
        'warnings': list(warnings or []),
    }
    validate_graph_artifact(artifact)
    return artifact


def coerce_graph_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get('schema_version') == GRAPH_ARTIFACT_SCHEMA_VERSION:
        artifact = dict(payload)
        artifact.setdefault('analysis_profile', 'python-v1')
        artifact.setdefault('languages', [])
        artifact.setdefault('file_manifest', {})
        validate_graph_artifact(artifact)
        return artifact

    repo_path = str(payload.get('repo', ''))
    revision = str(payload.get('revision', ''))
    graph = {
        'tree': payload.get('tree', []),
        'nodes': payload.get('nodes', []),
        'edges': payload.get('edges', []),
    }
    return build_graph_artifact(
        repo_path=repo_path,
        revision=revision,
        graph=graph,
        file_contents=payload.get('file_contents', {}),
        ref=str(payload.get('ref') or 'HEAD'),
        default_branch=payload.get('default_branch'),
        generated_at=payload.get('generated_at'),
        analysis_profile=str(payload.get('analysis_profile') or 'python-v1'),
        languages=payload.get('languages') or [],
        file_manifest=payload.get('file_manifest') or {},
    )


def _assert_required_fields(payload: Mapping[str, Any], required_fields: frozenset[str], location: str) -> None:
    missing = sorted(required_fields - payload.keys())
    if missing:
        raise ArtifactValidationError(f'{location} missing required fields: {", ".join(missing)}')


def validate_graph_artifact(artifact: Mapping[str, Any]) -> None:
    _assert_required_fields(artifact, REQUIRED_TOP_LEVEL_FIELDS, 'artifact')
    if artifact['schema_version'] != GRAPH_ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError('unsupported graph artifact schema version')
    if artifact['status'] not in ALLOWED_STATUSES:
        raise ArtifactValidationError('unknown artifact status')
    _repo_parts(str(artifact['repo']))

    if not isinstance(artifact['limits'], dict):
        raise ArtifactValidationError('limits must be an object')
    if not isinstance(artifact['tree'], list):
        raise ArtifactValidationError('tree must be a list')
    if not isinstance(artifact['nodes'], list):
        raise ArtifactValidationError('nodes must be a list')
    if not isinstance(artifact['edges'], list):
        raise ArtifactValidationError('edges must be a list')
    if not isinstance(artifact['entrypoints'], list):
        raise ArtifactValidationError('entrypoints must be a list')
    if not isinstance(artifact['key_modules'], list):
        raise ArtifactValidationError('key_modules must be a list')
    if not isinstance(artifact['summaries'], dict):
        raise ArtifactValidationError('summaries must be an object')
    if not isinstance(artifact['warnings'], list):
        raise ArtifactValidationError('warnings must be a list')

    seen_node_ids: set[str] = set()
    for node in artifact['nodes']:
        if not isinstance(node, dict):
            raise ArtifactValidationError('node must be an object')
        _assert_required_fields(node, REQUIRED_NODE_FIELDS, 'node')
        if node['kind'] not in ALLOWED_NODE_KINDS:
            raise ArtifactValidationError(f'unknown node kind: {node["kind"]}')
        if not node['id'] or node['id'] in seen_node_ids:
            raise ArtifactValidationError('node ids must be non-empty and unique')
        if node['path'] is not None and not isinstance(node['path'], str):
            raise ArtifactValidationError('node path must be a string or null')
        if not isinstance(node['metadata'], dict):
            raise ArtifactValidationError('node metadata must be an object')
        seen_node_ids.add(node['id'])

    seen_edge_ids: set[str] = set()
    for edge in artifact['edges']:
        if not isinstance(edge, dict):
            raise ArtifactValidationError('edge must be an object')
        _assert_required_fields(edge, REQUIRED_EDGE_FIELDS, 'edge')
        if edge['kind'] not in ALLOWED_EDGE_KINDS:
            raise ArtifactValidationError(f'unknown edge kind: {edge["kind"]}')
        if not edge['id'] or edge['id'] in seen_edge_ids:
            raise ArtifactValidationError('edge ids must be non-empty and unique')
        if not edge['source'] or not edge['target']:
            raise ArtifactValidationError('edge source and target are required')
        if edge['path'] is not None and not isinstance(edge['path'], str):
            raise ArtifactValidationError('edge path must be a string or null')
        confidence = edge['confidence']
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ArtifactValidationError('edge confidence must be a number between 0 and 1')
        if not isinstance(edge['metadata'], dict):
            raise ArtifactValidationError('edge metadata must be an object')
        seen_edge_ids.add(edge['id'])
