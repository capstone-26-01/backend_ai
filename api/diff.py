from __future__ import annotations

from typing import Any, Callable, Hashable, Mapping


GRAPH_DIFF_SCHEMA_VERSION = 'graph-diff.v1'

NODE_COMPARE_FIELDS = (
    'kind',
    'label',
    'path',
    'parent_id',
    'symbol',
    'language',
    'start_line',
    'end_line',
    'metadata',
)
EDGE_COMPARE_FIELDS = ('confidence', 'metadata')
TOP_LEVEL_METADATA_FIELDS = ('default_branch', 'entrypoints', 'key_modules', 'warnings')


class GraphDiffInputError(ValueError):
    pass


def _item_id(item: Mapping[str, Any]) -> str:
    return str(item.get('id', ''))


def _edge_identity(edge: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(edge.get('kind') or edge.get('type') or ''),
        str(edge.get('source') or ''),
        str(edge.get('target') or ''),
        str(edge.get('path') or edge.get('file') or ''),
    )


def _index_items(
    items: list[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], Hashable],
    *,
    item_kind: str,
    warnings: list[dict[str, Any]],
) -> dict[Hashable, Mapping[str, Any]]:
    indexed: dict[Hashable, Mapping[str, Any]] = {}
    for item in items:
        key = key_fn(item)
        if not key:
            warnings.append({'code': 'diff_missing_identity', 'item_kind': item_kind, 'item': dict(item)})
            continue
        if key in indexed:
            warnings.append({'code': 'diff_duplicate_identity', 'item_kind': item_kind, 'identity': str(key)})
        indexed[key] = item
    return indexed


def _changed_fields(
    base_item: Mapping[str, Any],
    head_item: Mapping[str, Any],
    fields: tuple[str, ...],
) -> list[str]:
    return [field for field in fields if base_item.get(field) != head_item.get(field)]


def _diff_indexed_items(
    base_items: dict[Hashable, Mapping[str, Any]],
    head_items: dict[Hashable, Mapping[str, Any]],
    *,
    compare_fields: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    base_keys = set(base_items)
    head_keys = set(head_items)
    added = [dict(head_items[key]) for key in sorted(head_keys - base_keys, key=str)]
    removed = [dict(base_items[key]) for key in sorted(base_keys - head_keys, key=str)]
    changed = []

    for key in sorted(base_keys & head_keys, key=str):
        base_item = base_items[key]
        head_item = head_items[key]
        fields = _changed_fields(base_item, head_item, compare_fields)
        if fields:
            changed.append(
                {
                    'id': _item_id(head_item) or str(key),
                    'identity': str(key),
                    'changed_fields': fields,
                    'before': dict(base_item),
                    'after': dict(head_item),
                }
            )

    return {'added': added, 'removed': removed, 'changed': changed}


def _metadata_changes(base_artifact: Mapping[str, Any], head_artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    changes = []
    for field in TOP_LEVEL_METADATA_FIELDS:
        if base_artifact.get(field) != head_artifact.get(field):
            changes.append(
                {
                    'field': field,
                    'before': base_artifact.get(field),
                    'after': head_artifact.get(field),
                }
            )
    return changes


def compare_graph_artifacts(base_artifact: Mapping[str, Any], head_artifact: Mapping[str, Any]) -> dict[str, Any]:
    base_repo = str(base_artifact.get('repo') or '')
    head_repo = str(head_artifact.get('repo') or '')
    if not base_repo or not head_repo:
        raise GraphDiffInputError('diff artifacts must include repo')
    if base_repo != head_repo:
        raise GraphDiffInputError('diff artifacts must belong to the same repo')

    warnings: list[dict[str, Any]] = []
    if base_artifact.get('schema_version') != head_artifact.get('schema_version'):
        warnings.append(
            {
                'code': 'schema_version_mismatch',
                'base_schema_version': base_artifact.get('schema_version'),
                'head_schema_version': head_artifact.get('schema_version'),
            }
        )

    base_nodes = _index_items(list(base_artifact.get('nodes', [])), _item_id, item_kind='node', warnings=warnings)
    head_nodes = _index_items(list(head_artifact.get('nodes', [])), _item_id, item_kind='node', warnings=warnings)
    base_edges = _index_items(list(base_artifact.get('edges', [])), _edge_identity, item_kind='edge', warnings=warnings)
    head_edges = _index_items(list(head_artifact.get('edges', [])), _edge_identity, item_kind='edge', warnings=warnings)
    node_diff = _diff_indexed_items(base_nodes, head_nodes, compare_fields=NODE_COMPARE_FIELDS)
    edge_diff = _diff_indexed_items(base_edges, head_edges, compare_fields=EDGE_COMPARE_FIELDS)
    metadata_changes = _metadata_changes(base_artifact, head_artifact)

    return {
        'schema_version': GRAPH_DIFF_SCHEMA_VERSION,
        'repo': head_repo,
        'base_revision': str(base_artifact.get('revision') or ''),
        'head_revision': str(head_artifact.get('revision') or ''),
        'summary': {
            'added_nodes': len(node_diff['added']),
            'removed_nodes': len(node_diff['removed']),
            'changed_nodes': len(node_diff['changed']),
            'added_edges': len(edge_diff['added']),
            'removed_edges': len(edge_diff['removed']),
            'changed_edges': len(edge_diff['changed']),
            'changed_metadata': len(metadata_changes),
        },
        'nodes': node_diff,
        'edges': edge_diff,
        'metadata': {'changed': metadata_changes},
        'warnings': warnings,
    }
