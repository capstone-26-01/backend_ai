from __future__ import annotations

from collections import Counter
import json
import os
from typing import Any, Mapping, cast

from llm.context_selection import build_context_for_files
from llm.services import _generate_answer


SUMMARY_PROMPT_VERSION = 'summary.v1'
SUMMARY_KIND_REPO_OVERVIEW = 'repo_overview'
SUMMARY_KIND_ONBOARDING = 'onboarding_guide'
SUMMARY_KIND_NODE = 'node'
SUMMARY_KINDS = {SUMMARY_KIND_REPO_OVERVIEW, SUMMARY_KIND_ONBOARDING}
MAX_PROMPT_JSON_CHARS = 6000
MAX_EXCERPT_CHARS = 5000


class SummaryUnavailable(RuntimeError):
    pass


class SummaryInputError(ValueError):
    pass


def summary_cache_key(kind: str, *, node_id: str | None = None, prompt_version: str | None = None) -> str:
    version = prompt_version or SUMMARY_PROMPT_VERSION
    if kind == SUMMARY_KIND_NODE:
        if not node_id:
            raise SummaryInputError('node_id is required for node summary')
        return f'{kind}:{node_id}:{version}'
    return f'{kind}:{version}'


def _nodes_by_id(analysis: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node.get('id')): dict(cast(Mapping[str, Any], node))
        for node in analysis.get('nodes', [])
        if cast(Mapping[str, Any], node).get('id')
    }


def _node_file(node: Mapping[str, Any]) -> str | None:
    file_path = node.get('path') or node.get('file')
    if isinstance(file_path, str) and file_path:
        return file_path
    return None


def _node_kind(node: Mapping[str, Any]) -> str:
    return str(node.get('kind') or node.get('type') or '')


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _entrypoint_ids(analysis: Mapping[str, Any]) -> list[str]:
    return _unique([
        str(entrypoint.get('id'))
        for entrypoint in analysis.get('entrypoints', [])
        if cast(Mapping[str, Any], entrypoint).get('id')
    ])


def _key_module_refs(analysis: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    ids = []
    paths = []
    for module in analysis.get('key_modules', []):
        module_mapping = cast(Mapping[str, Any], module)
        module_id = module_mapping.get('id')
        module_path = module_mapping.get('path')
        if isinstance(module_id, str):
            ids.append(module_id)
        if isinstance(module_path, str):
            paths.append(module_path)
    return _unique(ids), _unique(paths)


def _neighbor_node_ids(analysis: Mapping[str, Any], node_id: str) -> list[str]:
    neighbors = [node_id]
    for edge in analysis.get('edges', []):
        edge_mapping = cast(Mapping[str, Any], edge)
        source = str(edge_mapping.get('source', ''))
        target = str(edge_mapping.get('target', ''))
        if source == node_id:
            neighbors.append(target)
        if target == node_id:
            neighbors.append(source)
    return _unique(neighbors)


def _summary_source_refs(
    analysis: Mapping[str, Any],
    kind: str,
    *,
    node_id: str | None = None,
) -> tuple[list[str], list[str]]:
    nodes_by_id = _nodes_by_id(analysis)
    key_module_ids, key_module_paths = _key_module_refs(analysis)
    if kind == SUMMARY_KIND_NODE:
        if node_id not in nodes_by_id:
            raise SummaryInputError('node_id not found')
        source_nodes = [node for node in _neighbor_node_ids(analysis, cast(str, node_id)) if node in nodes_by_id]
    else:
        source_nodes = _unique([*_entrypoint_ids(analysis), *key_module_ids])

    source_files = [
        _node_file(nodes_by_id[source_node])
        for source_node in source_nodes
        if source_node in nodes_by_id and _node_file(nodes_by_id[source_node])
    ]
    source_files.extend(key_module_paths)
    if not source_files:
        source_files = sorted(cast(Mapping[str, str], analysis.get('file_contents', {})).keys())[:4]
    return source_nodes[:12], _unique(cast(list[str], source_files))[:6]


def _graph_overview(analysis: Mapping[str, Any]) -> dict[str, Any]:
    nodes = [cast(Mapping[str, Any], node) for node in analysis.get('nodes', [])]
    edges = [cast(Mapping[str, Any], edge) for edge in analysis.get('edges', [])]
    node_counts = Counter(_node_kind(node) for node in nodes)
    edge_counts = Counter(str(edge.get('kind') or edge.get('type') or '') for edge in edges)
    return {
        'repo': analysis.get('repo'),
        'revision': analysis.get('revision'),
        'node_counts': dict(sorted(node_counts.items())),
        'edge_counts': dict(sorted(edge_counts.items())),
        'entrypoints': list(analysis.get('entrypoints', []))[:8],
        'key_modules': list(analysis.get('key_modules', []))[:8],
        'warnings': list(analysis.get('warnings', []))[:8],
    }


def _node_payload(analysis: Mapping[str, Any], source_nodes: list[str]) -> list[dict[str, Any]]:
    nodes_by_id = _nodes_by_id(analysis)
    payload = []
    for node_id in source_nodes:
        node = nodes_by_id.get(node_id)
        if not node:
            continue
        payload.append(
            {
                'id': node_id,
                'kind': _node_kind(node),
                'label': node.get('label'),
                'path': _node_file(node),
                'start_line': node.get('start_line'),
                'end_line': node.get('end_line'),
                'metadata': node.get('metadata', {}),
            }
        )
    return payload


def _build_prompt_payload(
    analysis: Mapping[str, Any],
    kind: str,
    *,
    node_id: str | None = None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    source_nodes, source_files = _summary_source_refs(analysis, kind, node_id=node_id)
    excerpts = build_context_for_files(
        analysis,
        source_files,
        node_ids=source_nodes,
        max_chars=MAX_EXCERPT_CHARS,
    )
    payload = {
        'kind': kind,
        'target_node_id': node_id,
        'graph': _graph_overview(analysis),
        'source_nodes': _node_payload(analysis, source_nodes),
        'source_files': source_files,
        'code_excerpts': excerpts.context,
    }
    return payload, source_nodes, source_files


def _build_summary_messages(kind: str, prompt_payload: Mapping[str, Any]) -> list[dict[str, str]]:
    if kind == SUMMARY_KIND_ONBOARDING:
        task = (
            '새 contributor가 이 저장소를 빠르게 이해하도록 onboarding guide를 작성해. '
            'entrypoint, key module, 읽을 순서, 주의할 warning을 포함해.'
        )
    elif kind == SUMMARY_KIND_NODE:
        task = '선택된 node/file이 무엇을 하는지 짧게 설명하고 관련 호출/파일 근거를 포함해.'
    else:
        task = '저장소의 목적, 주요 구조, entrypoint, key module을 간결하게 요약해.'

    payload_json = json.dumps(prompt_payload, ensure_ascii=False, indent=2)
    if len(payload_json) > MAX_PROMPT_JSON_CHARS:
        payload_json = payload_json[:MAX_PROMPT_JSON_CHARS]

    return [
        {
            'role': 'system',
            'content': '너는 Python 코드베이스 onboarding 문서를 쓰는 시니어 백엔드 엔지니어야. 제공된 graph와 excerpt만 근거로 한국어로 답해.',
        },
        {
            'role': 'user',
            'content': f'{task}\n\n근거 JSON:\n{payload_json}',
        },
    ]


def generate_summary(analysis: Mapping[str, Any], kind: str, *, node_id: str | None = None) -> dict[str, Any]:
    if kind not in SUMMARY_KINDS and kind != SUMMARY_KIND_NODE:
        raise SummaryInputError('unsupported summary kind')

    prompt_payload, source_nodes, source_files = _build_prompt_payload(analysis, kind, node_id=node_id)
    try:
        text = _generate_answer(_build_summary_messages(kind, prompt_payload))
    except Exception as exc:
        raise SummaryUnavailable(str(exc)) from exc

    return {
        'kind': kind,
        'prompt_version': SUMMARY_PROMPT_VERSION,
        'model': {
            'openai': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            'gemini': os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'),
        },
        'target_id': node_id,
        'text': text,
        'source_nodes': source_nodes,
        'source_files': source_files,
        'warnings': list(analysis.get('warnings', []))[:8],
    }
