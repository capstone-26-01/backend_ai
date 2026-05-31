from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast
import os
import re

from api.readme_svg import select_overview_cards
from llm.context_selection import identifier_tokens, score_nodes


FILE_PATH_RE = re.compile(r'(?P<path>(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py|[A-Za-z0-9_.-]+\.py)(?::(?P<line>\d+))?')
PY_STACK_RE = re.compile(r'File "(?P<path>[^"]+\.py)", line (?P<line>\d+), in (?P<symbol>[A-Za-z_][A-Za-z0-9_]*)')
PYTEST_FRAME_RE = re.compile(r'(?P<path>(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py):(?P<line>\d+)(?::in\s+(?P<symbol>[A-Za-z_][A-Za-z0-9_]*))?')
BACKTICK_RE = re.compile(r'`([^`]{2,120})`')
CALL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(')
ERROR_LINE_RE = re.compile(r'(?i)\b(error|exception|traceback|failed|failure|timeout|crash|invalid)\b')
SYMBOL_TOKEN_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _string(value: Any) -> str:
    if value is None:
        return ''
    return str(value)


def _normalized_source_texts(
    issue: Mapping[str, Any],
    comments: Sequence[Mapping[str, Any]] | None,
) -> list[tuple[str, str]]:
    texts = [
        ('title', _string(issue.get('title'))),
        ('body', _string(issue.get('body') or issue.get('body_excerpt'))),
    ]
    for index, comment in enumerate(comments or [], start=1):
        texts.append((f'comment:{index}', _string(comment.get('body'))))
    return [(source, text) for source, text in texts if text]


def _dedupe(items: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        marker = tuple(item.get(key) for key in keys)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    return deduped


def _extract_file_mentions(source: str, text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for match in FILE_PATH_RE.finditer(text):
        path = match.group('path')
        line = match.group('line')
        mentions.append(
            {
                'path': path,
                'line': int(line) if line else None,
                'source': source,
                'confidence': 1.0 if '/' in path else 0.75,
            }
        )
    return mentions


def _extract_stack_frames(source: str, text: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for regex in (PY_STACK_RE, PYTEST_FRAME_RE):
        for match in regex.finditer(text):
            frames.append(
                {
                    'path': match.group('path'),
                    'line': int(match.group('line')),
                    'symbol': match.groupdict().get('symbol') or None,
                    'source': source,
                    'confidence': 1.0,
                }
            )
    return frames


def _symbol_from_fragment(fragment: str) -> str | None:
    candidate = fragment.strip().split('::')[-1].split('.')[-1]
    candidate = candidate.removesuffix('()')
    if SYMBOL_TOKEN_RE.fullmatch(candidate):
        return candidate
    return None


def _extract_symbol_mentions(source: str, text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for match in BACKTICK_RE.finditer(text):
        symbol = _symbol_from_fragment(match.group(1))
        if symbol:
            mentions.append({'symbol': symbol, 'source': source, 'confidence': 0.95})
    for match in CALL_RE.finditer(text):
        mentions.append({'symbol': match.group(1), 'source': source, 'confidence': 0.8})
    return mentions


def _extract_error_phrases(source: str, text: str) -> list[dict[str, Any]]:
    phrases: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if 5 <= len(line) <= 220 and ERROR_LINE_RE.search(line):
            phrases.append({'text': line, 'source': source})
    for match in BACKTICK_RE.finditer(text):
        quoted = match.group(1).strip()
        if 5 <= len(quoted) <= 220 and ERROR_LINE_RE.search(quoted):
            phrases.append({'text': quoted, 'source': source})
    return phrases


def _issue_labels(issue: Mapping[str, Any]) -> list[dict[str, str]]:
    labels = issue.get('labels')
    if not isinstance(labels, list):
        return []
    result: list[dict[str, str]] = []
    for label in labels:
        if not isinstance(label, Mapping):
            continue
        name = _string(label.get('name')).strip()
        if name:
            result.append({'name': name, 'description': _string(label.get('description'))})
    return result


def extract_issue_evidence(
    issue: Mapping[str, Any],
    comments: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    texts = _normalized_source_texts(issue, comments)
    file_mentions: list[dict[str, Any]] = []
    stack_frames: list[dict[str, Any]] = []
    symbol_mentions: list[dict[str, Any]] = []
    quoted_errors: list[dict[str, Any]] = []

    for source, text in texts:
        file_mentions.extend(_extract_file_mentions(source, text))
        stack_frames.extend(_extract_stack_frames(source, text))
        symbol_mentions.extend(_extract_symbol_mentions(source, text))
        quoted_errors.extend(_extract_error_phrases(source, text))

    for frame in stack_frames:
        file_mentions.append(
            {
                'path': frame['path'],
                'line': frame['line'],
                'source': frame['source'],
                'confidence': 1.0,
            }
        )
        if frame.get('symbol'):
            symbol_mentions.append(
                {
                    'symbol': frame['symbol'],
                    'source': frame['source'],
                    'confidence': 1.0,
                }
            )

    labels = _issue_labels(issue)
    query_parts = [
        _string(issue.get('title')),
        _string(issue.get('body') or issue.get('body_excerpt')),
        ' '.join(label['name'] for label in labels),
        ' '.join(comment.get('body', '') for comment in comments or [] if isinstance(comment.get('body'), str)),
        ' '.join(item['path'] for item in file_mentions),
        ' '.join(item['symbol'] for item in symbol_mentions),
    ]
    return {
        'query': ' '.join(part for part in query_parts if part),
        'file_mentions': _dedupe(file_mentions, 'path', 'line', 'source'),
        'symbol_mentions': _dedupe(symbol_mentions, 'symbol', 'source'),
        'stack_frames': _dedupe(stack_frames, 'path', 'line', 'symbol', 'source'),
        'quoted_errors': _dedupe(quoted_errors, 'text', 'source'),
        'labels': labels,
        'comments': [
            {
                'id': comment.get('id'),
                'author': comment.get('author'),
                'body': _string(comment.get('body')),
                'created_at': comment.get('created_at'),
                'updated_at': comment.get('updated_at'),
                'html_url': comment.get('html_url'),
            }
            for comment in comments or []
        ],
    }


def file_basename(path: str) -> str:
    return os.path.basename(path)


def _node_id(node: Mapping[str, Any]) -> str:
    return _string(node.get('id'))


def _node_kind(node: Mapping[str, Any]) -> str:
    return _string(node.get('kind') or node.get('type'))


def _node_path(node: Mapping[str, Any]) -> str | None:
    path = node.get('path') or node.get('file')
    if isinstance(path, str) and path:
        return path
    return None


def _node_label(node: Mapping[str, Any]) -> str:
    return _string(node.get('label') or node.get('symbol') or node.get('id'))


def _node_symbol(node: Mapping[str, Any]) -> str:
    return _string(node.get('symbol') or node.get('label'))


def _node_line_range(node: Mapping[str, Any]) -> tuple[int | None, int | None]:
    start = node.get('start_line')
    end = node.get('end_line')
    return (start if isinstance(start, int) and not isinstance(start, bool) else None, end if isinstance(end, int) and not isinstance(end, bool) else None)


def _display_node(node: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'id': _node_id(node),
        'kind': _node_kind(node),
        'label': _node_label(node),
        'path': _node_path(node),
        'start_line': node.get('start_line'),
        'end_line': node.get('end_line'),
        'metadata': dict(node.get('metadata') or {}),
    }


def _nodes_by_id(analysis: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _node_id(cast(Mapping[str, Any], node)): dict(cast(Mapping[str, Any], node))
        for node in analysis.get('nodes', [])
        if isinstance(node, Mapping) and _node_id(cast(Mapping[str, Any], node))
    }


def _entrypoint_ids(analysis: Mapping[str, Any]) -> set[str]:
    return {
        str(entrypoint.get('id'))
        for entrypoint in analysis.get('entrypoints', [])
        if isinstance(entrypoint, Mapping) and entrypoint.get('id')
    }


def _key_module_ids(analysis: Mapping[str, Any]) -> set[str]:
    return {
        str(module.get('id'))
        for module in analysis.get('key_modules', [])
        if isinstance(module, Mapping) and module.get('id')
    }


def _final_id_segment(node_id: str) -> str:
    return node_id.split('::')[-1].split('/')[-1]


def _line_matches(node: Mapping[str, Any], line: int | None) -> bool:
    if line is None:
        return False
    start, end = _node_line_range(node)
    if start is None or end is None:
        return False
    return start <= line <= end


def _add_score(
    scores: dict[str, int],
    evidence_by_node: dict[str, list[dict[str, str]]],
    node_id: str,
    amount: int,
    *,
    evidence_type: str,
    message: str,
) -> None:
    scores[node_id] = scores.get(node_id, 0) + amount
    evidence = evidence_by_node.setdefault(node_id, [])
    marker = (evidence_type, message)
    if not any((item['type'], item['message']) == marker for item in evidence):
        evidence.append({'type': evidence_type, 'message': message})


def _apply_file_boosts(
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
    scores: dict[str, int],
    evidence_by_node: dict[str, list[dict[str, str]]],
) -> None:
    for mention in evidence.get('file_mentions', []):
        if not isinstance(mention, Mapping):
            continue
        mentioned_path = _string(mention.get('path'))
        mentioned_line = mention.get('line') if isinstance(mention.get('line'), int) else None
        for node_id, node in nodes_by_id.items():
            node_path = _node_path(node)
            if node_path != mentioned_path and node_id != mentioned_path:
                continue
            if _line_matches(node, mentioned_line):
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    320,
                    evidence_type='stack_frame',
                    message=f'{mentioned_path}:{mentioned_line} stack trace line matches this node.',
                )
            elif _node_kind(node) in {'function', 'method', 'class'}:
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    180,
                    evidence_type='file_path',
                    message=f'Issue mentions file path {mentioned_path}.',
                )
            else:
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    240,
                    evidence_type='file_path',
                    message=f'Issue directly mentions file node {mentioned_path}.',
                )


def _apply_symbol_boosts(
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
    scores: dict[str, int],
    evidence_by_node: dict[str, list[dict[str, str]]],
) -> None:
    for mention in evidence.get('symbol_mentions', []):
        if not isinstance(mention, Mapping):
            continue
        symbol = _string(mention.get('symbol')).lower()
        if not symbol:
            continue
        for node_id, node in nodes_by_id.items():
            label = _node_label(node).lower()
            node_symbol = _node_symbol(node).lower()
            final_segment = _final_id_segment(node_id).lower()
            if symbol in {label, node_symbol, final_segment}:
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    280,
                    evidence_type='symbol',
                    message=f'Issue explicitly mentions symbol {symbol}.',
                )
            elif symbol in identifier_tokens(label) or symbol in identifier_tokens(node_id):
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    120,
                    evidence_type='symbol',
                    message=f'Issue symbol {symbol} partially matches this node.',
                )


def _apply_label_boosts(
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
    scores: dict[str, int],
    evidence_by_node: dict[str, list[dict[str, str]]],
) -> None:
    label_tokens = {
        token
        for label in evidence.get('labels', [])
        if isinstance(label, Mapping)
        for token in identifier_tokens(_string(label.get('name')))
    }
    if not label_tokens:
        return
    for node_id, node in nodes_by_id.items():
        node_tokens = identifier_tokens(' '.join([node_id, _node_label(node), _node_path(node) or '']))
        matched = sorted(label_tokens & node_tokens)
        if matched:
            _add_score(
                scores,
                evidence_by_node,
                node_id,
                20 * len(matched),
                evidence_type='label',
                message='Issue label matches node metadata: ' + ', '.join(matched[:4]),
            )


def rank_issue_candidates(
    analysis: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    max_candidates: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes_by_id = _nodes_by_id(analysis)
    base_scores, warnings = score_nodes(analysis, _string(evidence.get('query')))
    scores = {node_id: score.score for node_id, score in base_scores.items() if node_id in nodes_by_id}
    evidence_by_node: dict[str, list[dict[str, str]]] = {
        node_id: [{'type': 'lexical', 'message': 'Issue title/body/comment text matches graph metadata.'}]
        for node_id in scores
    }

    _apply_file_boosts(nodes_by_id, evidence, scores, evidence_by_node)
    _apply_symbol_boosts(nodes_by_id, evidence, scores, evidence_by_node)
    _apply_label_boosts(nodes_by_id, evidence, scores, evidence_by_node)

    if not scores:
        fallback_ids = sorted((_entrypoint_ids(analysis) | _key_module_ids(analysis)) & set(nodes_by_id))
        for node_id in fallback_ids[:max_candidates]:
            _add_score(
                scores,
                evidence_by_node,
                node_id,
                25,
                evidence_type='fallback',
                message='Issue evidence did not match directly; using entrypoint/key module fallback.',
            )
        warnings.append({'code': 'no_ranked_issue_nodes', 'message': 'Issue evidence did not match graph nodes directly.'})

    ranked_ids = sorted(scores, key=lambda node_id: (-scores[node_id], _node_path(nodes_by_id[node_id]) or '', node_id))
    if ranked_ids and scores[ranked_ids[0]] < 40:
        warnings.append({'code': 'low_confidence_issue_ranking', 'message': 'Issue evidence produced only weak graph matches.'})

    max_score = max([scores[node_id] for node_id in ranked_ids], default=1)
    candidates: list[dict[str, Any]] = []
    for rank, node_id in enumerate(ranked_ids[:max_candidates], start=1):
        node = nodes_by_id[node_id]
        normalized_score = round(min(1.0, max(0.05, scores[node_id] / max_score)), 3)
        evidence_items = evidence_by_node.get(node_id) or [{'type': 'fallback', 'message': 'Deterministic fallback candidate.'}]
        candidates.append(
            {
                'rank': rank,
                'score': normalized_score,
                'raw_score': scores[node_id],
                'node_id': node_id,
                'node': _display_node(node),
                'reason': evidence_items[0]['message'],
                'evidence': evidence_items[:6],
            }
        )
    return candidates, warnings


def _edge_source(edge: Mapping[str, Any]) -> str:
    return _string(edge.get('source'))


def _edge_target(edge: Mapping[str, Any]) -> str:
    return _string(edge.get('target'))


def _edge_kind(edge: Mapping[str, Any]) -> str:
    return _string(edge.get('kind') or edge.get('type'))


def _display_edge(edge: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'source': _edge_source(edge),
        'target': _edge_target(edge),
        'type': _edge_kind(edge),
        'metadata': dict(edge.get('metadata') or {}),
    }


def build_overview_graph_projection(
    analysis: Mapping[str, Any],
    *,
    node_limit: int = 80,
) -> dict[str, Any]:
    nodes_by_id = _nodes_by_id(analysis)
    edges = [cast(Mapping[str, Any], edge) for edge in analysis.get('edges', []) if isinstance(edge, Mapping)]
    entrypoints = [cast(Mapping[str, Any], item) for item in analysis.get('entrypoints', []) if isinstance(item, Mapping)]
    key_modules = [cast(Mapping[str, Any], item) for item in analysis.get('key_modules', []) if isinstance(item, Mapping)]
    cards, scores = select_overview_cards(nodes_by_id, edges, entrypoints, key_modules, node_limit=node_limit)
    selected_ids = [
        str(card.get('node_id'))
        for card in cards
        if card.get('node_id') in nodes_by_id
    ]
    selected_set = set(selected_ids)
    selected_edges = [
        _display_edge(edge)
        for edge in edges
        if _edge_source(edge) in selected_set and _edge_target(edge) in selected_set
    ]
    return {
        'nodes': [
            {
                **_display_node(nodes_by_id[node_id]),
                'overview_category': next((str(card.get('category')) for card in cards if card.get('node_id') == node_id), ''),
                'overview_score': scores.get(node_id, 0),
            }
            for node_id in selected_ids
        ],
        'edges': selected_edges,
        'node_ids': selected_ids,
        'limits': {'node_limit': node_limit, 'node_count': len(selected_ids), 'edge_count': len(selected_edges)},
    }


def _container_ids(node_id: str, node: Mapping[str, Any], nodes_by_id: Mapping[str, Mapping[str, Any]]) -> list[str]:
    containers: list[str] = []
    parent_id = node.get('parent_id') or node.get('parent')
    if isinstance(parent_id, str) and parent_id in nodes_by_id:
        containers.append(parent_id)
    path = _node_path(node)
    if path and path in nodes_by_id and path != node_id:
        containers.append(path)
    if '::' in node_id:
        ancestor = node_id.rsplit('::', 1)[0]
        if ancestor in nodes_by_id:
            containers.append(ancestor)
    result: list[str] = []
    for container_id in containers:
        if container_id not in result:
            result.append(container_id)
    return result


def build_focus_graph_projection(
    analysis: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_focus_nodes: int = 48,
    max_selected_nodes: int = 8,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    nodes_by_id = _nodes_by_id(analysis)
    edges = [cast(Mapping[str, Any], edge) for edge in analysis.get('edges', []) if isinstance(edge, Mapping)]
    max_focus_nodes = max(6, min(max_focus_nodes, 80))
    max_selected_nodes = max(1, min(max_selected_nodes, 20))

    ordered_ids = [
        str(candidate.get('node_id'))
        for candidate in candidates
        if candidate.get('node_id') in nodes_by_id
    ]
    included: list[str] = []

    def include(node_id: str) -> None:
        if node_id in nodes_by_id and node_id not in included and len(included) < max_focus_nodes:
            included.append(node_id)

    for node_id in ordered_ids:
        include(node_id)
        for container_id in _container_ids(node_id, nodes_by_id[node_id], nodes_by_id):
            include(container_id)
        if len(included) >= max_focus_nodes:
            break

    included_set = set(included)
    for edge in edges:
        if len(included) >= max_focus_nodes:
            break
        source = _edge_source(edge)
        target = _edge_target(edge)
        if source in included_set and target in nodes_by_id:
            include(target)
            included_set.add(target)
        if target in included_set and source in nodes_by_id:
            include(source)
            included_set.add(source)

    included_set = set(included)
    projected_edges = [
        _display_edge(edge)
        for edge in edges
        if _edge_source(edge) in included_set and _edge_target(edge) in included_set
    ]
    selected_node_ids = [node_id for node_id in ordered_ids if node_id in included_set][:max_selected_nodes]
    if ordered_ids and not selected_node_ids:
        warnings.append({'code': 'no_focus_highlights', 'message': 'Ranked candidates could not be highlighted in the focus graph.'})
    if len(set(ordered_ids)) > len(selected_node_ids):
        warnings.append({'code': 'focus_graph_truncated', 'message': 'Focus graph node cap excluded some ranked candidates.', 'max_focus_nodes': max_focus_nodes})

    focus_graph = {
        'nodes': [_display_node(nodes_by_id[node_id]) for node_id in included],
        'edges': projected_edges,
        'node_ids': included,
        'highlight_node_ids': selected_node_ids,
        'limits': {
            'max_focus_nodes': max_focus_nodes,
            'node_count': len(included),
            'edge_count': len(projected_edges),
        },
    }
    return focus_graph, selected_node_ids, warnings
