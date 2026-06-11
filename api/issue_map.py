from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, cast
import html
import os
import re

from api.readme_svg import select_overview_cards
from llm.context_selection import identifier_tokens, score_nodes


SOURCE_EXTENSIONS = ('py', 'jsx', 'tsx', 'mjs', 'cjs', 'mts', 'cts', 'js', 'ts')
SOURCE_EXTENSION_RE = r'(?:jsx|tsx|mjs|cjs|mts|cts|py|js|ts)(?=$|[^A-Za-z0-9_./-])'
FILE_PATH_RE = re.compile(rf'(?P<path>(?:[A-Za-z0-9_@.-]+/)*[A-Za-z0-9_@.-]+\.{SOURCE_EXTENSION_RE})(?::(?P<line>\d+))?(?::(?P<column>\d+))?')
PY_STACK_RE = re.compile(r'File "(?P<path>[^"]+\.py)", line (?P<line>\d+), in (?P<symbol>[A-Za-z_][A-Za-z0-9_]*)')
PYTEST_FRAME_RE = re.compile(r'(?P<path>(?:[A-Za-z0-9_@.-]+/)*[A-Za-z0-9_@.-]+\.py):(?P<line>\d+)(?::in\s+(?P<symbol>[A-Za-z_][A-Za-z0-9_]*))?')
JS_STACK_PAREN_RE = re.compile(rf'\bat\s+(?P<symbol>[A-Za-z_$][A-Za-z0-9_.$]*)\s+\((?P<path>[^()\s]+\.{SOURCE_EXTENSION_RE}):(?P<line>\d+):(?P<column>\d+)\)')
JS_STACK_BARE_RE = re.compile(rf'\bat\s+(?P<path>[^()\s]+\.{SOURCE_EXTENSION_RE}):(?P<line>\d+):(?P<column>\d+)')
BACKTICK_RE = re.compile(r'`([^`]{2,120})`')
QUOTED_STRING_RE = re.compile(r'(?P<quote>["\'])(?P<text>[^"\']{4,160})(?P=quote)')
CALL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(')
ERROR_LINE_RE = re.compile(r'(?i)\b(error|exception|traceback|failed|failure|timeout|crash|invalid)\b')
SYMBOL_TOKEN_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
EXCEPTION_RE = re.compile(r'\b(?P<class>(?:[A-Z][A-Za-z0-9_.]*(?:Error|Exception|Timeout|Failure)|Exception|Timeout|Failure))\b(?::\s*(?P<message>[^\n`]{3,180}))?')
ROUTE_RE = re.compile(r'(?<![\w.-])(?P<route>/(?:[A-Za-z0-9_.{}:-]+/)*[A-Za-z0-9_.{}:-]*/?)(?![\w.-])')
CONFIG_NAME_RE = re.compile(r'\b(?P<name>[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b')
TEST_NAME_RE = re.compile(r'\b(?P<pyname>test_[A-Za-z0-9_]+)\b|\b(?:it|test|describe)\s*\(\s*(?P<quote>["\'])(?P<jsname>[^"\']{2,120})(?P=quote)')
TEST_KEYWORD_RE = re.compile(r'(?i)\b(pytest|unittest|assertion|assert|failing test|test failure|tests? failed|jest|vitest|mocha|playwright|ts-jest|tsx|tsc)\b')
TEST_PATH_PART_RE = re.compile(r'(^|/)(__tests__/|tests?/|test_[^/]+\.py$|[^/]+_test\.py$|tests\.py$|[^/]+\.(?:test|spec)\.(?:js|jsx|ts|tsx)$)')


def _is_safe_issue_path(path: str) -> bool:
    if not path or path.startswith('/') or '\\' in path or '\x00' in path:
        return False
    candidate = PurePosixPath(path)
    return not candidate.is_absolute() and all(part not in {'', '.', '..'} for part in candidate.parts)


def _is_source_path(path: str) -> bool:
    return _is_safe_issue_path(path) and path.rsplit('.', 1)[-1] in SOURCE_EXTENSIONS


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


def _dedupe_limited(items: list[dict[str, Any]], limit: int, *keys: str) -> list[dict[str, Any]]:
    return _dedupe(items, *keys)[:limit]


def _extract_file_mentions(source: str, text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for match in FILE_PATH_RE.finditer(text):
        path = match.group('path')
        if not _is_source_path(path):
            continue
        line = match.group('line')
        column = match.group('column')
        mentions.append(
            {
                'path': path,
                'line': int(line) if line else None,
                'column': int(column) if column else None,
                'source': source,
                'confidence': 1.0 if '/' in path else 0.75,
            }
        )
    return mentions


def _extract_stack_frames(source: str, text: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for regex in (PY_STACK_RE, PYTEST_FRAME_RE, JS_STACK_PAREN_RE, JS_STACK_BARE_RE):
        for match in regex.finditer(text):
            path = match.group('path')
            if not _is_source_path(path):
                continue
            frames.append(
                {
                    'path': path,
                    'line': int(match.group('line')),
                    'column': int(match.groupdict().get('column') or 0) or None,
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


def _extract_exception_mentions(source: str, text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for match in EXCEPTION_RE.finditer(text):
        exception_class = match.group('class')
        message = (match.group('message') or '').strip()
        mentions.append(
            {
                'class': exception_class,
                'message': message,
                'text': f'{exception_class}: {message}' if message else exception_class,
                'source': source,
                'confidence': 0.95 if message else 0.8,
            }
        )
    return mentions


def _extract_route_mentions(source: str, text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for match in ROUTE_RE.finditer(text):
        route = match.group('route')
        if route in {'/', '//'} or _is_source_path(route.lstrip('/')):
            continue
        if not any(part.isalpha() for part in route):
            continue
        mentions.append({'route': route, 'source': source, 'confidence': 0.9})
    return mentions


def _extract_config_mentions(source: str, text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for match in CONFIG_NAME_RE.finditer(text):
        name = match.group('name')
        if len(name) < 4:
            continue
        mentions.append({'name': name, 'source': source, 'confidence': 0.85})
    return mentions


def _extract_quoted_strings(source: str, text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for regex in (BACKTICK_RE, QUOTED_STRING_RE):
        for match in regex.finditer(text):
            quoted = (match.group('text') if 'text' in match.groupdict() else match.group(1)).strip()
            if not quoted or FILE_PATH_RE.fullmatch(quoted) or SYMBOL_TOKEN_RE.fullmatch(quoted.removesuffix('()')):
                continue
            if len(quoted) < 4 or len(identifier_tokens(quoted)) > 12:
                continue
            mentions.append({'text': quoted, 'source': source, 'confidence': 0.75})
    return mentions


def _extract_test_mentions(source: str, text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for match in TEST_NAME_RE.finditer(text):
        name = match.groupdict().get('pyname') or match.groupdict().get('jsname')
        if name:
            mentions.append({'name': name, 'source': source, 'confidence': 0.95})
    for match in FILE_PATH_RE.finditer(text):
        path = match.group('path')
        if _is_test_path(path):
            mentions.append({'path': path, 'source': source, 'confidence': 0.9})
    if TEST_KEYWORD_RE.search(text):
        mentions.append({'keyword': 'test_failure', 'source': source, 'confidence': 0.7})
    return mentions


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
    exception_mentions: list[dict[str, Any]] = []
    route_mentions: list[dict[str, Any]] = []
    config_mentions: list[dict[str, Any]] = []
    test_mentions: list[dict[str, Any]] = []
    quoted_strings: list[dict[str, Any]] = []

    for source, text in texts:
        file_mentions.extend(_extract_file_mentions(source, text))
        stack_frames.extend(_extract_stack_frames(source, text))
        symbol_mentions.extend(_extract_symbol_mentions(source, text))
        quoted_errors.extend(_extract_error_phrases(source, text))
        exception_mentions.extend(_extract_exception_mentions(source, text))
        route_mentions.extend(_extract_route_mentions(source, text))
        config_mentions.extend(_extract_config_mentions(source, text))
        test_mentions.extend(_extract_test_mentions(source, text))
        quoted_strings.extend(_extract_quoted_strings(source, text))

    for frame in stack_frames:
        file_mentions.append(
            {
                'path': frame['path'],
                'line': frame['line'],
                'column': frame.get('column'),
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
    bounded_exception_mentions = _dedupe_limited(exception_mentions, 20, 'class', 'message', 'source')
    bounded_route_mentions = _dedupe_limited(route_mentions, 40, 'route', 'source')
    bounded_config_mentions = _dedupe_limited(config_mentions, 40, 'name', 'source')
    bounded_test_mentions = _dedupe_limited(test_mentions, 40, 'name', 'path', 'keyword', 'source')
    bounded_quoted_strings = _dedupe_limited(quoted_strings, 20, 'text', 'source')
    query_parts = [
        _string(issue.get('title')),
        _string(issue.get('body') or issue.get('body_excerpt')),
        ' '.join(label['name'] for label in labels),
        ' '.join(comment.get('body', '') for comment in comments or [] if isinstance(comment.get('body'), str)),
        ' '.join(item['path'] for item in file_mentions),
        ' '.join(item['symbol'] for item in symbol_mentions),
        ' '.join(item.get('text', '') for item in bounded_exception_mentions),
        ' '.join(item.get('route', '') for item in bounded_route_mentions),
        ' '.join(item.get('name', '') for item in bounded_config_mentions),
        ' '.join(item.get('name', '') or item.get('path', '') or item.get('keyword', '') for item in bounded_test_mentions),
        ' '.join(item.get('text', '') for item in bounded_quoted_strings),
    ]
    return {
        'query': ' '.join(part for part in query_parts if part),
        'file_mentions': _dedupe(file_mentions, 'path', 'line', 'source'),
        'symbol_mentions': _dedupe(symbol_mentions, 'symbol', 'source'),
        'stack_frames': _dedupe(stack_frames, 'path', 'line', 'symbol', 'source'),
        'quoted_errors': _dedupe(quoted_errors, 'text', 'source'),
        'exception_mentions': bounded_exception_mentions,
        'route_mentions': bounded_route_mentions,
        'config_mentions': bounded_config_mentions,
        'test_mentions': bounded_test_mentions,
        'quoted_strings': bounded_quoted_strings,
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


def _analysis_file_contents(analysis: Mapping[str, Any]) -> Mapping[str, Any]:
    file_contents = analysis.get('file_contents')
    return file_contents if isinstance(file_contents, Mapping) else {}


def _is_source_backed_node(analysis: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
    path = _node_path(node)
    if not path or not _is_source_path(path):
        return False
    file_contents = _analysis_file_contents(analysis)
    return not file_contents or path in file_contents


def _node_search_text(node_id: str, node: Mapping[str, Any], file_contents: Mapping[str, Any]) -> str:
    node_path = _node_path(node) or ''
    source_text = _string(file_contents.get(node_path))
    metadata = node.get('metadata') if isinstance(node.get('metadata'), Mapping) else {}
    metadata_text = ' '.join(_string(value) for value in metadata.values()) if isinstance(metadata, Mapping) else ''
    return '\n'.join([node_id, _node_label(node), _node_symbol(node), node_path, metadata_text, source_text]).lower()


def _is_test_path(path: str | None) -> bool:
    return bool(path and TEST_PATH_PART_RE.search(path))


def _is_test_node(node: Mapping[str, Any]) -> bool:
    return _is_test_path(_node_path(node))


def _issue_is_test_specific(evidence: Mapping[str, Any]) -> bool:
    if _safe_evidence_list(evidence.get('test_mentions')):
        return True
    query = _string(evidence.get('query'))
    return bool(TEST_KEYWORD_RE.search(query) or TEST_NAME_RE.search(query) or any(_is_test_path(match.group('path')) for match in FILE_PATH_RE.finditer(query)))


def _safe_evidence_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
    query_tokens = identifier_tokens(_string(evidence.get('query')))
    for mention in evidence.get('file_mentions', []):
        if not isinstance(mention, Mapping):
            continue
        mentioned_path = _string(mention.get('path'))
        mentioned_line = mention.get('line') if isinstance(mention.get('line'), int) else None
        for node_id, node in nodes_by_id.items():
            node_path = _node_path(node)
            exact_path_match = node_path == mentioned_path or node_id == mentioned_path
            basename_match = bool(node_path and '/' not in mentioned_path and file_basename(node_path) == mentioned_path)
            if not exact_path_match and not basename_match:
                continue
            if exact_path_match and _line_matches(node, mentioned_line):
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    850,
                    evidence_type='stack_frame',
                    message=f'{mentioned_path}:{mentioned_line} stack trace line matches this node.',
                )
            elif _node_kind(node) in {'function', 'method', 'class'}:
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    180 if exact_path_match else 80,
                    evidence_type='file_path' if exact_path_match else 'file_basename',
                    message=f'Issue mentions {"file path" if exact_path_match else "bare filename"} {mentioned_path}.',
                )
                node_tokens = identifier_tokens(' '.join([_node_label(node), _node_symbol(node), _final_id_segment(node_id)]))
                matched_context = sorted(node_tokens & query_tokens)
                if exact_path_match and matched_context:
                    _add_score(
                        scores,
                        evidence_by_node,
                        node_id,
                        260,
                        evidence_type='file_symbol_context',
                        message='Issue mentions this file and matching node text: ' + ', '.join(matched_context[:4]),
                    )
            else:
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    240 if exact_path_match else 70,
                    evidence_type='file_path' if exact_path_match else 'file_basename',
                    message=f'Issue directly mentions {"file node" if exact_path_match else "bare filename"} {mentioned_path}.',
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


def _apply_exception_boosts(
    analysis: Mapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
    scores: dict[str, int],
    evidence_by_node: dict[str, list[dict[str, str]]],
) -> None:
    file_contents = _analysis_file_contents(analysis)
    for mention in _safe_evidence_list(evidence.get('exception_mentions')):
        if not isinstance(mention, Mapping):
            continue
        exception_class = _string(mention.get('class')).lower()
        message = _string(mention.get('message')).lower()
        if not exception_class and not message:
            continue
        for node_id, node in nodes_by_id.items():
            node_text = _node_search_text(node_id, node, file_contents)
            if message and message in node_text:
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    180,
                    evidence_type='exception_message',
                    message=f'Issue exception message appears in {_node_path(node) or node_id}.',
                )
            elif exception_class and exception_class in node_text:
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    90,
                    evidence_type='exception_class',
                    message=f'Issue mentions exception {mention.get("class")}.',
                )


def _apply_route_boosts(
    analysis: Mapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
    scores: dict[str, int],
    evidence_by_node: dict[str, list[dict[str, str]]],
) -> None:
    file_contents = _analysis_file_contents(analysis)
    for mention in _safe_evidence_list(evidence.get('route_mentions')):
        if not isinstance(mention, Mapping):
            continue
        route = _string(mention.get('route'))
        if not route:
            continue
        route_lower = route.lower()
        route_tokens = identifier_tokens(route)
        for node_id, node in nodes_by_id.items():
            node_path = _node_path(node) or ''
            node_text = _node_search_text(node_id, node, file_contents)
            path_tokens = identifier_tokens(node_path)
            is_route_owner = route_lower in node_text
            is_api_route_module = bool(route_tokens & path_tokens) and any(token in path_tokens for token in {'api', 'url', 'urls', 'view', 'views', 'router'})
            if is_route_owner:
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    180 if _node_kind(node) in {'function', 'method', 'class'} else 110,
                    evidence_type='route',
                    message=f'Issue route {route} appears in this file or node context.',
                )
            elif is_api_route_module:
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    70,
                    evidence_type='route',
                    message=f'Issue route {route} matches API/view routing metadata.',
                )


def _apply_config_boosts(
    analysis: Mapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
    scores: dict[str, int],
    evidence_by_node: dict[str, list[dict[str, str]]],
) -> None:
    file_contents = _analysis_file_contents(analysis)
    for mention in _safe_evidence_list(evidence.get('config_mentions')):
        if not isinstance(mention, Mapping):
            continue
        name = _string(mention.get('name'))
        if not name:
            continue
        lowered = name.lower()
        for node_id, node in nodes_by_id.items():
            node_path = _node_path(node) or ''
            node_text = _node_search_text(node_id, node, file_contents)
            path_tokens = identifier_tokens(node_path)
            if lowered in node_text:
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    170,
                    evidence_type='config',
                    message=f'Issue mentions config/env name {name} found in this context.',
                )
            elif any(token in path_tokens for token in {'config', 'settings', 'env'}):
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    60,
                    evidence_type='config',
                    message=f'Issue mentions config/env name {name}; this is a config-related module.',
                )


def _apply_quoted_string_boosts(
    analysis: Mapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
    scores: dict[str, int],
    evidence_by_node: dict[str, list[dict[str, str]]],
) -> None:
    file_contents = _analysis_file_contents(analysis)
    for key in ('quoted_errors', 'quoted_strings'):
        for mention in _safe_evidence_list(evidence.get(key)):
            if not isinstance(mention, Mapping):
                continue
            text = _string(mention.get('text')).strip()
            if len(text) < 4:
                continue
            lowered = text.lower()
            for node_id, node in nodes_by_id.items():
                node_text = _node_search_text(node_id, node, file_contents)
                if lowered not in node_text:
                    continue
                _add_score(
                    scores,
                    evidence_by_node,
                    node_id,
                    150 if key == 'quoted_errors' else 110,
                    evidence_type='quoted_error' if key == 'quoted_errors' else 'quoted_string',
                    message=f'Issue quoted text appears in {_node_path(node) or node_id}.',
                )


def _apply_test_boosts_or_penalties(
    analysis: Mapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
    scores: dict[str, int],
    evidence_by_node: dict[str, list[dict[str, str]]],
) -> None:
    test_specific = _issue_is_test_specific(evidence)
    test_names = {
        _string(mention.get('name')).lower()
        for mention in _safe_evidence_list(evidence.get('test_mentions'))
        if isinstance(mention, Mapping) and mention.get('name')
    }
    test_paths = {
        _string(mention.get('path'))
        for mention in _safe_evidence_list(evidence.get('test_mentions'))
        if isinstance(mention, Mapping) and mention.get('path')
    }
    matched_test_ids: set[str] = set()
    for node_id, node in nodes_by_id.items():
        if not _is_test_node(node):
            continue
        if not test_specific:
            current_score = scores.get(node_id)
            if current_score is not None:
                scores[node_id] = max(1, int(current_score * 0.6))
                evidence_by_node.setdefault(node_id, []).append({'type': 'test_penalty', 'message': 'Test node de-prioritized because issue is not test-specific.'})
            continue
        node_path = _node_path(node) or ''
        final_segment = _final_id_segment(node_id).lower()
        if node_path in test_paths or final_segment in test_names or _node_label(node).lower() in test_names:
            matched_test_ids.add(node_id)
            _add_score(
                scores,
                evidence_by_node,
                node_id,
                220,
                evidence_type='test',
                message='Issue explicitly names this failing test context.',
            )
    if not matched_test_ids:
        return
    for edge in analysis.get('edges', []):
        if not isinstance(edge, Mapping):
            continue
        source = _edge_source(edge)
        target = _edge_target(edge)
        if source in matched_test_ids and target in nodes_by_id:
            production_id = target
        elif target in matched_test_ids and source in nodes_by_id:
            production_id = source
        else:
            continue
        production_node = nodes_by_id[production_id]
        if _is_test_node(production_node):
            continue
        _add_score(
            scores,
            evidence_by_node,
            production_id,
            160,
            evidence_type='test_related_production',
            message='Failing test is connected to this production node in the analysis graph.',
        )


def _weak_issue_evidence(scores: Mapping[str, int], evidence_by_node: Mapping[str, list[dict[str, str]]], ranked_ids: Sequence[str], nodes_by_id: Mapping[str, Mapping[str, Any]]) -> bool:
    if not ranked_ids:
        return False
    top_id = ranked_ids[0]
    top_score = scores.get(top_id, 0)
    evidence_types = {item.get('type') for item in evidence_by_node.get(top_id, [])}
    if evidence_types and evidence_types.issubset({'label', 'fallback', 'lexical', 'test_penalty'}):
        return True
    if top_score < 40:
        return True
    top_kinds = [_node_kind(nodes_by_id[node_id]) for node_id in ranked_ids[: min(3, len(ranked_ids))]]
    return bool(top_kinds) and all(kind in {'directory', 'file'} for kind in top_kinds)


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
    _apply_exception_boosts(analysis, nodes_by_id, evidence, scores, evidence_by_node)
    _apply_route_boosts(analysis, nodes_by_id, evidence, scores, evidence_by_node)
    _apply_config_boosts(analysis, nodes_by_id, evidence, scores, evidence_by_node)
    _apply_quoted_string_boosts(analysis, nodes_by_id, evidence, scores, evidence_by_node)
    _apply_test_boosts_or_penalties(analysis, nodes_by_id, evidence, scores, evidence_by_node)

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
    source_ranked_ids = [node_id for node_id in ranked_ids if _is_source_backed_node(analysis, nodes_by_id[node_id])]
    if source_ranked_ids:
        ranked_ids = source_ranked_ids
    if ranked_ids and scores[ranked_ids[0]] < 40:
        warnings.append({'code': 'low_confidence_issue_ranking', 'message': 'Issue evidence produced only weak graph matches.'})
    if _weak_issue_evidence(scores, evidence_by_node, ranked_ids, nodes_by_id):
        warnings.append({'code': 'weak_issue_evidence', 'message': 'Top issue match is based on weak or generic evidence.'})

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


def _node_window(node: Mapping[str, Any], total_lines: int, *, padding: int = 3) -> tuple[int, int] | None:
    start, end = _node_line_range(node)
    if start is None or end is None or total_lines < 1:
        return None
    return max(1, start - padding), min(total_lines, end + padding)


def _format_window(lines: list[str], start: int, end: int) -> str:
    return '\n'.join(f'{line_number:>4}: {lines[line_number - 1]}' for line_number in range(start, end + 1))


def build_code_context(
    analysis: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_context_files: int = 4,
    max_context_chars: int = 12000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    max_context_files = max(1, min(max_context_files, 10))
    max_context_chars = max(1000, min(max_context_chars, 40000))
    nodes_by_id = _nodes_by_id(analysis)
    file_contents = analysis.get('file_contents')
    if not isinstance(file_contents, Mapping):
        file_contents = {}

    files: list[str] = []
    nodes_by_file: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        node_id = str(candidate.get('node_id'))
        node = nodes_by_id.get(node_id)
        if node is None:
            continue
        path = _node_path(node)
        if not path or path not in file_contents:
            continue
        if path not in files:
            files.append(path)
        nodes_by_file.setdefault(path, []).append(node)
        if len(files) >= max_context_files:
            break

    used_chars = 0
    truncated = False
    context_files: list[dict[str, Any]] = []
    for path in files:
        code = _string(file_contents.get(path))
        lines = code.splitlines()
        windows = [
            window
            for node in nodes_by_file.get(path, [])
            if (window := _node_window(node, len(lines))) is not None
        ]
        if windows:
            excerpts = []
            for start, end in sorted(set(windows)):
                excerpts.append({'start_line': start, 'end_line': end, 'text': _format_window(lines, start, end)})
        else:
            excerpts = [{'start_line': 1, 'end_line': min(len(lines), 80), 'text': '\n'.join(lines[:80])}]

        file_text = '\n\n'.join(excerpt['text'] for excerpt in excerpts)
        remaining = max_context_chars - used_chars
        if remaining <= 0:
            truncated = True
            break
        file_truncated = len(file_text) > remaining
        if file_truncated:
            file_text = file_text[:remaining]
            truncated = True
        used_chars += len(file_text)
        context_files.append(
            {
                'path': path,
                'node_ids': [_node_id(node) for node in nodes_by_file.get(path, []) if _node_id(node)],
                'excerpts': excerpts,
                'text': file_text,
                'truncated': file_truncated,
            }
        )
        if used_chars >= max_context_chars:
            truncated = True
            break

    if truncated or len(files) > len(context_files):
        warnings.append(
            {
                'code': 'code_context_truncated',
                'message': 'Code context가 설정된 파일/문자 한도 내로 잘렸습니다.',
                'max_context_files': max_context_files,
                'max_context_chars': max_context_chars,
            }
        )
    return {
        'files': context_files,
        'file_count': len(context_files),
        'max_context_files': max_context_files,
        'max_context_chars': max_context_chars,
        'truncated': bool(truncated),
    }, warnings


def _inert_text(value: Any, *, max_chars: int = 600) -> str:
    text = _string(value).replace('\x00', '').strip()
    text = html.escape(text, quote=False)
    return text[:max_chars]


def _focus_node_paths(focus_graph: Mapping[str, Any]) -> dict[str, str | None]:
    node_paths: dict[str, str | None] = {}
    for node in focus_graph.get('nodes') or []:
        if not isinstance(node, Mapping):
            continue
        node_id = _string(node.get('id'))
        if not node_id:
            continue
        path = node.get('path')
        node_paths[node_id] = path if isinstance(path, str) and path else None
    return node_paths


def _allowed_explanation_paths(focus_graph: Mapping[str, Any], code_context: Mapping[str, Any]) -> set[str]:
    paths = {path for path in _focus_node_paths(focus_graph).values() if path}
    for file_context in code_context.get('files') or []:
        if not isinstance(file_context, Mapping):
            continue
        path = file_context.get('path')
        if isinstance(path, str) and path:
            paths.add(path)
    return paths


def _float_in_range(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(min(1.0, max(0.0, number)), 3)


def _confidence_level(score: float) -> str:
    if score >= 0.75:
        return 'high'
    if score >= 0.35:
        return 'medium'
    return 'low'


def _sanitize_llm_hypotheses(
    value: Any,
    node_paths: Mapping[str, str | None],
    fallback: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    if not isinstance(value, list):
        return fallback, 1, 0

    result: list[dict[str, Any]] = []
    dropped = 0
    rewritten = 0
    for item in value[:8]:
        if not isinstance(item, Mapping):
            dropped += 1
            continue
        node_id = _string(item.get('node_id'))
        if node_id not in node_paths:
            dropped += 1
            continue
        kind = _inert_text(item.get('kind') or ('likely_origin' if not result else 'related_area'), max_chars=40)
        confidence = _float_in_range(item.get('confidence'), default=0.5)
        rationale = _inert_text(item.get('rationale') or item.get('reason') or '', max_chars=800)
        if not rationale:
            rationale = 'Explanation output did not include a usable rationale.'
            rewritten += 1
        result.append(
            {
                'kind': kind,
                'node_id': node_id,
                'confidence': confidence,
                'rationale': rationale,
            }
        )

    if not result:
        return fallback, dropped + 1, rewritten
    return result, dropped, rewritten


def _sanitize_llm_investigation_path(
    value: Any,
    node_paths: Mapping[str, str | None],
    allowed_paths: set[str],
    fallback: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    if not isinstance(value, list):
        return fallback, 1, 0

    result: list[dict[str, Any]] = []
    dropped = 0
    rewritten = 0
    for item in value[:10]:
        if not isinstance(item, Mapping):
            dropped += 1
            continue
        node_id = _string(item.get('node_id'))
        if node_id not in node_paths:
            dropped += 1
            continue

        path = item.get('path')
        if not isinstance(path, str) or not path:
            path = node_paths[node_id]
        elif path not in allowed_paths:
            path = node_paths[node_id]
            rewritten += 1

        result.append(
            {
                'step': len(result) + 1,
                'node_id': node_id,
                'path': path,
                'action': _inert_text(item.get('action') or 'inspect', max_chars=80),
                'why': _inert_text(item.get('why') or item.get('rationale') or '', max_chars=800),
            }
        )

    if not result:
        return fallback, dropped + 1, rewritten
    return result, dropped, rewritten


def _sanitize_llm_confidence(
    value: Any,
    fallback: dict[str, Any],
    *,
    source: str = 'llm',
) -> tuple[dict[str, Any], int, int]:
    if not isinstance(value, Mapping):
        return fallback, 1, 0

    fallback_score = _float_in_range(fallback.get('score'), default=0.0)
    score = _float_in_range(value.get('score'), default=fallback_score)
    level = _inert_text(value.get('level') or '', max_chars=20).lower()
    rewritten = 0
    if level not in {'high', 'medium', 'low'}:
        level = _confidence_level(score)
        rewritten += 1

    reasons = [
        _inert_text(reason, max_chars=500)
        for reason in (value.get('reasons') or [])
        if _inert_text(reason, max_chars=500)
    ][:5]
    if not reasons:
        reasons = [_inert_text(reason, max_chars=500) for reason in fallback.get('reasons', []) if _inert_text(reason, max_chars=500)][:5]
        rewritten += 1

    return (
        {
            'level': level,
            'score': score,
            'reasons': reasons,
            'warning_codes': list(fallback.get('warning_codes') or []),
            'source': source,
        },
        0,
        rewritten,
    )


def sanitize_issue_explanation_output(
    output: Mapping[str, Any],
    *,
    focus_graph: Mapping[str, Any],
    code_context: Mapping[str, Any],
    fallback_hypotheses: list[dict[str, Any]],
    fallback_investigation_path: list[dict[str, Any]],
    fallback_confidence: dict[str, Any],
    source: str = 'llm',
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    node_paths = _focus_node_paths(focus_graph)
    allowed_paths = _allowed_explanation_paths(focus_graph, code_context)
    warnings: list[dict[str, Any]] = []

    hypotheses, dropped_hypotheses, rewritten_hypotheses = _sanitize_llm_hypotheses(
        output.get('hypotheses'),
        node_paths,
        fallback_hypotheses,
    )
    investigation_path, dropped_steps, rewritten_steps = _sanitize_llm_investigation_path(
        output.get('investigation_path'),
        node_paths,
        allowed_paths,
        fallback_investigation_path,
    )
    confidence, dropped_confidence, rewritten_confidence = _sanitize_llm_confidence(
        output.get('confidence'),
        fallback_confidence,
        source=source,
    )

    dropped = dropped_hypotheses + dropped_steps + dropped_confidence
    rewritten = rewritten_hypotheses + rewritten_steps + rewritten_confidence
    if dropped or rewritten:
        warnings.append(
            {
                'code': 'llm_output_sanitized',
                'message': 'Issue explanation에서 허용되지 않은 node/path 또는 잘못된 필드를 제거하거나 보정했습니다.',
                'dropped_items': dropped,
                'rewritten_items': rewritten,
            }
        )

    return {
        'hypotheses': hypotheses,
        'investigation_path': investigation_path,
        'confidence': confidence,
    }, warnings
