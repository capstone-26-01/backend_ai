from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import os
import re


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
