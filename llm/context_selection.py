from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Mapping, cast

from parser.language_registry import LANGUAGE_BY_ID, language_for_path


DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_MAX_CONTEXT_FILES = 4
LINE_PADDING = 3

STOPWORDS = {
    'a', 'an', 'and', 'are', 'at', 'defined', 'does', 'file', 'how', 'in', 'is',
    'of', 'point', 'the', 'to', 'what', 'where', 'which',
}

ENTRYPOINT_HINTS = {'main', 'run', 'cli', 'entry', 'entrypoint', 'app'}
ENTRYPOINT_QUESTION_TOKENS = {'entry', 'entrypoint', 'point', 'main', 'start', '시작', '진입', '엔트리', '실행'}
RELATIONSHIP_EDGE_KINDS = {'calls', 'imports', 'inherits', 'references'}


def _is_supported_source_file(file_path: str) -> bool:
    return language_for_path(file_path, enabled_languages=LANGUAGE_BY_ID.keys()) is not None


def _source_basename_stem(file_path: str) -> str:
    basename = os.path.basename(file_path.lower())
    for suffix in ('.d.ts', '.tsx', '.jsx', '.mts', '.cts', '.mjs', '.cjs', '.ts', '.js', '.py'):
        if basename.endswith(suffix):
            return basename.removesuffix(suffix)
    return os.path.splitext(basename)[0]


@dataclass(frozen=True)
class QaContext:
    context: str
    citations: list[str]
    selected_nodes: list[str]
    context_files: list[str]
    context_summary: dict[str, Any]
    warnings: list[dict[str, Any]]


@dataclass(frozen=True)
class RankedNodeScore:
    node_id: str
    score: int
    file_path: str


def identifier_tokens(value: str) -> set[str]:
    normalized = value.lower().replace('/', '_').replace('.', '_').replace('-', '_').replace(':', '_')
    tokens = {token for token in re.split(r'[^0-9A-Za-z가-힣_]+|_+', normalized) if len(token) > 1}
    if len(normalized) > 1:
        tokens.add(normalized)
    for token in re.findall(r'[A-Za-z0-9_./:-]+|[가-힣]+', value.lower()):
        if len(token) > 1:
            tokens.add(token)
    return tokens


def question_tokens(question: str) -> list[str]:
    raw_tokens = [
        token
        for token in re.findall(r'[A-Za-z0-9_./:-]+|[가-힣]+', question.lower())
        if len(token) > 1
    ]
    expanded_tokens: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        if token in STOPWORDS:
            continue
        candidates = [token]
        if token == 'evaluation':
            candidates.append('eval')
        for candidate in candidates:
            if candidate not in seen:
                expanded_tokens.append(candidate)
                seen.add(candidate)
    return expanded_tokens


def _node_id(node: Mapping[str, Any]) -> str:
    return str(node.get('id', ''))


def _node_file(node: Mapping[str, Any]) -> str | None:
    file_path = node.get('path') or node.get('file')
    if not isinstance(file_path, str) or not file_path:
        return None
    return file_path


def _node_parent_id(node: Mapping[str, Any]) -> str | None:
    parent_id = node.get('parent_id') or node.get('parent')
    if not isinstance(parent_id, str) or not parent_id:
        return None
    return parent_id


def _node_label(node: Mapping[str, Any]) -> str:
    return str(node.get('symbol') or node.get('label') or _node_id(node))


def _edge_kind(edge: Mapping[str, Any]) -> str:
    return str(edge.get('kind') or edge.get('type') or '')


def _nodes_by_id(analysis: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _node_id(cast(Mapping[str, Any], node)): dict(cast(Mapping[str, Any], node))
        for node in analysis.get('nodes', [])
        if _node_id(cast(Mapping[str, Any], node))
    }


def _entrypoint_ids(analysis: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for entrypoint in analysis.get('entrypoints', []):
        entrypoint_id = cast(Mapping[str, Any], entrypoint).get('id')
        if isinstance(entrypoint_id, str):
            ids.add(entrypoint_id)
    return ids


def _key_module_refs(analysis: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    paths: set[str] = set()
    for module in analysis.get('key_modules', []):
        module_mapping = cast(Mapping[str, Any], module)
        module_id = module_mapping.get('id')
        module_path = module_mapping.get('path')
        if isinstance(module_id, str):
            ids.add(module_id)
        if isinstance(module_path, str):
            paths.add(module_path)
    return ids, paths


def _selected_file_exists(analysis: Mapping[str, Any], selected_file_path: str) -> bool:
    file_contents = cast(Mapping[str, str], analysis.get('file_contents', {}))
    if selected_file_path in file_contents:
        return True
    return any(_node_file(cast(Mapping[str, Any], node)) == selected_file_path for node in analysis.get('nodes', []))


def _relationship_proximity(
    analysis: Mapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    selected_node_ids: list[str],
) -> dict[str, int]:
    proximity: dict[str, int] = {}
    selected_set = set(selected_node_ids)
    for node_id in selected_node_ids:
        proximity[node_id] = max(proximity.get(node_id, 0), 140)

        parent_id = _node_parent_id(nodes_by_id[node_id])
        if parent_id and parent_id in nodes_by_id:
            proximity[parent_id] = max(proximity.get(parent_id, 0), 90)

        selected_file = _node_file(nodes_by_id[node_id])
        for candidate_id, candidate in nodes_by_id.items():
            if _node_parent_id(candidate) == node_id:
                proximity[candidate_id] = max(proximity.get(candidate_id, 0), 80)
            if selected_file and candidate_id not in selected_set and _node_file(candidate) == selected_file:
                proximity[candidate_id] = max(proximity.get(candidate_id, 0), 35)

    for edge in analysis.get('edges', []):
        edge_mapping = cast(Mapping[str, Any], edge)
        source = str(edge_mapping.get('source', ''))
        target = str(edge_mapping.get('target', ''))
        kind = _edge_kind(edge_mapping)
        if source in selected_set and target in nodes_by_id:
            proximity[target] = max(proximity.get(target, 0), 85 if kind in RELATIONSHIP_EDGE_KINDS else 60)
        if target in selected_set and source in nodes_by_id:
            proximity[source] = max(proximity.get(source, 0), 75 if kind in RELATIONSHIP_EDGE_KINDS else 55)

    return proximity


def _score_node(
    node: Mapping[str, Any],
    tokens: list[str],
    *,
    entrypoint_ids: set[str],
    key_module_ids: set[str],
    key_module_paths: set[str],
    is_entrypoint_question: bool,
) -> int:
    node_id = _node_id(node)
    file_path = _node_file(node) or ''
    label = _node_label(node)
    label_lower = label.lower()
    basename = os.path.basename(file_path.lower())
    basename_stem = _source_basename_stem(file_path)
    file_tokens = identifier_tokens(file_path)
    label_tokens = identifier_tokens(label)
    id_tokens = identifier_tokens(node_id)

    score = 0
    for token in tokens:
        if token == label_lower or token == basename_stem:
            score += 70
        if token in label_tokens:
            score += 35
        if token in id_tokens:
            score += 25
        if token in file_tokens:
            score += 18
        if token in file_path.lower() or token in node_id.lower():
            score += 6

    if is_entrypoint_question:
        if node_id in entrypoint_ids:
            score += 80
        if basename.startswith('run_') or label_lower == 'main':
            score += 35
        if any(hint in file_tokens for hint in ENTRYPOINT_HINTS):
            score += 18
    elif node_id in entrypoint_ids:
        score += 8

    if node_id in key_module_ids or file_path in key_module_paths:
        score += 12

    return score


def _summary_text_scores(
    analysis: Mapping[str, Any],
    tokens: list[str],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    if not tokens:
        return {}
    summary_scores: dict[str, int] = {}
    summaries = analysis.get('summaries', {})
    if not isinstance(summaries, Mapping):
        return summary_scores

    for summary in summaries.values():
        if not isinstance(summary, Mapping):
            continue
        text = str(summary.get('text', '')).lower()
        if not text or not any(token in text for token in tokens):
            continue
        source_nodes = [node_id for node_id in summary.get('source_nodes', []) if isinstance(node_id, str)]
        source_files = {file_path for file_path in summary.get('source_files', []) if isinstance(file_path, str)}
        for node_id in source_nodes:
            if node_id in nodes_by_id:
                summary_scores[node_id] = max(summary_scores.get(node_id, 0), 30)
        for node_id, node in nodes_by_id.items():
            if _node_file(node) in source_files:
                summary_scores[node_id] = max(summary_scores.get(node_id, 0), 12)
    return summary_scores


def score_nodes(
    analysis: Mapping[str, Any],
    question: str,
    *,
    selected_node_id: str | None = None,
    selected_file_path: str | None = None,
) -> tuple[dict[str, RankedNodeScore], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    nodes_by_id = _nodes_by_id(analysis)
    tokens = question_tokens(question)
    selected_node_ids: list[str] = []
    if selected_node_id:
        if selected_node_id in nodes_by_id:
            selected_node_ids.append(selected_node_id)
        else:
            warnings.append({'code': 'invalid_selected_node', 'node_id': selected_node_id})

    if selected_file_path and not _selected_file_exists(analysis, selected_file_path):
        warnings.append({'code': 'invalid_selected_file', 'path': selected_file_path})

    proximity = _relationship_proximity(analysis, nodes_by_id, selected_node_ids)
    entrypoint_ids = _entrypoint_ids(analysis)
    key_module_ids, key_module_paths = _key_module_refs(analysis)
    is_entrypoint_question = bool(set(tokens) & ENTRYPOINT_QUESTION_TOKENS)
    summary_scores = _summary_text_scores(analysis, tokens, nodes_by_id)

    scores: dict[str, int] = {}
    for node_id, node in nodes_by_id.items():
        score = proximity.get(node_id, 0) + summary_scores.get(node_id, 0)
        if selected_file_path and _node_file(node) == selected_file_path:
            score += 80
        score += _score_node(
            node,
            tokens,
            entrypoint_ids=entrypoint_ids,
            key_module_ids=key_module_ids,
            key_module_paths=key_module_paths,
            is_entrypoint_question=is_entrypoint_question,
        )
        if score > 0:
            scores[node_id] = score

    if not scores:
        for node_id in sorted(entrypoint_ids | key_module_ids):
            if node_id in nodes_by_id:
                scores[node_id] = 10

    scored_nodes = {
        node_id: RankedNodeScore(
            node_id=node_id,
            score=score,
            file_path=_node_file(nodes_by_id[node_id]) or '',
        )
        for node_id, score in scores.items()
        if node_id in nodes_by_id
    }
    return scored_nodes, warnings


def rank_nodes(
    analysis: Mapping[str, Any],
    question: str,
    *,
    selected_node_id: str | None = None,
    selected_file_path: str | None = None,
    max_nodes: int = 24,
) -> tuple[list[str], list[dict[str, Any]]]:
    scores, warnings = score_nodes(
        analysis,
        question,
        selected_node_id=selected_node_id,
        selected_file_path=selected_file_path,
    )
    ranked = sorted(scores, key=lambda node_id: (-scores[node_id].score, scores[node_id].file_path, node_id))
    return ranked[:max_nodes], warnings


def rank_files(analysis: Mapping[str, Any], question: str, max_files: int = DEFAULT_MAX_CONTEXT_FILES) -> list[str]:
    ranked_nodes, _warnings = rank_nodes(analysis, question, max_nodes=max_files * 8)
    nodes_by_id = _nodes_by_id(analysis)
    ranked_files: list[str] = []
    for node_id in ranked_nodes:
        file_path = _node_file(nodes_by_id[node_id])
        if file_path and _is_supported_source_file(file_path) and file_path not in ranked_files:
            ranked_files.append(file_path)
        if len(ranked_files) >= max_files:
            return ranked_files
    if ranked_files:
        return ranked_files

    fallback_files = sorted({
        _node_file(cast(Mapping[str, Any], node))
        for node in analysis.get('nodes', [])
        if _is_supported_source_file(_node_file(cast(Mapping[str, Any], node)) or '')
    })
    for file_path in fallback_files:
        if file_path and file_path not in ranked_files:
            ranked_files.append(file_path)
        if len(ranked_files) >= max_files:
            break
    return ranked_files


def _line_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _line_window(node: Mapping[str, Any], total_lines: int) -> tuple[int, int] | None:
    start_line = _line_or_none(node.get('start_line'))
    end_line = _line_or_none(node.get('end_line'))
    if start_line is None or end_line is None or total_lines < 1 or start_line > total_lines:
        return None
    start = max(1, start_line - LINE_PADDING)
    end = min(total_lines, end_line + LINE_PADDING)
    if end < start:
        return None
    return start, end


def _merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _format_lines(lines: list[str], start: int, end: int) -> str:
    return '\n'.join(f'{line_number:>4}: {lines[line_number - 1]}' for line_number in range(start, end + 1))


def _build_file_section(file_path: str, code: str, nodes: list[Mapping[str, Any]]) -> tuple[str, list[str]]:
    node_ids = [_node_id(node) for node in nodes if _node_id(node)]
    header_lines = [f'# 파일: {file_path}']
    if node_ids:
        header_lines.append('# 관련 노드: ' + ', '.join(node_ids[:8]))

    lines = code.splitlines()
    windows = _merge_windows([window for node in nodes if (window := _line_window(node, len(lines))) is not None])
    if not windows:
        return '\n'.join(header_lines) + '\n' + code, node_ids

    excerpts = []
    for start, end in windows:
        excerpts.append(f'## {file_path}:{start}-{end}\n{_format_lines(lines, start, end)}')
    return '\n'.join(header_lines) + '\n' + '\n\n'.join(excerpts), node_ids


def _append_section(
    sections: list[str],
    section: str,
    *,
    used_chars: int,
    max_chars: int,
) -> tuple[int, bool, bool]:
    if used_chars + len(section) <= max_chars:
        sections.append(section)
        return used_chars + len(section), True, False
    if used_chars == 0:
        sections.append(section[:max_chars])
        return max_chars, True, True
    return used_chars, False, True


def build_context_for_files(
    analysis: Mapping[str, Any],
    files: list[str],
    *,
    node_ids: list[str] | None = None,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> QaContext:
    sections: list[str] = []
    included_files: list[str] = []
    included_nodes: list[str] = []
    warnings: list[dict[str, Any]] = []
    used_chars = 0
    truncated = False
    file_contents = cast(Mapping[str, str], analysis.get('file_contents', {}))
    nodes_by_id = _nodes_by_id(analysis)
    wanted_node_ids = set(node_ids or [])

    for file_path in files:
        code = file_contents.get(file_path)
        if not code:
            continue

        nodes_for_file = [
            node
            for node_id, node in nodes_by_id.items()
            if _node_file(node) == file_path and (not wanted_node_ids or node_id in wanted_node_ids)
        ]
        section_body, section_node_ids = _build_file_section(file_path, code, nodes_for_file)
        section = '\n\n' + section_body
        used_chars, included, did_truncate = _append_section(sections, section, used_chars=used_chars, max_chars=max_chars)
        truncated = truncated or did_truncate
        if not included:
            break
        included_files.append(file_path)
        included_nodes.extend(node_id for node_id in section_node_ids if node_id not in included_nodes)
        if used_chars >= max_chars:
            break

    if truncated:
        warnings.append({'code': 'context_truncated', 'max_context_chars': max_chars})

    return QaContext(
        context=''.join(sections),
        citations=included_files,
        selected_nodes=included_nodes,
        context_files=included_files,
        context_summary={
            'strategy': 'file_list',
            'file_count': len(included_files),
            'node_count': len(included_nodes),
            'max_context_files': len(files),
            'max_context_chars': max_chars,
            'truncated': truncated,
        },
        warnings=warnings,
    )


def build_qa_context(
    analysis: Mapping[str, Any],
    question: str,
    *,
    selected_node_id: str | None = None,
    selected_file_path: str | None = None,
    max_context_files: int = DEFAULT_MAX_CONTEXT_FILES,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> QaContext:
    max_context_files = max(1, min(max_context_files, 10))
    ranked_node_ids, warnings = rank_nodes(
        analysis,
        question,
        selected_node_id=selected_node_id,
        selected_file_path=selected_file_path,
    )
    nodes_by_id = _nodes_by_id(analysis)
    files: list[str] = []
    if selected_file_path and _selected_file_exists(analysis, selected_file_path):
        files.append(selected_file_path)

    for node_id in ranked_node_ids:
        file_path = _node_file(nodes_by_id[node_id])
        if file_path and _is_supported_source_file(file_path) and file_path not in files:
            files.append(file_path)
        if len(files) >= max_context_files:
            break

    if len(files) < max_context_files:
        for file_path in rank_files(analysis, question, max_files=max_context_files):
            if file_path not in files:
                files.append(file_path)
            if len(files) >= max_context_files:
                break

    context = build_context_for_files(
        analysis,
        files[:max_context_files],
        node_ids=ranked_node_ids,
        max_chars=max_chars,
    )
    all_warnings = [*warnings, *context.warnings]
    summary = {
        **context.context_summary,
        'strategy': 'selected_node' if selected_node_id or selected_file_path else 'ranked',
        'question_tokens': question_tokens(question)[:20],
        'max_context_files': max_context_files,
    }
    return QaContext(
        context=context.context,
        citations=context.citations,
        selected_nodes=context.selected_nodes,
        context_files=context.context_files,
        context_summary=summary,
        warnings=all_warnings,
    )
