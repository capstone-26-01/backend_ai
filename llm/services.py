from __future__ import annotations

import os
import re
from typing import Any, cast

from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

STOPWORDS = {
    'a', 'an', 'and', 'are', 'at', 'defined', 'does', 'file', 'how', 'in', 'is',
    'of', 'point', 'the', 'to', 'what', 'where', 'which',
}

ENTRYPOINT_HINTS = {'main', 'run', 'serve', 'cli', 'entry', 'app'}


def _identifier_tokens(value: str) -> set[str]:
    normalized = value.lower().replace('/', '_').replace('.', '_').replace('-', '_')
    tokens = {token for token in re.split(r'[^a-z0-9_]+|_+', normalized) if token}
    if normalized:
        tokens.add(normalized)
    return tokens


def _question_tokens(question: str) -> list[str]:
    raw_tokens = [token for token in re.findall(r'[a-zA-Z0-9_./-]+', question.lower()) if len(token) > 1]
    filtered_tokens = [token for token in raw_tokens if token not in STOPWORDS]

    expanded_tokens: list[str] = []
    for token in filtered_tokens:
        expanded_tokens.append(token)
        if token == 'evaluation':
            expanded_tokens.append('eval')
        if token == 'controller':
            expanded_tokens.append('serve')
        if token == 'worker':
            expanded_tokens.append('serve')
    return expanded_tokens


def _rank_files(analysis: dict[str, Any], question: str, max_files: int = 4) -> list[str]:
    scores: dict[str, int] = {}
    tokens = _question_tokens(question)

    for node in analysis['nodes']:
        file_path = node.get('file')
        if not file_path:
            continue

        label = str(node.get('label', '')).lower()
        node_id = str(node.get('id', '')).lower()
        basename = os.path.basename(file_path.lower())
        file_tokens = _identifier_tokens(file_path)
        label_tokens = _identifier_tokens(label)
        id_tokens = _identifier_tokens(node_id)
        score = 0
        for token in tokens:
            if token == label or token == basename.removesuffix('.py'):
                score += 50
            if token in label_tokens or token in id_tokens:
                score += 20
            if token in file_tokens:
                score += 15
            if token in file_path.lower() or token in node_id:
                score += 5

        if {'entry', 'point'} & set(re.findall(r'[a-zA-Z0-9_./-]+', question.lower())):
            if basename.startswith('run_') or label == 'main':
                score += 25
            if any(hint in file_tokens for hint in ENTRYPOINT_HINTS):
                score += 10

        if score > 0:
            scores[file_path] = max(scores.get(file_path, 0), score)

    if scores:
        return [file_path for file_path, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:max_files]]

    fallback_files = sorted({node['file'] for node in analysis['nodes'] if node.get('file', '').endswith('.py')})
    return fallback_files[:max_files]


def _build_context(analysis: dict[str, Any], files: list[str], max_chars: int = 8000) -> tuple[str, list[str]]:
    sections: list[str] = []
    included_files: list[str] = []
    used_chars = 0
    file_contents = cast(dict[str, str], analysis.get('file_contents', {}))

    for file_path in files:
        code = file_contents.get(file_path)
        if not code:
            continue

        section = f'\n\n# 파일: {file_path}\n{code}'
        if used_chars + len(section) > max_chars:
            if used_chars == 0:
                section = section[:max_chars]
            else:
                break
        sections.append(section)
        included_files.append(file_path)
        used_chars += len(section)
        if used_chars >= max_chars:
            break

    return ''.join(sections), included_files


def answer_question(repo_path: str, analysis: dict[str, Any], question: str) -> dict[str, object]:
    ranked_files = _rank_files(analysis, question)
    context, included_files = _build_context(analysis, ranked_files)

    if not included_files or not context.strip():
        return {
            'answer': '분석 가능한 Python 코드 문맥을 찾지 못했습니다.',
            'citations': [],
        }

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {
                'role': 'system',
                'content': '너는 코드 분석 전문가야. 선택된 코드와 그래프 단서를 근거로만 답하고 한국어로 답변해.',
            },
            {
                'role': 'user',
                'content': f'다음 선택된 코드만 참고해서 질문에 답해줘:\n{context}\n\n질문: {question}',
            },
        ],
    )

    return {
        'answer': response.choices[0].message.content,
        'citations': included_files,
    }
