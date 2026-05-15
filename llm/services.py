from __future__ import annotations

import logging
import os
import re
from typing import Any, cast

import requests
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'

STOPWORDS = {
    'a', 'an', 'and', 'are', 'at', 'defined', 'does', 'file', 'how', 'in', 'is',
    'of', 'point', 'the', 'to', 'what', 'where', 'which',
}

ENTRYPOINT_HINTS = {'main', 'run', 'cli', 'entry', 'app'}


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


def _build_messages(context: str, question: str) -> list[dict[str, str]]:
    return [
        {
            'role': 'system',
            'content': '너는 코드 분석 전문가야. 선택된 코드와 그래프 단서를 근거로만 답하고 한국어로 답변해.',
        },
        {
            'role': 'user',
            'content': f'다음 선택된 코드만 참고해서 질문에 답해줘:\n{context}\n\n질문: {question}',
        },
    ]


def _get_openai_client() -> OpenAI | None:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    for candidate in payload.get('candidates', []):
        parts = candidate.get('content', {}).get('parts', [])
        texts = [part.get('text', '') for part in parts if part.get('text')]
        if texts:
            return '\n'.join(texts)
    raise RuntimeError('Gemini 응답에서 텍스트를 찾지 못했습니다.')


def _answer_with_openai(messages: list[dict[str, str]]) -> str:
    client = _get_openai_client()
    if client is None:
        raise RuntimeError('OPENAI_API_KEY가 설정되지 않았습니다.')

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=cast(list[ChatCompletionMessageParam], messages),
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError('OpenAI 응답이 비어 있습니다.')
    return content


def _answer_with_gemini(messages: list[dict[str, str]]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY가 설정되지 않았습니다.')

    system_prompt = '\n\n'.join(message['content'] for message in messages if message['role'] == 'system')
    user_prompt = '\n\n'.join(message['content'] for message in messages if message['role'] == 'user')

    payload: dict[str, Any] = {
        'contents': [
            {
                'role': 'user',
                'parts': [{'text': user_prompt}],
            }
        ]
    }
    if system_prompt:
        payload['system_instruction'] = {
            'parts': [{'text': system_prompt}],
        }

    response = requests.post(
        GEMINI_API_URL.format(model=GEMINI_MODEL),
        headers={
            'x-goog-api-key': api_key,
            'Content-Type': 'application/json',
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return _extract_gemini_text(cast(dict[str, Any], response.json()))


def _generate_answer(messages: list[dict[str, str]]) -> str:
    openai_key = os.getenv('OPENAI_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')

    if openai_key:
        try:
            return _answer_with_openai(messages)
        except Exception:
            if not gemini_key:
                raise
            logger.warning('OpenAI 호출 실패로 Gemini 폴백을 사용합니다.', exc_info=True)

    if gemini_key:
        return _answer_with_gemini(messages)

    raise RuntimeError('사용 가능한 AI API 키가 없습니다.')


def answer_question(repo_path: str, analysis: dict[str, Any], question: str) -> dict[str, object]:
    ranked_files = _rank_files(analysis, question)
    context, included_files = _build_context(analysis, ranked_files)

    if not included_files or not context.strip():
        return {
            'answer': '분석 가능한 Python 코드 문맥을 찾지 못했습니다.',
            'citations': [],
        }

    messages = _build_messages(context, question)
    answer = _generate_answer(messages)

    return {
        'answer': answer,
        'citations': included_files,
    }
