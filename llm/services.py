from __future__ import annotations

import logging
import os
from typing import Any, cast

import requests
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from llm.context_selection import build_context_for_files, build_qa_context, identifier_tokens, question_tokens, rank_files

logger = logging.getLogger(__name__)

OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'


def _identifier_tokens(value: str) -> set[str]:
    return identifier_tokens(value)


def _question_tokens(question: str) -> list[str]:
    return question_tokens(question)


def _rank_files(analysis: dict[str, Any], question: str, max_files: int = 4) -> list[str]:
    return rank_files(analysis, question, max_files=max_files)


def _build_context(analysis: dict[str, Any], files: list[str], max_chars: int = 8000) -> tuple[str, list[str]]:
    context = build_context_for_files(analysis, files, max_chars=max_chars)
    return context.context, context.context_files


def _build_messages(context: str, question: str) -> list[dict[str, str]]:
    return [
        {
            'role': 'system',
            'content': '너는 코드 분석 전문가야. 선택된 코드와 그래프 단서를 근거로만 답하고 한국어로 답변해.',
        },
        {
            'role': 'user',
            'content': f'다음 선택된 그래프/코드 문맥만 참고해서 질문에 답해줘:\n{context}\n\n질문: {question}',
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


def _answer_question_classic(
    repo_path: str,
    analysis: dict[str, Any],
    question: str,
    *,
    selected_node_id: str | None = None,
    selected_file_path: str | None = None,
    max_context_files: int = 4,
) -> dict[str, object]:
    qa_context = build_qa_context(
        analysis,
        question,
        selected_node_id=selected_node_id,
        selected_file_path=selected_file_path,
        max_context_files=max_context_files,
    )

    if not qa_context.context_files or not qa_context.context.strip():
        return {
            'answer': '분석 가능한 Python 코드 문맥을 찾지 못했습니다.',
            'citations': [],
            'selected_nodes': qa_context.selected_nodes,
            'context_files': qa_context.context_files,
            'context_summary': qa_context.context_summary,
            'tool_trace': [],
            'warnings': [{'code': 'no_context'}, *qa_context.warnings],
        }

    messages = _build_messages(qa_context.context, question)
    answer = _generate_answer(messages)

    return {
        'answer': answer,
        'citations': qa_context.citations,
        'selected_nodes': qa_context.selected_nodes,
        'context_files': qa_context.context_files,
        'context_summary': qa_context.context_summary,
        'tool_trace': [],
        'warnings': qa_context.warnings,
    }


def answer_question(
    repo_path: str,
    analysis: dict[str, Any],
    question: str,
    *,
    selected_node_id: str | None = None,
    selected_file_path: str | None = None,
    max_context_files: int = 4,
) -> dict[str, object]:
    if os.getenv('QA_ENGINE', 'classic').lower() == 'smolagents':
        try:
            from llm.agents import answer_question_with_smolagents

            return answer_question_with_smolagents(
                repo_path,
                analysis,
                question,
                selected_node_id=selected_node_id,
                selected_file_path=selected_file_path,
                max_context_files=max_context_files,
            )
        except Exception as exc:
            logger.warning('smolagents QA failed; falling back to classic QA.', exc_info=True)
            response = _answer_question_classic(
                repo_path,
                analysis,
                question,
                selected_node_id=selected_node_id,
                selected_file_path=selected_file_path,
                max_context_files=max_context_files,
            )
            warnings = list(response.get('warnings', []))
            warnings.append({'code': 'smolagents_fallback', 'message': str(exc)})
            response['warnings'] = warnings
            return response

    return _answer_question_classic(
        repo_path,
        analysis,
        question,
        selected_node_id=selected_node_id,
        selected_file_path=selected_file_path,
        max_context_files=max_context_files,
    )
