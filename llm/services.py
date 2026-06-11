from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping, cast

from django.conf import settings
import requests
import yaml

from llm.context_selection import build_context_for_files, build_qa_context, identifier_tokens, question_tokens, rank_files
from llm.issue_harness import (
    IssueHarnessUnavailable,
    build_qa_harness_job,
    command_from_string,
    run_issue_harness,
    stream_issue_harness,
)

logger = logging.getLogger(__name__)

OPENCODE_DEFAULT_MODEL = 'kimi-k2.5'
OPENCODE_DEFAULT_CHAT_COMPLETIONS_URL = 'https://opencode.ai/zen/v1/chat/completions'
APP_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config.yaml'


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


def _opencode_api_key() -> str:
    return os.getenv('OPENCODE_API_KEY', '').strip()


@lru_cache(maxsize=1)
def _load_app_config() -> Mapping[str, Any]:
    if not APP_CONFIG_PATH.exists():
        return {}
    with APP_CONFIG_PATH.open(encoding='utf-8') as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, Mapping):
        raise RuntimeError('config.yaml은 YAML object여야 합니다.')
    return cast(Mapping[str, Any], payload)


def _opencode_config() -> Mapping[str, Any]:
    config = _load_app_config().get('opencode', {})
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise RuntimeError('config.yaml의 opencode 설정은 YAML object여야 합니다.')
    return cast(Mapping[str, Any], config)


def _normalize_opencode_model_id(model: str) -> str:
    value = model.strip()
    if value.startswith('opencode/'):
        return value.removeprefix('opencode/').strip()
    return value


def _allowed_opencode_models() -> set[str]:
    raw = _opencode_config().get('allowed_models', [])
    if isinstance(raw, str):
        raw_models = raw.split(',')
    elif isinstance(raw, list):
        raw_models = [str(item) for item in raw]
    else:
        return set()
    return {_normalize_opencode_model_id(model) for model in raw_models if _normalize_opencode_model_id(model)}


def _resolve_opencode_model(model: str | None = None) -> str:
    configured_model = str(_opencode_config().get('model') or OPENCODE_DEFAULT_MODEL)
    selected = _normalize_opencode_model_id(model or configured_model)
    if not selected:
        raise RuntimeError('config.yaml에 OpenCode Zen 모델이 설정되지 않았습니다.')
    allowed = _allowed_opencode_models()
    if allowed and selected not in allowed:
        raise RuntimeError('허용되지 않은 OpenCode Zen 모델입니다.')
    return selected


def _opencode_timeout_seconds() -> int:
    return max(1, int(_opencode_config().get('timeout_seconds') or 60))


def _opencode_max_tokens() -> int:
    return max(1, int(_opencode_config().get('max_tokens') or 1200))


def _opencode_chat_completions_url() -> str:
    return str(_opencode_config().get('chat_completions_url') or OPENCODE_DEFAULT_CHAT_COMPLETIONS_URL)


def opencode_model_metadata(model: str | None = None) -> dict[str, str]:
    return {
        'provider': 'opencode_zen',
        'model': _resolve_opencode_model(model),
        'endpoint': _opencode_chat_completions_url(),
    }


def _opencode_headers(*, accept: str) -> dict[str, str]:
    api_key = _opencode_api_key()
    if not api_key:
        raise RuntimeError('OPENCODE_API_KEY가 설정되지 않았습니다.')
    return {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Accept': accept,
        'User-Agent': 'capstone-backend-ai/1.0',
    }


def _build_opencode_payload(messages: list[dict[str, str]], *, model: str | None = None, stream: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'model': _resolve_opencode_model(model),
        'messages': messages,
        'temperature': 0,
        'max_tokens': _opencode_max_tokens(),
    }
    if stream:
        payload['stream'] = True
    return payload


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, Mapping) and isinstance(part.get('text'), str):
                texts.append(cast(str, part['text']))
        return ''.join(texts)
    return ''


def _extract_opencode_text(payload: Mapping[str, Any]) -> str:
    for choice in payload.get('choices') or []:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get('message')
        if isinstance(message, Mapping):
            text = _content_to_text(message.get('content'))
            if text:
                return text
        text = _content_to_text(choice.get('text'))
        if text:
            return text
    raise RuntimeError('OpenCode Zen 응답에서 텍스트를 찾지 못했습니다.')


def _extract_opencode_delta(payload: Mapping[str, Any]) -> str:
    for choice in payload.get('choices') or []:
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get('delta')
        if isinstance(delta, Mapping):
            text = _content_to_text(delta.get('content'))
            if text:
                return text
        message = choice.get('message')
        if isinstance(message, Mapping):
            text = _content_to_text(message.get('content'))
            if text:
                return text
        text = _content_to_text(choice.get('text'))
        if text:
            return text
    return ''


def _answer_with_opencode_zen(messages: list[dict[str, str]], *, model: str | None = None) -> str:
    response = requests.post(
        _opencode_chat_completions_url(),
        headers=_opencode_headers(accept='application/json'),
        json=_build_opencode_payload(messages, model=model),
        timeout=(10, _opencode_timeout_seconds()),
    )
    response.raise_for_status()
    return _extract_opencode_text(cast(Mapping[str, Any], response.json()))


def _iter_opencode_sse_data(response) -> Iterator[str]:
    for raw_line in response.iter_lines(decode_unicode=True):
        if isinstance(raw_line, bytes):
            line = raw_line.decode('utf-8', errors='replace')
        else:
            line = str(raw_line)
        line = line.strip()
        if not line or line.startswith(':'):
            continue
        if not line.startswith('data:'):
            continue
        data = line[len('data:'):].strip()
        if data == '[DONE]':
            break
        if data:
            yield data


def _stream_answer_with_opencode_zen(messages: list[dict[str, str]], *, model: str | None = None) -> Iterator[str]:
    response = requests.post(
        _opencode_chat_completions_url(),
        headers=_opencode_headers(accept='text/event-stream'),
        json=_build_opencode_payload(messages, model=model, stream=True),
        timeout=(10, _opencode_timeout_seconds()),
        stream=True,
    )
    try:
        response.raise_for_status()
        for data in _iter_opencode_sse_data(response):
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                logger.debug('OpenCode Zen SSE JSON parse skipped: %s', data)
                continue
            text = _extract_opencode_delta(cast(Mapping[str, Any], payload))
            if text:
                yield text
    finally:
        close = getattr(response, 'close', None)
        if callable(close):
            close()


def _generate_answer(messages: list[dict[str, str]], *, model: str | None = None) -> str:
    return _answer_with_opencode_zen(messages, model=model)


def _stream_generate_answer(messages: list[dict[str, str]], *, model: str | None = None) -> Iterator[str]:
    emitted = False
    for chunk in _stream_answer_with_opencode_zen(messages, model=model):
        emitted = True
        yield chunk
    if not emitted:
        raise RuntimeError('OpenCode Zen 스트리밍 응답이 비어 있습니다.')


def _qa_harness_enabled() -> bool:
    return bool(getattr(settings, 'QA_HARNESS_ENABLED', False))


def _qa_harness_command() -> list[str] | None:
    command = str(getattr(settings, 'ISSUE_HARNESS_COMMAND', '') or '').strip()
    if not command:
        return None
    return command_from_string(command)


def _qa_harness_warning(error: IssueHarnessUnavailable) -> dict[str, Any]:
    return {
        'code': error.code,
        'message': error.message or 'QA harness를 사용할 수 없어 기존 QA 경로로 폴백했습니다.',
        'source': 'qa_harness',
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


QA_PROGRESS_TOOLS = {
    'get_question_context',
    'list_repo_files',
    'search_repo_symbols',
    'search_repo_text',
    'read_repo_file',
    'get_node',
    'get_neighbors',
    'read_node_context',
    'finish_repo_qa_transcript',
}
QA_PROGRESS_ARG_KEYS = {
    'search_repo_symbols': {'query'},
    'search_repo_text': {'query'},
    'read_repo_file': {'path'},
    'get_node': {'node_id'},
    'get_neighbors': {'node_id'},
    'read_node_context': {'node_id'},
}


def _short_string(value: Any, *, limit: int = 180) -> str:
    text = str(value)
    return text if len(text) <= limit else f'{text[:limit]}...'


def _safe_progress_args(tool_name: str, value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    allowed_keys = QA_PROGRESS_ARG_KEYS.get(tool_name, set())
    return {key: _short_string(value[key]) for key in allowed_keys if key in value and isinstance(value[key], (str, int, float, bool))}


def _answer_token_chunks(value: Any, *, chunk_size: int = 80) -> Iterator[str]:
    text = str(value or '')
    for index in range(0, len(text), chunk_size):
        chunk = text[index:index + chunk_size]
        if chunk:
            yield chunk


def _qa_stream_error_event(error: IssueHarnessUnavailable) -> dict[str, object]:
    return {
        'event': 'error',
        'data': {
            'error': error.message or 'QA harness 실행 중 오류가 발생했습니다.',
            'code': error.code,
            'source': 'qa_harness',
        },
    }


def _iter_mappings(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _iter_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_mappings(child)


def _qa_progress_from_pi_event(event: Mapping[str, Any]) -> list[dict[str, object]]:
    progress: list[dict[str, object]] = []
    for item in _iter_mappings(event):
        tool_name = item.get('toolName') or item.get('name') or item.get('tool')
        if not isinstance(tool_name, str) or tool_name not in QA_PROGRESS_TOOLS:
            continue
        role = str(item.get('role') or item.get('type') or item.get('event') or '').lower()
        arguments = item.get('arguments') or item.get('args') or item.get('input') or item.get('parameters') or {}
        event_name = 'harness_tool_result' if 'result' in role else 'harness_tool_call'
        progress.append({'event': event_name, 'data': {'name': tool_name, 'arguments': _safe_progress_args(tool_name, arguments)}})
    message = event.get('message')
    usage = message.get('usage') if isinstance(message, Mapping) else None
    if isinstance(usage, Mapping):
        progress.append(
            {
                'event': 'harness_usage',
                'data': {
                    key: usage[key]
                    for key in ('input', 'output', 'cacheRead', 'cacheWrite', 'totalTokens')
                    if isinstance(usage.get(key), (int, float))
                },
            }
        )
    return progress


def _qa_harness_job_and_env(
    repo_path: str,
    analysis: dict[str, Any],
    question: str,
    *,
    selected_node_id: str | None = None,
    selected_file_path: str | None = None,
    max_context_files: int = 4,
    model: str | None = None,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    revision = str(analysis.get('revision') or '')
    job = build_qa_harness_job(
        repo_path=repo_path,
        revision=revision,
        question=question,
        analysis=analysis,
        selected_node_id=selected_node_id,
        selected_file_path=selected_file_path,
        max_context_files=max_context_files,
    )
    extra_env = {'ISSUE_HARNESS_PI_MODEL': _normalize_opencode_model_id(model)} if model else None
    return job, extra_env


def _qa_harness_response(job: Mapping[str, Any], result: Any) -> dict[str, object]:
    output = result.output
    warnings = [dict(item) for item in output.get('warnings', []) if isinstance(item, Mapping)]
    return {
        'answer': str(output.get('answer') or ''),
        'citations': _string_list(output.get('citations')),
        'selected_nodes': _string_list(output.get('selected_nodes')),
        'context_files': _string_list(output.get('context_files')),
        'context_summary': {
            'strategy': 'pi_harness',
            'truncated': False,
            'file_contents_truncated': bool(job.get('limits', {}).get('file_contents_truncated')) if isinstance(job.get('limits'), Mapping) else False,
        },
        'tool_trace': result.tool_calls,
        'warnings': warnings,
        'harness': {
            'enabled': True,
            'source': 'pi_harness',
            'metadata': result.metadata,
        },
    }


def _answer_question_with_harness(
    repo_path: str,
    analysis: dict[str, Any],
    question: str,
    *,
    selected_node_id: str | None = None,
    selected_file_path: str | None = None,
    max_context_files: int = 4,
    model: str | None = None,
) -> dict[str, object]:
    job, extra_env = _qa_harness_job_and_env(
        repo_path=repo_path,
        question=question,
        analysis=analysis,
        selected_node_id=selected_node_id,
        selected_file_path=selected_file_path,
        max_context_files=max_context_files,
        model=model,
    )
    result = run_issue_harness(
        job,
        command=_qa_harness_command(),
        timeout_seconds=int(getattr(settings, 'ISSUE_HARNESS_TIMEOUT_SECONDS', 180)),
        extra_env=extra_env,
    )
    return _qa_harness_response(job, result)


def _answer_question_classic(
    repo_path: str,
    analysis: dict[str, Any],
    question: str,
    *,
    selected_node_id: str | None = None,
    selected_file_path: str | None = None,
    max_context_files: int = 4,
    model: str | None = None,
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
            'answer': '분석 가능한 source 코드 문맥을 찾지 못했습니다.',
            'citations': [],
            'selected_nodes': qa_context.selected_nodes,
            'context_files': qa_context.context_files,
            'context_summary': qa_context.context_summary,
            'tool_trace': [],
            'warnings': [{'code': 'no_context'}, *qa_context.warnings],
        }

    messages = _build_messages(qa_context.context, question)
    answer = _generate_answer(messages, model=model)

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
    model: str | None = None,
) -> dict[str, object]:
    if _qa_harness_enabled():
        try:
            return _answer_question_with_harness(
                repo_path,
                analysis,
                question,
                selected_node_id=selected_node_id,
                selected_file_path=selected_file_path,
                max_context_files=max_context_files,
                model=model,
            )
        except IssueHarnessUnavailable as error:
            response = _answer_question_classic(
                repo_path,
                analysis,
                question,
                selected_node_id=selected_node_id,
                selected_file_path=selected_file_path,
                max_context_files=max_context_files,
                model=model,
            )
            response['warnings'] = [*cast(list[Any], response.get('warnings', [])), _qa_harness_warning(error)]
            return response

    return _answer_question_classic(
        repo_path,
        analysis,
        question,
        selected_node_id=selected_node_id,
        selected_file_path=selected_file_path,
        max_context_files=max_context_files,
        model=model,
    )


def stream_answer_question(
    repo_path: str,
    analysis: dict[str, Any],
    question: str,
    *,
    selected_node_id: str | None = None,
    selected_file_path: str | None = None,
    max_context_files: int = 4,
    model: str | None = None,
) -> Iterator[dict[str, object]]:
    fallback_warning: dict[str, Any] | None = None
    if _qa_harness_enabled():
        harness_stream: Iterator[dict[str, Any]] | None = None
        try:
            job, extra_env = _qa_harness_job_and_env(
                repo_path,
                analysis,
                question,
                selected_node_id=selected_node_id,
                selected_file_path=selected_file_path,
                max_context_files=max_context_files,
                model=model,
            )
            yield {
                'event': 'harness_start',
                'data': {
                    'repo': repo_path,
                    'revision': job.get('repo', {}).get('revision') if isinstance(job.get('repo'), Mapping) else None,
                    'source': 'pi_harness',
                },
            }
            response = None
            emitted_progress: set[str] = set()
            harness_stream = stream_issue_harness(
                job,
                command=_qa_harness_command(),
                timeout_seconds=int(getattr(settings, 'ISSUE_HARNESS_TIMEOUT_SECONDS', 180)),
                extra_env=extra_env,
            )
            for item in harness_stream:
                if item.get('kind') == 'progress':
                    event = item.get('event')
                    if not isinstance(event, Mapping):
                        continue
                    for progress in _qa_progress_from_pi_event(event):
                        key = json.dumps(progress, ensure_ascii=False, sort_keys=True)
                        if key in emitted_progress:
                            continue
                        emitted_progress.add(key)
                        yield progress
                elif item.get('kind') == 'result':
                    result = item.get('result')
                    if result is not None:
                        response = _qa_harness_response(job, result)
            if response is None:
                raise IssueHarnessUnavailable('harness_missing_final', 'QA harness did not return a final result.')
            answer = str(response.get('answer') or '')
            yield {
                'event': 'meta',
                'data': {
                    'repo': repo_path,
                    'citations': response.get('citations', []),
                    'selected_nodes': response.get('selected_nodes', []),
                    'context_files': response.get('context_files', []),
                    'context_summary': response.get('context_summary', {}),
                    'warnings': response.get('warnings', []),
                    'harness': response.get('harness'),
                },
            }
            for chunk in _answer_token_chunks(answer):
                yield {'event': 'token', 'data': {'text': chunk}}
            yield {'event': 'final', 'data': response}
            return
        except IssueHarnessUnavailable as error:
            yield _qa_stream_error_event(error)
            return
        finally:
            if harness_stream is not None:
                close = getattr(harness_stream, 'close', None)
                if callable(close):
                    close()

    qa_context = build_qa_context(
        analysis,
        question,
        selected_node_id=selected_node_id,
        selected_file_path=selected_file_path,
        max_context_files=max_context_files,
    )

    meta = {
        'repo': repo_path,
        'citations': qa_context.citations,
        'selected_nodes': qa_context.selected_nodes,
        'context_files': qa_context.context_files,
        'context_summary': qa_context.context_summary,
        'warnings': [*qa_context.warnings, *([fallback_warning] if fallback_warning else [])],
    }

    if not qa_context.context_files or not qa_context.context.strip():
        warnings = [{'code': 'no_context'}, *qa_context.warnings, *([fallback_warning] if fallback_warning else [])]
        yield {
            'event': 'final',
            'data': {
                'answer': '분석 가능한 source 코드 문맥을 찾지 못했습니다.',
                'citations': [],
                'selected_nodes': qa_context.selected_nodes,
                'context_files': qa_context.context_files,
                'context_summary': qa_context.context_summary,
                'tool_trace': [],
                'warnings': warnings,
            },
        }
        return

    yield {'event': 'meta', 'data': meta}

    messages = _build_messages(qa_context.context, question)
    answer_parts: list[str] = []
    for text in _stream_generate_answer(messages, model=model):
        if not text:
            continue
        answer_parts.append(text)
        yield {'event': 'token', 'data': {'text': text}}

    yield {
        'event': 'final',
        'data': {
            'answer': ''.join(answer_parts),
            'citations': qa_context.citations,
            'selected_nodes': qa_context.selected_nodes,
            'context_files': qa_context.context_files,
            'context_summary': qa_context.context_summary,
            'tool_trace': [],
            'warnings': [*qa_context.warnings, *([fallback_warning] if fallback_warning else [])],
        },
    }
