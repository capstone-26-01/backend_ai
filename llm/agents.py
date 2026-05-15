from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
from typing import Any

from llm.context_selection import build_qa_context
from llm.services import OPENAI_MODEL
from llm.tools import ArtifactToolbox, ToolLimitExceeded, build_smolagents_tools


class AgentUnavailable(RuntimeError):
    pass


class AgentTimedOut(RuntimeError):
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _prime_tool_trace(toolbox: ArtifactToolbox, question: str, selected_nodes: list[str], context_files: list[str]) -> list[dict[str, Any]]:
    warnings = []
    planned_calls = [
        lambda: toolbox.search_symbols(question, limit=5),
        toolbox.get_entrypoints,
        toolbox.get_key_modules,
    ]
    if toolbox.analysis.get('summaries'):
        planned_calls.append(lambda: toolbox.get_summary('repo_overview'))
    if selected_nodes:
        planned_calls.append(lambda: toolbox.get_node(selected_nodes[0]))
        planned_calls.append(lambda: toolbox.get_neighbors(selected_nodes[0], limit=8))
    for file_path in context_files[:1]:
        planned_calls.append(lambda file_path=file_path: toolbox.get_file_excerpt(file_path))

    for call in planned_calls:
        try:
            call()
        except ToolLimitExceeded as exc:
            warnings.append({'code': 'tool_call_limit_exceeded', 'message': str(exc)})
            break
    return warnings


def _run_smolagents_agent(tools, task: str, *, max_steps: int, timeout_seconds: int) -> str:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise AgentUnavailable('OPENAI_API_KEY is required for QA_ENGINE=smolagents')

    from smolagents import OpenAIServerModel, ToolCallingAgent

    model = OpenAIServerModel(model_id=os.getenv('OPENAI_MODEL', OPENAI_MODEL), api_key=api_key)
    agent = ToolCallingAgent(tools=tools, model=model)
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(agent.run, task, max_steps=max_steps)
    try:
        return str(future.result(timeout=timeout_seconds))
    except TimeoutError as exc:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise AgentTimedOut(f'smolagents execution exceeded {timeout_seconds} seconds') from exc
    finally:
        if future.done():
            executor.shutdown(wait=True)


def answer_question_with_smolagents(
    repo_path: str,
    analysis: dict[str, Any],
    question: str,
    *,
    selected_node_id: str | None = None,
    selected_file_path: str | None = None,
    max_context_files: int = 4,
) -> dict[str, object]:
    if not os.getenv('OPENAI_API_KEY'):
        raise AgentUnavailable('OPENAI_API_KEY is required for QA_ENGINE=smolagents')

    max_steps = _env_int('SMOLAGENTS_MAX_STEPS', 4)
    max_tool_calls = _env_int('SMOLAGENTS_MAX_TOOL_CALLS', 8)
    timeout_seconds = _env_int('SMOLAGENTS_TIMEOUT_SECONDS', 30)
    qa_context = build_qa_context(
        analysis,
        question,
        selected_node_id=selected_node_id,
        selected_file_path=selected_file_path,
        max_context_files=max_context_files,
    )
    toolbox = ArtifactToolbox(analysis, max_tool_calls=max_tool_calls)
    warnings = _prime_tool_trace(toolbox, question, qa_context.selected_nodes, qa_context.context_files)
    tools = build_smolagents_tools(toolbox)
    task = (
        f'You are answering a question about {repo_path}. '
        'Use only the provided artifact tools. Do not assume files outside tool results. '
        'Answer in Korean with concrete citations.\n\n'
        f'Question: {question}\n'
        f'Initial context files: {qa_context.context_files}\n'
        f'Initial selected nodes: {qa_context.selected_nodes}'
    )
    answer = _run_smolagents_agent(tools, task, max_steps=max_steps, timeout_seconds=timeout_seconds)

    return {
        'answer': answer,
        'citations': qa_context.citations,
        'selected_nodes': qa_context.selected_nodes,
        'context_files': qa_context.context_files,
        'context_summary': qa_context.context_summary,
        'tool_trace': toolbox.tool_trace,
        'warnings': [*qa_context.warnings, *warnings],
    }
