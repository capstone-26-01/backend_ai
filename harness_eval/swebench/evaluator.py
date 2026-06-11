from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath
from statistics import median
from typing import Any, Mapping

from harness_eval.swebench.sample_builder import utc_now_iso


FAILURE_INVALID_JSON = 'invalid_json'
FAILURE_MISSING_FINAL = 'missing_final'
FAILURE_CONTRACT_FAILED = 'contract_failed'
FAILURE_NO_FINAL_NODES = 'no_final_nodes'
FAILURE_NO_RESOLVED_PATHS = 'no_resolved_paths'
FAILURE_NO_GOLD_OVERLAP = 'no_gold_file_overlap'
FAILURE_TIMEOUT = 'timeout'
FAILURE_PROVIDER_RATE_LIMITED = 'provider_rate_limited'
FAILURE_RUNNER_ERROR = 'runner_error'


def normalize_repo_path(path: str | None) -> str | None:
    if not path:
        return None
    value = str(path).strip().replace('\\', '/').strip('/')
    posix = PurePosixPath(value)
    if posix.is_absolute() or not value or any(part in {'', '.', '..'} for part in posix.parts):
        return None
    return posix.as_posix()


def _collect_values(value: Any, key: str) -> list[Any]:
    matches: list[Any] = []
    if isinstance(value, Mapping):
        for item_key, item_value in value.items():
            if item_key == key:
                matches.append(item_value)
            matches.extend(_collect_values(item_value, key))
    elif isinstance(value, list):
        for item in value:
            matches.extend(_collect_values(item, key))
    return matches


def _tool_names(transcript: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for call in transcript.get('tool_calls') or []:
        if isinstance(call, Mapping):
            name = call.get('name') or call.get('tool')
            if isinstance(name, str) and name:
                names.append(name)
    return names


def _final_lists(final: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    hypotheses = final.get('hypotheses')
    investigation_path = final.get('investigation_path')
    return (
        [item for item in hypotheses if isinstance(item, Mapping)] if isinstance(hypotheses, list) else [],
        [item for item in investigation_path if isinstance(item, Mapping)] if isinstance(investigation_path, list) else [],
    )


def _node_path_map(artifact: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(artifact, Mapping):
        return {}
    paths: dict[str, str] = {}
    for node in artifact.get('nodes') or []:
        if not isinstance(node, Mapping):
            continue
        node_id = node.get('id')
        path = normalize_repo_path(str(node.get('path') or node.get('file') or ''))
        if isinstance(node_id, str) and path:
            paths[node_id] = path
    return paths


def _paths_from_final(final: Mapping[str, Any], artifact: Mapping[str, Any] | None) -> tuple[set[str], set[str]]:
    node_paths = _node_path_map(artifact)
    paths: set[str] = set()
    node_ids: set[str] = set()
    for value in _collect_values(final, 'path'):
        if isinstance(value, str):
            normalized = normalize_repo_path(value)
            if normalized:
                paths.add(normalized)
    for value in _collect_values(final, 'node_id'):
        if not isinstance(value, str) or not value:
            continue
        node_ids.add(value)
        mapped = normalize_repo_path(node_paths.get(value))
        if mapped:
            paths.add(mapped)
        elif '::' in value:
            inferred = normalize_repo_path(value.split('::', 1)[0])
            if inferred:
                paths.add(inferred)
    return paths, node_ids


def _contract_failure(tool_names: list[str], has_final_nodes: bool) -> str | None:
    if not tool_names:
        return 'no tool calls recorded'
    if 'get_issue_context' not in tool_names:
        return 'get_issue_context was not called'
    if 'list_repo_files' not in tool_names:
        return 'list_repo_files was not called'
    if 'search_repo_symbols' not in tool_names and 'search_repo_text' not in tool_names:
        return 'no symbol or text search was called'
    if has_final_nodes and not {'read_repo_file', 'read_node_context', 'get_neighbors'} & set(tool_names):
        return 'final nodes were named without code or graph inspection'
    return None


def evaluate_transcript(
    sample: Mapping[str, Any],
    transcript: Any,
    *,
    artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expect = sample.get('expect') if isinstance(sample.get('expect'), Mapping) else {}
    gold_files = {path for path in (normalize_repo_path(str(item)) for item in expect.get('gold_source_files') or []) if path}
    base_report: dict[str, Any] = {
        'sample_id': sample.get('id'),
        'variant_id': transcript.get('variant_id') if isinstance(transcript, Mapping) else None,
        'passed': False,
        'failure_reason': None,
        'contract_pass': False,
        'gold_source_files': sorted(gold_files),
        'returned_paths': [],
        'matched_gold_files': [],
        'found_node_ids': [],
        'tool_calls': [],
    }
    if not isinstance(transcript, Mapping):
        return {**base_report, 'failure_reason': FAILURE_INVALID_JSON}
    final = transcript.get('final')
    if not isinstance(final, Mapping):
        return {**base_report, 'failure_reason': FAILURE_MISSING_FINAL}
    if not isinstance(final.get('hypotheses'), list) or not isinstance(final.get('investigation_path'), list):
        return {**base_report, 'failure_reason': FAILURE_MISSING_FINAL}

    hypotheses, investigation_path = _final_lists(final)
    tool_names = _tool_names(transcript)
    returned_paths, found_node_ids = _paths_from_final(final, artifact)
    has_final_nodes = bool(hypotheses or investigation_path or found_node_ids)
    transcript_error = transcript.get('error') if isinstance(transcript.get('error'), str) else None
    contract_error = transcript_error or _contract_failure(tool_names, has_final_nodes)
    contract_pass = contract_error is None
    matched = sorted(returned_paths & gold_files)

    report = {
        **base_report,
        'variant_id': transcript.get('variant_id'),
        'contract_pass': contract_pass,
        'contract_error': contract_error,
        'returned_paths': sorted(returned_paths),
        'matched_gold_files': matched,
        'found_node_ids': sorted(found_node_ids),
        'tool_calls': tool_names,
    }
    if contract_error is not None:
        report['failure_reason'] = FAILURE_CONTRACT_FAILED
    elif not has_final_nodes:
        report['failure_reason'] = FAILURE_NO_FINAL_NODES
    elif not returned_paths:
        report['failure_reason'] = FAILURE_NO_RESOLVED_PATHS
    elif not matched:
        report['failure_reason'] = FAILURE_NO_GOLD_OVERLAP
    else:
        report['passed'] = True
        report['failure_reason'] = None
    return report


def summarize_results(
    results: list[Mapping[str, Any]],
    *,
    dataset: str,
    split: str,
) -> dict[str, Any]:
    variants: list[dict[str, Any]] = []
    by_variant: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        variant_id = str(result.get('variant_id') or 'unknown')
        by_variant.setdefault(variant_id, []).append(result)
    for variant_id, rows in sorted(by_variant.items()):
        total = len(rows)
        passed = sum(1 for row in rows if row.get('passed') is True)
        latencies = sorted(int(row['latency_ms']) for row in rows if isinstance(row.get('latency_ms'), int))
        costs = [float(row['cost_usd']) for row in rows if isinstance(row.get('cost_usd'), (int, float))]
        p95_index = max(0, min(len(latencies) - 1, round(len(latencies) * 0.95) - 1)) if latencies else 0
        total_cost = round(sum(costs), 6) if costs else None
        variants.append(
            {
                'variant_id': variant_id,
                'model_id': rows[0].get('model_id'),
                'passed': passed,
                'failed': total - passed,
                'total': total,
                'pass_rate': round(passed / total, 4) if total else 0.0,
                'median_latency_ms': int(median(latencies)) if latencies else None,
                'p95_latency_ms': latencies[p95_index] if latencies else None,
                'total_cost_usd': total_cost,
                'avg_cost_usd': round(total_cost / total, 6) if total and total_cost is not None else None,
            }
        )
    return {
        'schema_version': 1,
        'dataset': dataset,
        'split': split,
        'sample_count': len({result.get('sample_id') for result in results}),
        'generated_at_utc': utc_now_iso(),
        'variants': variants,
        'results': list(results),
        'failures_by_reason': dict(Counter(str(result.get('failure_reason')) for result in results if result.get('failure_reason'))),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        '# SWE-bench Pi Harness Localization Report',
        '',
        '## Model Comparison',
        '',
        '| variant | model | passed | total | pass rate | median latency | p95 latency | total cost |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for variant in report.get('variants') or []:
        if not isinstance(variant, Mapping):
            continue
        lines.append(
            '| {variant} | {model} | {passed} | {total} | {rate:.1f}% | {median} | {p95} | {cost} |'.format(
                variant=variant.get('variant_id'),
                model=variant.get('model_id') or '',
                passed=variant.get('passed'),
                total=variant.get('total'),
                rate=float(variant.get('pass_rate') or 0) * 100,
                median=variant.get('median_latency_ms') or '',
                p95=variant.get('p95_latency_ms') or '',
                cost=variant.get('total_cost_usd') if variant.get('total_cost_usd') is not None else '',
            )
        )
    lines.extend(['', '## Failures By Reason', '', '| reason | count |', '|---|---:|'])
    for reason, count in sorted((report.get('failures_by_reason') or {}).items()):
        lines.append(f'| {reason} | {count} |')
    lines.extend(['', '## Worst Misses', '', '| sample | gold files | returned paths | reason |', '|---|---|---|---|'])
    for result in report.get('results') or []:
        if not isinstance(result, Mapping) or result.get('passed'):
            continue
        lines.append(
            '| {sample} | {gold} | {returned} | {reason} |'.format(
                sample=result.get('sample_id'),
                gold=', '.join(result.get('gold_source_files') or []),
                returned=', '.join(result.get('returned_paths') or []),
                reason=result.get('failure_reason'),
            )
        )
    lines.append('')
    return '\n'.join(lines)
