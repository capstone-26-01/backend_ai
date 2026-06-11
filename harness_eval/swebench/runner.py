from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import argparse
import json
import os
import subprocess
import sys
import time

from harness_eval.evaluator import OPENCODE_ZEN_CHAT_COMPLETIONS_ENDPOINT, validate_matrix
from harness_eval.swebench.evaluator import (
    FAILURE_CONTRACT_FAILED,
    FAILURE_INVALID_JSON,
    FAILURE_MISSING_FINAL,
    FAILURE_PROVIDER_RATE_LIMITED,
    FAILURE_RUNNER_ERROR,
    FAILURE_TIMEOUT,
    evaluate_transcript,
    markdown_report,
    summarize_results,
)
from harness_eval.swebench.importer import (
    DEFAULT_DATASET,
    DEFAULT_SAMPLES_DIR,
    DEFAULT_SPLIT,
    FORBIDDEN_MODEL_VISIBLE_FIELDS,
    build_sample_from_row,
    import_rows,
    load_dataset_rows,
)
from harness_eval.swebench.sample_builder import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_REPORTS_DIR,
    DEFAULT_REPOS_DIR,
    DEFAULT_TRANSCRIPTS_DIR,
    ArtifactPrepareError,
    build_runtime_job,
    load_json,
    load_manifest,
    prepare_artifact_for_sample,
    sample_paths,
    utc_now_iso,
    visible_sample_packet,
    write_json,
    write_manifest,
)
from llm.issue_harness import IssueHarnessUnavailable, run_issue_harness


FREE_MODEL_FALLBACK_ORDER = [
    {'display_name': 'DeepSeek V4 Flash Free', 'raw_model_id': 'deepseek-v4-flash-free'},
    {'display_name': 'Qwen3.6 Plus Free', 'raw_model_id': 'qwen3.6-plus-free'},
    {'display_name': 'MiniMax M3 Free', 'raw_model_id': 'minimax-m3-free'},
    {'display_name': 'Kimi K2.5', 'raw_model_id': 'kimi-k2.5'},
]


def load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path('.env'))


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_sample(path: str | Path) -> dict[str, Any]:
    return load_json(path)


def _artifact_for_sample(sample: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    entry = (manifest.get('artifacts') or {}).get(str(sample.get('id')))
    if not isinstance(entry, Mapping) or not entry.get('artifact_path'):
        raise ArtifactPrepareError('missing_artifact_manifest_entry', f'missing artifact manifest entry for {sample.get("id")}')
    return load_json(str(entry['artifact_path']))


def _ensure_live_enabled(args: argparse.Namespace) -> tuple[bool, str]:
    if getattr(args, 'command', None):
        return True, ''
    if not getattr(args, 'live', False):
        return False, 'Pass --live to intentionally call the runtime Pi/OpenCode harness.'
    if os.getenv('RUN_OPENCODE_LIVE_TESTS', '').strip().lower() != 'true':
        return False, 'RUN_OPENCODE_LIVE_TESTS=true is required for live calls.'
    if not os.getenv('OPENCODE_API_KEY', '').strip():
        return False, 'OPENCODE_API_KEY is not set.'
    return True, ''


def _pi_command(model: str, provider: str, passthrough: Sequence[str] | None = None) -> list[str]:
    if passthrough:
        command = list(passthrough)
        return command[1:] if command and command[0] == '--' else command
    provider = 'opencode' if provider == 'opencode_zen' else provider
    return [sys.executable, '-m', 'llm.pi_issue_runner', '--provider', provider, '--model', model]


def _transcript_path(sample_id: str, variant_id: str, transcripts_dir: Path = DEFAULT_TRANSCRIPTS_DIR) -> Path:
    safe = ''.join(char if char.isalnum() or char in {'-', '_', '.'} else '_' for char in f'{variant_id}__{sample_id}')
    return transcripts_dir / f'{safe}__{int(time.time())}.json'


def _failure_reason_from_harness(code: str) -> str:
    if code == 'harness_timeout':
        return FAILURE_TIMEOUT
    if code in {'provider_rate_limited', 'harness_rate_limited'}:
        return FAILURE_PROVIDER_RATE_LIMITED
    if code == 'harness_invalid_json':
        return FAILURE_INVALID_JSON
    if code == 'harness_missing_final':
        return FAILURE_MISSING_FINAL
    if code in {
        'harness_no_tool_calls',
        'harness_tool_budget_exceeded',
        'harness_missing_issue_context',
        'harness_missing_file_listing',
        'harness_missing_search',
        'harness_missing_inspection',
    }:
        return FAILURE_CONTRACT_FAILED
    return FAILURE_RUNNER_ERROR


def _cost_usd_from_usage(usage: Any) -> float | None:
    if not isinstance(usage, Mapping):
        return None
    for key in ('cost_usd', 'costUSD', 'totalCostUSD'):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    cost = usage.get('cost')
    if isinstance(cost, Mapping):
        total = cost.get('total')
        if isinstance(total, (int, float)):
            return float(total)
    if isinstance(cost, (int, float)):
        return float(cost)
    return None


def _collect_json_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_collect_json_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_collect_json_keys(child))
    return keys


def validate_sample(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, message: str) -> None:
        checks.append({'name': name, 'passed': passed, 'message': message})

    repo = sample.get('repo') if isinstance(sample.get('repo'), Mapping) else {}
    issue = sample.get('issue') if isinstance(sample.get('issue'), Mapping) else {}
    raw_expect = sample.get('expect')
    expect = raw_expect if isinstance(raw_expect, Mapping) else {}
    add('schema_version', sample.get('schema_version') == 1, 'schema_version must be 1')
    add('id', isinstance(sample.get('id'), str) and bool(sample.get('id')), 'id is required')
    add('source', isinstance(sample.get('source'), Mapping), 'source object is required')
    add('repo', isinstance(repo.get('full_name'), str) and bool(repo.get('full_name')) and isinstance(repo.get('revision'), str) and bool(repo.get('revision')), 'repo.full_name and repo.revision are required')
    add('issue', isinstance(issue.get('body'), str), 'issue.body is required')
    add('expect', isinstance(raw_expect, Mapping), 'hidden expect block is required')
    add('gold_source_files', isinstance(expect.get('gold_source_files'), list) and bool(expect.get('gold_source_files')), 'expect.gold_source_files must be non-empty')
    add('gold_hunks', isinstance(expect.get('gold_hunks'), list) and bool(expect.get('gold_hunks')), 'expect.gold_hunks must be non-empty')

    visible = visible_sample_packet(sample)
    visible_keys = _collect_json_keys(visible)
    for field in sorted(FORBIDDEN_MODEL_VISIBLE_FIELDS):
        add(f'no_visible_{field}', field not in visible_keys, f'{field} must not appear as a visible job packet field')
    for forbidden in ('patch', 'test_patch'):
        add(f'no_sample_{forbidden}', forbidden not in sample, f'{forbidden} must not be stored in generated sample')
    return checks


def cmd_import_samples(args: argparse.Namespace) -> int:
    rows = load_dataset_rows(args.dataset, args.split)
    report = import_rows(rows, output_dir=Path(args.output_dir), dataset=args.dataset, split=args.split, limit=args.limit)
    _print_json(report)
    return 0 if report['written_count'] else 1


def cmd_import_jsonl(args: argparse.Namespace) -> int:
    rows = []
    with Path(args.jsonl).open(encoding='utf-8') as file:
        for line in file:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    rows.append(payload)
    report = import_rows(rows, output_dir=Path(args.output_dir), dataset=args.dataset, split=args.split, limit=args.limit)
    _print_json(report)
    return 0 if report['written_count'] else 1


def cmd_validate_samples(args: argparse.Namespace) -> int:
    sample_reports = []
    for path in sample_paths(args.samples_dir):
        sample = _load_sample(path)
        checks = validate_sample(sample)
        sample_reports.append({'path': str(path), 'sample_id': sample.get('id'), 'passed': all(check['passed'] for check in checks), 'checks': checks})
    report = {'passed': bool(sample_reports) and all(item['passed'] for item in sample_reports), 'samples': sample_reports}
    _print_json(report)
    return 0 if report['passed'] else 1


def cmd_prepare_artifacts(args: argparse.Namespace) -> int:
    manifest = load_manifest(Path(args.manifest))
    artifacts = manifest.setdefault('artifacts', {})
    prepared: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    paths = sample_paths(args.samples_dir)
    if args.limit is not None:
        paths = paths[: args.limit]
    for path in paths:
        if args.target_prepared is not None and len(prepared) >= args.target_prepared:
            break
        sample = _load_sample(path)
        try:
            entry = prepare_artifact_for_sample(
                sample,
                repos_dir=Path(args.repos_dir),
                artifacts_dir=Path(args.artifacts_dir),
            )
        except ArtifactPrepareError as exc:
            skipped.append({'sample_id': sample.get('id'), 'reason': exc.reason, 'detail': exc.message})
            continue
        artifacts[str(sample.get('id'))] = entry
        prepared.append(entry)
    manifest['generated_at_utc'] = utc_now_iso()
    write_manifest(manifest, Path(args.manifest))
    if skipped:
        write_json(Path(args.reports_dir) / 'prepare_skips.json', {'skips': skipped})
    report = {'passed': bool(prepared), 'prepared_count': len(prepared), 'skipped_count': len(skipped), 'prepared': prepared, 'skipped': skipped, 'manifest_path': args.manifest}
    _print_json(report)
    return 0 if prepared else 1


def cmd_render_job(args: argparse.Namespace) -> int:
    sample = _load_sample(args.sample)
    artifact = _artifact_for_sample(sample, Path(args.manifest))
    job = build_runtime_job(sample, artifact)
    visible_keys = _collect_json_keys(job)
    leaks = [field for field in sorted(FORBIDDEN_MODEL_VISIBLE_FIELDS) if field in visible_keys]
    if leaks:
        _print_json({'passed': False, 'leaks': leaks})
        return 1
    _print_json(job)
    return 0


def cmd_evaluate_transcript(args: argparse.Namespace) -> int:
    sample = _load_sample(args.sample)
    artifact = _artifact_for_sample(sample, Path(args.manifest)) if args.manifest else None
    transcript = load_json(args.transcript)
    report = evaluate_transcript(sample, transcript, artifact=artifact)
    _print_json(report)
    return 0 if report['passed'] else 1


def run_one_sample(
    sample: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any],
    variant_id: str,
    model: str,
    provider: str,
    timeout: int,
    command: Sequence[str] | None,
    write_transcript: bool,
    transcripts_dir: Path,
) -> dict[str, Any]:
    job = build_runtime_job(sample, artifact)
    started = time.monotonic()
    try:
        result = run_issue_harness(
            job,
            command=_pi_command(model, provider, command),
            timeout_seconds=timeout,
        )
    except IssueHarnessUnavailable as exc:
        latency_ms = round((time.monotonic() - started) * 1000)
        return {
            'sample_id': sample.get('id'),
            'variant_id': variant_id,
            'model_id': model,
            'passed': False,
            'failure_reason': _failure_reason_from_harness(exc.code),
            'runner_error_code': exc.code,
            'runner_error': exc.message,
            'contract_pass': False,
            'gold_source_files': list((sample.get('expect') or {}).get('gold_source_files') or []) if isinstance(sample.get('expect'), Mapping) else [],
            'returned_paths': [],
            'matched_gold_files': [],
            'latency_ms': latency_ms,
            'cost_usd': None,
        }
    latency_ms = round((time.monotonic() - started) * 1000)
    transcript = {
        'sample_id': sample.get('id'),
        'variant_id': variant_id,
        'tool_calls': result.tool_calls,
        'final': result.output,
        'pi_metadata': result.metadata.get('pi_metadata') or {},
    }
    if result.metadata.get('harness_error'):
        transcript['error'] = result.metadata.get('harness_error')
    transcript_path = None
    if write_transcript:
        transcript_path = _transcript_path(str(sample.get('id')), variant_id, transcripts_dir)
        write_json(transcript_path, transcript)
    report = evaluate_transcript(sample, transcript, artifact=artifact)
    usage = ((result.metadata.get('pi_metadata') or {}).get('usage') or {}) if isinstance(result.metadata, Mapping) else {}
    report.update(
        {
            'model_id': model,
            'latency_ms': latency_ms,
            'cost_usd': _cost_usd_from_usage(usage),
            'usage': usage if isinstance(usage, Mapping) else None,
            'transcript_path': str(transcript_path) if transcript_path else None,
        }
    )
    return report


def cmd_run_sample(args: argparse.Namespace) -> int:
    enabled, reason = _ensure_live_enabled(args)
    if not enabled:
        _print_json({'passed': False, 'skipped': True, 'reason': reason})
        return 2
    sample = _load_sample(args.sample)
    artifact = _artifact_for_sample(sample, Path(args.manifest))
    report = run_one_sample(
        sample,
        artifact=artifact,
        variant_id=args.variant_id or args.model,
        model=args.model,
        provider=args.provider,
        timeout=args.timeout,
        command=args.command,
        write_transcript=args.write_transcript,
        transcripts_dir=Path(args.transcripts_dir),
    )
    _print_json(report)
    return 0 if report['passed'] else 1


def _matrix_variants(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    variants = matrix.get('variants')
    return [dict(variant) for variant in variants if isinstance(variant, Mapping)] if isinstance(variants, list) else []


def cmd_run_matrix(args: argparse.Namespace) -> int:
    enabled, reason = _ensure_live_enabled(args)
    if not enabled:
        _print_json({'passed': False, 'skipped': True, 'reason': reason})
        return 2
    matrix = load_json(args.matrix)
    matrix_checks = validate_matrix(matrix)
    if not all(check.passed for check in matrix_checks):
        _print_json({'passed': False, 'error': 'invalid matrix', 'checks': [check.__dict__ for check in matrix_checks]})
        return 1
    variants = _matrix_variants(matrix)
    paths = sample_paths(args.samples_dir)[: args.limit]
    results: list[dict[str, Any]] = []
    for variant in variants:
        model = variant.get('model') if isinstance(variant.get('model'), Mapping) else {}
        raw_model = str(model.get('raw_model_id') or args.model)
        variant_id = str(variant.get('id') or raw_model)
        provider = str((variant.get('endpoint') or {}).get('provider') or args.provider) if isinstance(variant.get('endpoint'), Mapping) else args.provider
        for path in paths:
            sample = _load_sample(path)
            try:
                artifact = _artifact_for_sample(sample, Path(args.manifest))
            except ArtifactPrepareError as exc:
                results.append({'sample_id': sample.get('id'), 'variant_id': variant_id, 'model_id': raw_model, 'passed': False, 'failure_reason': FAILURE_RUNNER_ERROR, 'runner_error_code': exc.reason, 'runner_error': exc.message})
                continue
            results.append(
                run_one_sample(
                    sample,
                    artifact=artifact,
                    variant_id=variant_id,
                    model=raw_model,
                    provider=provider,
                    timeout=args.timeout,
                    command=args.command,
                    write_transcript=args.write_transcripts,
                    transcripts_dir=Path(args.transcripts_dir),
                )
            )
    report = summarize_results(results, dataset=str(matrix.get('dataset') or DEFAULT_DATASET), split=str(matrix.get('split') or DEFAULT_SPLIT))
    if args.write_report:
        report_dir = Path(args.reports_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        stem = f'swebench_report_{int(time.time())}'
        report['report_paths'] = {'json': str(report_dir / f'{stem}.json'), 'markdown': str(report_dir / f'{stem}.md')}
        write_json(report_dir / f'{stem}.json', report)
        (report_dir / f'{stem}.md').write_text(markdown_report(report), encoding='utf-8')
    _print_json(report)
    return 0 if all(result.get('passed') for result in results) else 1


def cmd_smoke20(args: argparse.Namespace) -> int:
    enabled, reason = _ensure_live_enabled(args)
    if not enabled:
        _print_json({'passed': False, 'skipped': True, 'reason': reason})
        return 2

    paths = sample_paths(args.samples_dir)[:20]
    if len(paths) != 20:
        _print_json({'passed': False, 'error': 'smoke-20 requires exactly 20 available sample files', 'sample_count': len(paths)})
        return 1

    fallback_reports: list[dict[str, Any]] = []
    final_report: dict[str, Any] | None = None
    for item in FREE_MODEL_FALLBACK_ORDER:
        model = item['raw_model_id']
        variant_id = f'opencode-{model}-runtime-pi'
        probe_sample = _load_sample(paths[0])
        try:
            probe_artifact = _artifact_for_sample(probe_sample, Path(args.manifest))
        except ArtifactPrepareError as exc:
            _print_json({'passed': False, 'error': 'missing prepared smoke artifact', 'runner_error_code': exc.reason, 'runner_error': exc.message})
            return 1
        probe = run_one_sample(
            probe_sample,
            artifact=probe_artifact,
            variant_id=variant_id,
            model=model,
            provider=args.provider,
            timeout=args.timeout,
            command=args.command,
            write_transcript=args.write_transcripts,
            transcripts_dir=Path(args.transcripts_dir),
        )
        if probe.get('failure_reason') == FAILURE_RUNNER_ERROR:
            fallback_reports.append({'model_id': model, 'display_name': item['display_name'], 'probe': probe, 'fallback': True})
            continue

        results = [probe]
        for path in paths[1:]:
            sample = _load_sample(path)
            try:
                artifact = _artifact_for_sample(sample, Path(args.manifest))
            except ArtifactPrepareError as exc:
                results.append({'sample_id': sample.get('id'), 'variant_id': variant_id, 'model_id': model, 'passed': False, 'failure_reason': FAILURE_RUNNER_ERROR, 'runner_error_code': exc.reason, 'runner_error': exc.message})
                continue
            results.append(
                run_one_sample(
                    sample,
                    artifact=artifact,
                    variant_id=variant_id,
                    model=model,
                    provider=args.provider,
                    timeout=args.timeout,
                    command=args.command,
                    write_transcript=args.write_transcripts,
                    transcripts_dir=Path(args.transcripts_dir),
                )
            )

        final_report = summarize_results(results, dataset=DEFAULT_DATASET, split=DEFAULT_SPLIT)
        final_report['fallback_attempts'] = fallback_reports
        final_report['selected_model_order'] = [entry['raw_model_id'] for entry in FREE_MODEL_FALLBACK_ORDER]
        final_report['selected_model_id'] = model
        if args.write_report:
            report_dir = Path(args.reports_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            stem = f'swebench_smoke20_{model}_{int(time.time())}'
            final_report['report_paths'] = {'json': str(report_dir / f'{stem}.json'), 'markdown': str(report_dir / f'{stem}.md')}
            write_json(report_dir / f'{stem}.json', final_report)
            (report_dir / f'{stem}.md').write_text(markdown_report(final_report), encoding='utf-8')
        _print_json(final_report)
        return 0

    _print_json({'passed': False, 'error': 'all smoke-20 fallback models failed before producing evaluable transcripts', 'fallback_attempts': fallback_reports})
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run SWE-bench file-localization evaluation for the runtime Pi issue harness.')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    import_samples = subparsers.add_parser('import-samples')
    import_samples.add_argument('--dataset', default=DEFAULT_DATASET)
    import_samples.add_argument('--split', default=DEFAULT_SPLIT)
    import_samples.add_argument('--limit', type=int, default=None)
    import_samples.add_argument('--output-dir', default=str(DEFAULT_SAMPLES_DIR))
    import_samples.set_defaults(func=cmd_import_samples)

    import_jsonl = subparsers.add_parser('import-jsonl')
    import_jsonl.add_argument('jsonl')
    import_jsonl.add_argument('--dataset', default='local-jsonl')
    import_jsonl.add_argument('--split', default='local')
    import_jsonl.add_argument('--limit', type=int, default=None)
    import_jsonl.add_argument('--output-dir', default=str(DEFAULT_SAMPLES_DIR))
    import_jsonl.set_defaults(func=cmd_import_jsonl)

    validate_samples = subparsers.add_parser('validate-samples')
    validate_samples.add_argument('--samples-dir', default=str(DEFAULT_SAMPLES_DIR))
    validate_samples.set_defaults(func=cmd_validate_samples)

    prepare = subparsers.add_parser('prepare-artifacts')
    prepare.add_argument('--samples-dir', default=str(DEFAULT_SAMPLES_DIR))
    prepare.add_argument('--limit', type=int, default=None)
    prepare.add_argument('--target-prepared', type=int, default=None)
    prepare.add_argument('--repos-dir', default=str(DEFAULT_REPOS_DIR))
    prepare.add_argument('--artifacts-dir', default=str(DEFAULT_ARTIFACTS_DIR))
    prepare.add_argument('--reports-dir', default=str(DEFAULT_REPORTS_DIR))
    prepare.add_argument('--manifest', default=str(DEFAULT_MANIFEST_PATH))
    prepare.set_defaults(func=cmd_prepare_artifacts)

    render = subparsers.add_parser('render-job')
    render.add_argument('sample')
    render.add_argument('--manifest', default=str(DEFAULT_MANIFEST_PATH))
    render.set_defaults(func=cmd_render_job)

    evaluate = subparsers.add_parser('evaluate-transcript')
    evaluate.add_argument('sample')
    evaluate.add_argument('transcript')
    evaluate.add_argument('--manifest', default=str(DEFAULT_MANIFEST_PATH))
    evaluate.set_defaults(func=cmd_evaluate_transcript)

    run_sample = subparsers.add_parser('run-sample')
    run_sample.add_argument('sample')
    run_sample.add_argument('--live', action='store_true')
    run_sample.add_argument('--variant-id')
    run_sample.add_argument('--model', default='kimi-k2.5')
    run_sample.add_argument('--provider', default='opencode')
    run_sample.add_argument('--timeout', type=int, default=180)
    run_sample.add_argument('--manifest', default=str(DEFAULT_MANIFEST_PATH))
    run_sample.add_argument('--transcripts-dir', default=str(DEFAULT_TRANSCRIPTS_DIR))
    run_sample.add_argument('--write-transcript', action='store_true')
    run_sample.add_argument('command', nargs=argparse.REMAINDER)
    run_sample.set_defaults(func=cmd_run_sample)

    run_matrix = subparsers.add_parser('run-matrix')
    run_matrix.add_argument('--live', action='store_true')
    run_matrix.add_argument('--samples-dir', default=str(DEFAULT_SAMPLES_DIR))
    run_matrix.add_argument('--matrix', required=True)
    run_matrix.add_argument('--limit', type=int, default=100)
    run_matrix.add_argument('--model', default='kimi-k2.5')
    run_matrix.add_argument('--provider', default='opencode')
    run_matrix.add_argument('--timeout', type=int, default=180)
    run_matrix.add_argument('--manifest', default=str(DEFAULT_MANIFEST_PATH))
    run_matrix.add_argument('--reports-dir', default=str(DEFAULT_REPORTS_DIR))
    run_matrix.add_argument('--transcripts-dir', default=str(DEFAULT_TRANSCRIPTS_DIR))
    run_matrix.add_argument('--write-report', action='store_true')
    run_matrix.add_argument('--write-transcripts', action='store_true')
    run_matrix.add_argument('command', nargs=argparse.REMAINDER)
    run_matrix.set_defaults(func=cmd_run_matrix)

    smoke20 = subparsers.add_parser('smoke-20')
    smoke20.add_argument('--live', action='store_true')
    smoke20.add_argument('--samples-dir', default=str(DEFAULT_SAMPLES_DIR))
    smoke20.add_argument('--provider', default='opencode')
    smoke20.add_argument('--timeout', type=int, default=180)
    smoke20.add_argument('--manifest', default=str(DEFAULT_MANIFEST_PATH))
    smoke20.add_argument('--reports-dir', default=str(DEFAULT_REPORTS_DIR))
    smoke20.add_argument('--transcripts-dir', default=str(DEFAULT_TRANSCRIPTS_DIR))
    smoke20.add_argument('--write-report', action='store_true', default=True)
    smoke20.add_argument('--write-transcripts', action='store_true')
    smoke20.add_argument('command', nargs=argparse.REMAINDER)
    smoke20.set_defaults(func=cmd_smoke20)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
