from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

from harness_eval.evaluator import (
    DEFAULT_RAW_MODEL_ID,
    OPENCODE_ZEN_CHAT_COMPLETIONS_ENDPOINT,
    build_chat_completion_payload,
    build_job_packet,
    evaluate_transcript,
    load_json,
    sample_paths,
    transcript_paths,
    validate_matrix,
    validate_sample,
)


DEFAULT_MATRIX = Path(__file__).resolve().parent / 'harness_matrix.sample.json'
DEFAULT_SAMPLES = Path(__file__).resolve().parent / 'samples'
DEFAULT_TRANSCRIPTS = Path(__file__).resolve().parent / 'sample_transcripts'
DEFAULT_OUTPUT_DIR = Path('temp') / 'harness_eval'


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _checks_payload(checks) -> dict[str, object]:
    return {
        'passed': all(check.passed for check in checks),
        'checks': [check.__dict__ for check in checks],
    }


def cmd_validate_samples(args: argparse.Namespace) -> int:
    reports = []
    for path in sample_paths(args.samples_dir):
        sample = load_json(path)
        report = _checks_payload(validate_sample(sample))
        report['path'] = str(path)
        reports.append(report)
    _print_json({'passed': all(report['passed'] for report in reports), 'samples': reports})
    return 0 if all(report['passed'] for report in reports) else 1


def cmd_validate_matrix(args: argparse.Namespace) -> int:
    matrix = load_json(args.matrix)
    report = _checks_payload(validate_matrix(matrix))
    _print_json(report)
    return 0 if report['passed'] else 1


def cmd_evaluate_transcript(args: argparse.Namespace) -> int:
    sample = load_json(args.sample)
    transcript = load_json(args.transcript)
    report = evaluate_transcript(sample, transcript)
    _print_json(report)
    return 0 if report['passed'] else 1


def cmd_evaluate_transcripts(args: argparse.Namespace) -> int:
    samples_by_id = {load_json(path)['id']: load_json(path) for path in sample_paths(args.samples_dir)}
    reports = []
    for path in transcript_paths(args.transcripts_dir):
        transcript = load_json(path)
        sample_id = transcript.get('sample_id')
        if sample_id not in samples_by_id:
            reports.append({'path': str(path), 'passed': False, 'error': f'unknown sample_id {sample_id}'})
            continue
        report = evaluate_transcript(samples_by_id[str(sample_id)], transcript)
        report['path'] = str(path)
        reports.append(report)
    _print_json({'passed': all(report.get('passed') for report in reports), 'transcripts': reports})
    return 0 if all(report.get('passed') for report in reports) else 1


def cmd_render_job(args: argparse.Namespace) -> int:
    sample = load_json(args.sample)
    _print_json(build_job_packet(sample))
    return 0


def cmd_run_harness(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        _print_json({'passed': False, 'error': 'missing harness command after --'})
        return 2

    sample = load_json(args.sample)
    job_packet = build_job_packet(sample)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(job_packet, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _print_json({'passed': False, 'error': 'harness command timed out', 'timeout_seconds': args.timeout, 'stdout': exc.stdout, 'stderr': exc.stderr})
        return 1

    latency_ms = round((time.monotonic() - started) * 1000)
    try:
        transcript = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        _print_json(
            {
                'passed': False,
                'error': 'harness stdout was not valid JSON',
                'detail': str(exc),
                'returncode': completed.returncode,
                'stdout_preview': completed.stdout[:1000],
                'stderr_preview': completed.stderr[:1000],
            }
        )
        return 1

    if not isinstance(transcript, dict):
        _print_json({'passed': False, 'error': 'harness stdout JSON must be an object'})
        return 1

    report = evaluate_transcript(sample, transcript)
    report.update(
        {
            'harness_returncode': completed.returncode,
            'latency_ms': latency_ms,
            'stderr_preview': completed.stderr[:1000],
        }
    )
    if args.write_result:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f'harness_run_{sample.get("id")}_{int(time.time())}.json'
        output_path.write_text(
            json.dumps({'job': job_packet, 'transcript': transcript, 'report': report}, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        report['output_path'] = str(output_path)
    _print_json(report)
    return 0 if completed.returncode == 0 and report['passed'] else 1


def cmd_dry_run_matrix(args: argparse.Namespace) -> int:
    matrix = load_json(args.matrix)
    samples = [load_json(path) for path in sample_paths(args.samples_dir)]
    variants = matrix.get('variants') or []
    report = {
        'passed': True,
        'variant_count': len(variants),
        'sample_count': len(samples),
        'runs': [
            {
                'variant_id': variant.get('id'),
                'raw_model_id': (variant.get('model') or {}).get('raw_model_id'),
                'opencode_model_id': (variant.get('model') or {}).get('opencode_model_id'),
                'sample_id': sample.get('id'),
                'tools': variant.get('tools', []),
                'mcp_servers': variant.get('mcp_servers', []),
                'skills': variant.get('skills', []),
            }
            for variant in variants
            if isinstance(variant, dict)
            for sample in samples
        ],
    }
    _print_json(report)
    return 0


def _opencode_api_key() -> str:
    return os.getenv('OPENCODE_API_KEY', '').strip()


def _post_json(url: str, payload: dict[str, object], *, timeout_seconds: int) -> tuple[int, dict[str, str], dict[str, object]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        method='POST',
        headers={
            'Authorization': f'Bearer {_opencode_api_key()}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'capstone-pi-harness-eval/1.0',
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.status, dict(response.headers), json.loads(response.read().decode('utf-8'))


def _ensure_live_enabled(args: argparse.Namespace) -> tuple[bool, str]:
    if not args.live:
        return False, 'Pass --live to intentionally call OpenCode Zen.'
    if os.getenv('RUN_OPENCODE_LIVE_TESTS', '').strip().lower() != 'true':
        return False, 'RUN_OPENCODE_LIVE_TESTS=true is required for live calls.'
    if not _opencode_api_key():
        return False, 'OPENCODE_API_KEY is not set.'
    return True, ''


def cmd_live_smoke(args: argparse.Namespace) -> int:
    enabled, reason = _ensure_live_enabled(args)
    if not enabled:
        _print_json({'passed': False, 'skipped': True, 'reason': reason})
        return 2

    payload = {
        'model': args.model,
        'messages': [
            {'role': 'system', 'content': 'Return JSON only.'},
            {'role': 'user', 'content': '{"ping":"opencode-kimi-smoke","reply_with":{"ok":true}}'},
        ],
        'temperature': 0,
        'max_tokens': 80,
    }
    started = time.monotonic()
    try:
        status, headers, response = _post_json(args.endpoint, payload, timeout_seconds=args.timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')[:2000]
        _print_json({'passed': False, 'status': exc.code, 'error': detail})
        return 1
    except urllib.error.URLError as exc:
        _print_json({'passed': False, 'error': str(exc)})
        return 1
    latency_ms = round((time.monotonic() - started) * 1000)
    choices = response.get('choices') or []
    usage_present = isinstance(response.get('usage'), dict)
    response_id = response.get('id')
    checked_at_utc = _utc_now_iso()
    passed = status == 200 and bool(choices) and usage_present and bool(response_id)
    result = {
        'passed': passed,
        'status': status,
        'checked_at_utc': checked_at_utc,
        'endpoint': args.endpoint,
        'raw_model_id': args.model,
        'opencode_model_id': f'opencode/{args.model}' if not args.model.startswith('opencode/') else args.model,
        'latency_ms': latency_ms,
        'response_id': response_id,
        'usage': response.get('usage'),
        'usage_present': usage_present,
        'usage_receipt_present': usage_present and bool(response_id),
        'dashboard_correlation': {
            'response_id': response_id,
            'checked_at_utc': checked_at_utc,
            'raw_model_id': args.model,
            'endpoint': args.endpoint,
        },
        'request_headers': {
            key.lower(): value
            for key, value in headers.items()
            if key.lower() in {'x-request-id', 'x-ratelimit-limit-requests', 'x-ratelimit-remaining-requests'}
        },
        'content_preview': str(((choices[0] or {}).get('message') or {}).get('content') or '')[:500] if choices else '',
    }
    if args.write_result:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f'live_smoke_{int(time.time())}.json'
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        result['output_path'] = str(output_path)
    _print_json(result)
    return 0 if passed else 1


def cmd_live_sample(args: argparse.Namespace) -> int:
    enabled, reason = _ensure_live_enabled(args)
    if not enabled:
        _print_json({'passed': False, 'skipped': True, 'reason': reason})
        return 2

    sample = load_json(args.sample)
    variant = {
        'model': {'raw_model_id': args.model},
        'endpoint': {'chat_completions_url': args.endpoint},
    }
    payload = build_chat_completion_payload(sample, variant)
    started = time.monotonic()
    try:
        status, headers, response = _post_json(args.endpoint, payload, timeout_seconds=args.timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')[:2000]
        _print_json({'passed': False, 'status': exc.code, 'error': detail})
        return 1
    except urllib.error.URLError as exc:
        _print_json({'passed': False, 'error': str(exc)})
        return 1

    latency_ms = round((time.monotonic() - started) * 1000)
    choices = response.get('choices') or []
    raw_content = str(((choices[0] or {}).get('message') or {}).get('content') or '') if choices else ''
    json_parse_error = None
    try:
        transcript = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        json_parse_error = str(exc)
        transcript = {'sample_id': sample.get('id'), 'variant_id': args.model, 'tool_calls': [], 'final': {}}
    report = evaluate_transcript(sample, transcript)
    eval_passed = report['passed']
    usage_present = isinstance(response.get('usage'), dict)
    response_id = response.get('id')
    live_call_passed = status == 200 and bool(choices) and usage_present and bool(response_id)
    content_json_valid = json_parse_error is None and isinstance(transcript, dict)
    checked_at_utc = _utc_now_iso()
    report['passed'] = bool(live_call_passed and content_json_valid and eval_passed)
    report.update(
        {
            'eval_passed': eval_passed,
            'live_call_passed': live_call_passed,
            'content_json_valid': content_json_valid,
            'json_parse_error': json_parse_error,
            'status': status,
            'checked_at_utc': checked_at_utc,
            'endpoint': args.endpoint,
            'raw_model_id': args.model,
            'latency_ms': latency_ms,
            'response_id': response_id,
            'usage': response.get('usage'),
            'usage_present': usage_present,
            'usage_receipt_present': usage_present and bool(response_id),
            'dashboard_correlation': {
                'response_id': response_id,
                'checked_at_utc': checked_at_utc,
                'raw_model_id': args.model,
                'endpoint': args.endpoint,
            },
            'request_headers': {
                key.lower(): value
                for key, value in headers.items()
                if key.lower() in {'x-request-id', 'x-ratelimit-limit-requests', 'x-ratelimit-remaining-requests'}
            },
            'raw_content_preview': raw_content[:1000],
        }
    )
    if args.write_result:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f'live_sample_{sample.get("id")}_{int(time.time())}.json'
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        report['output_path'] = str(output_path)
    _print_json(report)
    return 0 if report['passed'] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run and deterministically score issue-map Pi/OpenCode job samples.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    validate_samples = subparsers.add_parser('validate-samples')
    validate_samples.add_argument('--samples-dir', default=str(DEFAULT_SAMPLES))
    validate_samples.set_defaults(func=cmd_validate_samples)

    validate_matrix = subparsers.add_parser('validate-matrix')
    validate_matrix.add_argument('--matrix', default=str(DEFAULT_MATRIX))
    validate_matrix.set_defaults(func=cmd_validate_matrix)

    evaluate_transcript_parser = subparsers.add_parser('evaluate-transcript')
    evaluate_transcript_parser.add_argument('sample')
    evaluate_transcript_parser.add_argument('transcript')
    evaluate_transcript_parser.set_defaults(func=cmd_evaluate_transcript)

    evaluate_transcripts_parser = subparsers.add_parser('evaluate-transcripts')
    evaluate_transcripts_parser.add_argument('--samples-dir', default=str(DEFAULT_SAMPLES))
    evaluate_transcripts_parser.add_argument('--transcripts-dir', default=str(DEFAULT_TRANSCRIPTS))
    evaluate_transcripts_parser.set_defaults(func=cmd_evaluate_transcripts)

    render_job = subparsers.add_parser('render-job')
    render_job.add_argument('sample')
    render_job.set_defaults(func=cmd_render_job)

    run_harness = subparsers.add_parser('run-harness')
    run_harness.add_argument('sample')
    run_harness.add_argument('--timeout', type=int, default=120)
    run_harness.add_argument('--write-result', action='store_true')
    run_harness.add_argument('command', nargs=argparse.REMAINDER)
    run_harness.set_defaults(func=cmd_run_harness)

    dry_run_matrix = subparsers.add_parser('dry-run-matrix')
    dry_run_matrix.add_argument('--matrix', default=str(DEFAULT_MATRIX))
    dry_run_matrix.add_argument('--samples-dir', default=str(DEFAULT_SAMPLES))
    dry_run_matrix.set_defaults(func=cmd_dry_run_matrix)

    live_smoke = subparsers.add_parser('live-smoke')
    live_smoke.add_argument('--live', action='store_true')
    live_smoke.add_argument('--model', default=DEFAULT_RAW_MODEL_ID)
    live_smoke.add_argument('--endpoint', default=OPENCODE_ZEN_CHAT_COMPLETIONS_ENDPOINT)
    live_smoke.add_argument('--timeout', type=int, default=90)
    live_smoke.add_argument('--write-result', action='store_true')
    live_smoke.set_defaults(func=cmd_live_smoke)

    live_sample = subparsers.add_parser('live-sample')
    live_sample.add_argument('sample')
    live_sample.add_argument('--live', action='store_true')
    live_sample.add_argument('--model', default=DEFAULT_RAW_MODEL_ID)
    live_sample.add_argument('--endpoint', default=OPENCODE_ZEN_CHAT_COMPLETIONS_ENDPOINT)
    live_sample.add_argument('--timeout', type=int, default=120)
    live_sample.add_argument('--write-result', action='store_true')
    live_sample.set_defaults(func=cmd_live_sample)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
