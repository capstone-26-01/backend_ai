from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
import json
import os
import subprocess

from api.artifacts import build_graph_artifact, coerce_graph_artifact
from api.issue_map import extract_issue_evidence, rank_issue_candidates
from llm.issue_harness import build_issue_harness_job
from parser.language_registry import ignored_reason, language_for_path, normalize_enabled_languages
from parser.services import parse_repo


DEFAULT_TEMP_ROOT = Path('temp') / 'harness_eval' / 'swebench'
DEFAULT_PLAYGROUND_ROOT = Path('playground') / 'harness_eval' / 'swebench'
DEFAULT_ARTIFACTS_DIR = DEFAULT_TEMP_ROOT / 'artifacts'
DEFAULT_TRANSCRIPTS_DIR = DEFAULT_TEMP_ROOT / 'transcripts'
DEFAULT_REPORTS_DIR = DEFAULT_TEMP_ROOT / 'reports'
DEFAULT_MANIFEST_PATH = DEFAULT_TEMP_ROOT / 'artifact_manifest.json'
DEFAULT_REPOS_DIR = DEFAULT_PLAYGROUND_ROOT / 'repos'
DEFAULT_MAX_TOTAL_ANALYZED_BYTES = int(os.getenv('SWEBENCH_MAX_TOTAL_ANALYZED_BYTES', '20000000'))
DEFAULT_GIT_TIMEOUT_SECONDS = int(os.getenv('SWEBENCH_GIT_TIMEOUT_SECONDS', '120'))
RESOLVER_MANIFEST_NAMES = {'tsconfig.json', 'jsconfig.json', 'pnpm-workspace.yaml', 'package.json'}


class ArtifactPrepareError(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding='utf-8') as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sample_paths(samples_dir: str | Path) -> list[Path]:
    return sorted(Path(samples_dir).glob('*.json'))


def _safe_segment(value: str) -> str:
    if not value or value in {'.', '..'} or '/' in value or '\\' in value:
        raise ArtifactPrepareError('unsafe_repo', f'unsafe repository segment: {value}')
    return value


def repo_checkout_path(repo_full_name: str, repos_dir: Path = DEFAULT_REPOS_DIR) -> Path:
    parts = repo_full_name.split('/')
    if len(parts) != 2:
        raise ArtifactPrepareError('unsafe_repo', 'repo must use owner/name format')
    owner, repo = (_safe_segment(parts[0]), _safe_segment(parts[1]))
    return repos_dir / owner / repo


def _run_git(args: Sequence[str], *, cwd: Path | None = None, timeout: int = DEFAULT_GIT_TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', *args],
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def ensure_repo_checkout(repo_full_name: str, revision: str, *, repos_dir: Path = DEFAULT_REPOS_DIR) -> Path:
    checkout = repo_checkout_path(repo_full_name, repos_dir)
    if not (checkout / '.git').exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        clone = _run_git(['clone', '--no-checkout', '--filter=blob:none', f'https://github.com/{repo_full_name}.git', str(checkout)], cwd=None)
        if clone.returncode != 0:
            raise ArtifactPrepareError('checkout_failed', clone.stderr[-1200:] or clone.stdout[-1200:])
    fetch = _run_git(['fetch', '--depth', '1', 'origin', revision], cwd=checkout)
    if fetch.returncode != 0:
        fetch = _run_git(['fetch', 'origin', revision], cwd=checkout, timeout=max(DEFAULT_GIT_TIMEOUT_SECONDS, 300))
    if fetch.returncode != 0:
        raise ArtifactPrepareError('checkout_failed', fetch.stderr[-1200:] or fetch.stdout[-1200:])
    checkout_cmd = _run_git(['checkout', '--force', revision], cwd=checkout)
    if checkout_cmd.returncode != 0:
        raise ArtifactPrepareError('checkout_failed', checkout_cmd.stderr[-1200:] or checkout_cmd.stdout[-1200:])
    return checkout


def list_checkout_files(checkout: Path) -> list[str]:
    completed = _run_git(['ls-files'], cwd=checkout)
    if completed.returncode != 0:
        raise ArtifactPrepareError('file_listing_failed', completed.stderr[-1200:] or completed.stdout[-1200:])
    files = []
    for raw in completed.stdout.splitlines():
        path = PurePosixPath(raw.strip())
        if not raw.strip() or path.is_absolute() or any(part in {'', '.', '..'} for part in path.parts):
            continue
        files.append(path.as_posix())
    return sorted(files)


def _enabled_languages_for_profile(analysis_profile: str) -> tuple[str, ...]:
    if analysis_profile == 'multi-lang-js-ts-v1':
        return ('python', 'javascript', 'typescript')
    return normalize_enabled_languages(['python'])


def _read_text(checkout: Path, path: str) -> str | None:
    full_path = (checkout / path).resolve(strict=False)
    resolved_checkout = checkout.resolve()
    if full_path != resolved_checkout and resolved_checkout not in full_path.parents:
        return None
    try:
        return full_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return full_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None


def _source_file_manifest(files: Sequence[str], enabled_languages: tuple[str, ...]) -> tuple[list[str], list[str], dict[str, Any], list[dict[str, Any]]]:
    selected: list[str] = []
    detected: set[str] = set()
    manifest: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    for file_path in sorted(files):
        spec = language_for_path(file_path, enabled_languages=enabled_languages)
        skip_reason = ignored_reason(file_path, language_spec=spec) if spec is not None else None
        if spec is not None:
            detected.add(spec.id)
        content_stored = spec is not None and skip_reason is None
        manifest[file_path] = {
            'path': file_path,
            'language': spec.id if spec is not None else None,
            'language_family': spec.family if spec is not None else None,
            'support_level': spec.support_level if spec is not None else 'file_only',
            'content_stored': content_stored,
            'skip_reason': skip_reason if spec is not None else 'unsupported_language',
        }
        if spec is None:
            continue
        if skip_reason is not None:
            warnings.append({'code': f'{skip_reason}_file_skipped', 'message': 'SWE-bench source context skipped file.', 'path': file_path, 'skip_reason': skip_reason})
            continue
        selected.append(file_path)
    return selected, sorted(detected), manifest, warnings


def build_analysis_artifact_for_checkout(
    *,
    repo_full_name: str,
    revision: str,
    checkout: Path,
    analysis_profile: str = 'python-v1',
    max_total_analyzed_bytes: int = DEFAULT_MAX_TOTAL_ANALYZED_BYTES,
) -> dict[str, Any]:
    files = list_checkout_files(checkout)
    enabled_languages = _enabled_languages_for_profile(analysis_profile)
    selected_files, languages, file_manifest, selection_warnings = _source_file_manifest(files, enabled_languages)
    file_contents: dict[str, str] = {}
    resolver_file_contents: dict[str, str] = {}
    total_bytes = 0

    for file_path in sorted(files):
        if PurePosixPath(file_path).name not in RESOLVER_MANIFEST_NAMES:
            continue
        content = _read_text(checkout, file_path)
        if content is not None:
            resolver_file_contents[file_path] = content

    for file_path in selected_files:
        content = _read_text(checkout, file_path)
        if content is None:
            file_manifest[file_path]['content_stored'] = False
            file_manifest[file_path]['skip_reason'] = 'missing_content'
            continue
        total_bytes += len(content.encode('utf-8'))
        if total_bytes > max_total_analyzed_bytes:
            raise ArtifactPrepareError('too_large', 'repository source text exceeds SWE-bench harness analysis limit')
        file_contents[file_path] = content
        file_manifest[file_path]['byte_size'] = len(content.encode('utf-8'))
        file_manifest[file_path]['skip_reason'] = None

    graph = parse_repo(
        repo_full_name,
        files,
        lambda _repo_path, file_path: file_contents.get(file_path),
        enabled_languages=enabled_languages,
        resolver_file_contents=resolver_file_contents,
    )
    return build_graph_artifact(
        repo_path=repo_full_name,
        revision=revision,
        graph=graph,
        file_contents=file_contents,
        ref=revision,
        analysis_profile=analysis_profile,
        languages=languages,
        file_manifest=file_manifest,
        entrypoints=graph.get('entrypoints', []),
        key_modules=graph.get('key_modules', []),
        warnings=[*selection_warnings, *list(graph.get('warnings', []))],
        limits={'max_total_analyzed_bytes': max_total_analyzed_bytes},
    )


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        return {'schema_version': 1, 'generated_at_utc': utc_now_iso(), 'artifacts': {}}
    manifest = load_json(path)
    if not isinstance(manifest.get('artifacts'), dict):
        manifest['artifacts'] = {}
    return manifest


def write_manifest(manifest: Mapping[str, Any], path: Path = DEFAULT_MANIFEST_PATH) -> None:
    write_json(path, manifest)


def artifact_path_for_sample(sample: Mapping[str, Any], artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> Path:
    sample_id = str(sample.get('id') or 'sample')
    safe_id = ''.join(char if char.isalnum() or char in {'-', '_', '.'} else '_' for char in sample_id)
    return artifacts_dir / f'{safe_id}.json'


def prepare_artifact_for_sample(
    sample: Mapping[str, Any],
    *,
    repos_dir: Path = DEFAULT_REPOS_DIR,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> dict[str, Any]:
    repo = sample.get('repo') if isinstance(sample.get('repo'), Mapping) else {}
    expect = sample.get('expect') if isinstance(sample.get('expect'), Mapping) else {}
    repo_full_name = str(repo.get('full_name') or '')
    revision = str(repo.get('revision') or '')
    analysis_profile = str(repo.get('analysis_profile') or 'python-v1')
    if not repo_full_name or not revision:
        raise ArtifactPrepareError('missing_repo_or_revision', 'sample repo.full_name and repo.revision are required')
    checkout = ensure_repo_checkout(repo_full_name, revision, repos_dir=repos_dir)
    artifact = build_analysis_artifact_for_checkout(
        repo_full_name=repo_full_name,
        revision=revision,
        checkout=checkout,
        analysis_profile=analysis_profile,
    )
    gold_files = {str(path) for path in expect.get('gold_source_files') or [] if isinstance(path, str)}
    available_gold = sorted(gold_files & set(artifact.get('file_contents') or {}))
    if not available_gold:
        raise ArtifactPrepareError('gold_file_unavailable', 'no hidden gold source file is present in bounded artifact file_contents')
    job = build_runtime_job(sample, artifact)
    bounded_gold = sorted(gold_files & set(job.get('file_contents') or {}))
    if not bounded_gold:
        raise ArtifactPrepareError('gold_file_unavailable', 'no hidden gold source file survived bounded runtime job construction')
    artifact_path = artifact_path_for_sample(sample, artifacts_dir)
    write_json(artifact_path, artifact)
    return {
        'sample_id': sample.get('id'),
        'artifact_path': str(artifact_path),
        'checkout_path': str(checkout),
        'repo': repo_full_name,
        'revision': revision,
        'analysis_profile': analysis_profile,
        'gold_source_files_available': available_gold,
        'gold_source_files_in_job': bounded_gold,
        'prepared_at_utc': utc_now_iso(),
    }


def visible_sample_packet(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'schema_version': 1,
        'job_id': sample.get('id'),
        'task': sample.get('task'),
        'source': sample.get('source'),
        'repo': sample.get('repo'),
        'issue': sample.get('issue'),
    }


def build_runtime_job(sample: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    artifact = coerce_graph_artifact(artifact)
    repo = sample.get('repo') if isinstance(sample.get('repo'), Mapping) else {}
    issue = sample.get('issue') if isinstance(sample.get('issue'), Mapping) else {}
    comments: list[dict[str, Any]] = []
    evidence = extract_issue_evidence(issue, comments)
    candidates, warnings = rank_issue_candidates(artifact, evidence, max_candidates=20)
    analysis = dict(artifact)
    if warnings:
        analysis['warnings'] = [*list(analysis.get('warnings') or []), *warnings]
    return build_issue_harness_job(
        repo_path=str(repo.get('full_name') or artifact.get('repo') or ''),
        revision=str(repo.get('revision') or artifact.get('revision') or ''),
        issue=issue,
        comments=comments,
        evidence=evidence,
        candidates=candidates,
        analysis=analysis,
    )
