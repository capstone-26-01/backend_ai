from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import re

from harness_eval.swebench.diff_labels import PatchLabelError, labels_from_patch


DEFAULT_DATASET = 'SWE-bench/SWE-bench_Verified'
DEFAULT_SPLIT = 'test'
DEFAULT_ANALYSIS_PROFILE = 'python-v1'
DEFAULT_SAMPLES_DIR = Path('harness_eval') / 'swebench_samples'
DEFAULT_SKIP_REPORT = Path('temp') / 'harness_eval' / 'swebench' / 'reports' / 'import_skips.json'
FORBIDDEN_MODEL_VISIBLE_FIELDS = {
    'patch',
    'test_patch',
    'expect',
    'gold_source_files',
    'gold_hunks',
    'FAIL_TO_PASS',
    'PASS_TO_PASS',
}


@dataclass(frozen=True)
class ImportSkip:
    instance_id: str
    reason: str
    detail: str = ''

    def as_dict(self) -> dict[str, str]:
        payload = {'instance_id': self.instance_id, 'reason': self.reason}
        if self.detail:
            payload['detail'] = self.detail
        return payload


def _string(value: Any) -> str:
    return value if isinstance(value, str) else str(value or '')


def sample_id_for_instance(instance_id: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', instance_id.strip())
    return f'swebench__{safe}'


def sample_filename(sample_id: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', sample_id) + '.json'


def issue_number_from_row(row: Mapping[str, Any]) -> int:
    issue_url = _string(row.get('issue_url'))
    match = re.search(r'/issues/(\d+)', issue_url)
    if match:
        return int(match.group(1))
    instance_id = _string(row.get('instance_id'))
    match = re.search(r'-(\d+)(?:$|[^0-9])', instance_id)
    if match:
        return int(match.group(1))
    return 0


def _issue_title(row: Mapping[str, Any]) -> str:
    instance_id = _string(row.get('instance_id')) or 'unknown'
    problem = _string(row.get('problem_statement')).strip()
    for line in problem.splitlines():
        title = line.strip().strip('#').strip()
        if title:
            return title[:180]
    return f'SWE-bench issue {instance_id}'


def build_sample_from_row(
    row: Mapping[str, Any],
    *,
    dataset: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    analysis_profile: str = DEFAULT_ANALYSIS_PROFILE,
) -> tuple[dict[str, Any] | None, ImportSkip | None]:
    instance_id = _string(row.get('instance_id')).strip()
    if not instance_id:
        return None, ImportSkip('', 'missing_instance_id')
    repo = _string(row.get('repo')).strip()
    revision = _string(row.get('base_commit')).strip()
    patch = _string(row.get('patch'))
    if not repo:
        return None, ImportSkip(instance_id, 'missing_repo')
    if not revision:
        return None, ImportSkip(instance_id, 'missing_base_commit')
    if not patch.strip():
        return None, ImportSkip(instance_id, 'empty_patch')
    try:
        labels = labels_from_patch(patch)
    except PatchLabelError as exc:
        return None, ImportSkip(instance_id, 'patch_parse_failed', str(exc))
    if not labels.gold_source_files:
        return None, ImportSkip(instance_id, 'no_source_labels')

    sample_id = sample_id_for_instance(instance_id)
    problem_statement = _string(row.get('problem_statement'))
    sample = {
        'schema_version': 1,
        'id': sample_id,
        'source': {
            'kind': 'swebench',
            'dataset': dataset,
            'split': split,
            'instance_id': instance_id,
            'issue_url': _string(row.get('issue_url')) or None,
            'pr_url': _string(row.get('pr_url')) or None,
        },
        'task': 'Analyze the provided repository and identify the code nodes most relevant to the selected issue.',
        'repo': {
            'kind': 'github',
            'full_name': repo,
            'revision': revision,
            'analysis_profile': analysis_profile,
        },
        'issue': {
            'number': issue_number_from_row(row),
            'title': _issue_title(row),
            'body': problem_statement,
            'labels': [],
            'html_url': _string(row.get('issue_url')) or None,
        },
        'expect': {
            'label_source': 'swebench_patch',
            'label_confidence': 'gold_file_silver_origin',
            **labels.as_expect_fields(),
            'forbidden_tools': ['shell', 'filesystem', 'network', 'github_api'],
        },
    }
    return sample, None


def load_dataset_rows(dataset: str, split: str) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('datasets is required for import-samples; install requirements.txt') from exc
    loaded = load_dataset(dataset, split=split)
    for row in loaded:
        if isinstance(row, Mapping):
            yield row


def write_sample(sample: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_id = _string(sample.get('id'))
    path = output_dir / sample_filename(sample_id)
    path.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def import_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    output_dir: Path = DEFAULT_SAMPLES_DIR,
    dataset: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    limit: int | None = None,
    skip_report_path: Path = DEFAULT_SKIP_REPORT,
) -> dict[str, Any]:
    written: list[str] = []
    skips: list[ImportSkip] = []
    attempted = 0
    for row in rows:
        attempted += 1
        sample, skip = build_sample_from_row(row, dataset=dataset, split=split)
        if skip is not None:
            skips.append(skip)
            continue
        if sample is None:
            skips.append(ImportSkip(_string(row.get('instance_id')), 'unknown_import_error'))
            continue
        written.append(str(write_sample(sample, output_dir)))
        if limit is not None and len(written) >= limit:
            break

    skip_report_path.parent.mkdir(parents=True, exist_ok=True)
    skip_report_path.write_text(
        json.dumps({'dataset': dataset, 'split': split, 'skips': [skip.as_dict() for skip in skips]}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return {
        'dataset': dataset,
        'split': split,
        'attempted_rows': attempted,
        'written_count': len(written),
        'written_paths': written,
        'skipped_count': len(skips),
        'skip_report_path': str(skip_report_path),
    }
