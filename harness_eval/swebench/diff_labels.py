from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Sequence


SOURCE_SUFFIXES = ('.py',)
TEST_PATH_PARTS = {'tests', 'test', 'testing'}
DOC_PATH_PARTS = {'docs', 'doc', 'examples', 'example'}
METADATA_SUFFIXES = (
    '.md',
    '.rst',
    '.txt',
    '.json',
    '.yml',
    '.yaml',
    '.toml',
    '.ini',
)


@dataclass(frozen=True)
class HunkLabel:
    path: str
    old_start: int
    old_length: int
    new_start: int
    new_length: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            'path': self.path,
            'old_start': self.old_start,
            'old_length': self.old_length,
            'new_start': self.new_start,
            'new_length': self.new_length,
        }


@dataclass(frozen=True)
class PatchLabels:
    gold_source_files: list[str]
    gold_hunks: list[HunkLabel]
    skipped_files: list[dict[str, str]]

    def as_expect_fields(self) -> dict[str, object]:
        return {
            'gold_source_files': self.gold_source_files,
            'gold_hunks': [hunk.as_dict() for hunk in self.gold_hunks],
        }


class PatchLabelError(ValueError):
    pass


def normalize_diff_path(path: str | None) -> str | None:
    if not path or path == '/dev/null':
        return None
    value = str(path).strip()
    if value.startswith(('a/', 'b/')):
        value = value[2:]
    value = value.strip('/')
    posix = PurePosixPath(value)
    if posix.is_absolute() or not value or any(part in {'', '.', '..'} for part in posix.parts):
        return None
    return posix.as_posix()


def is_source_label_path(path: str, *, source_suffixes: Sequence[str] = SOURCE_SUFFIXES) -> tuple[bool, str | None]:
    normalized = normalize_diff_path(path)
    if normalized is None:
        return False, 'unsafe_path'
    posix = PurePosixPath(normalized)
    lowered_parts = [part.lower() for part in posix.parts]
    lowered_name = posix.name.lower()
    lowered_path = normalized.lower()

    if not any(lowered_name.endswith(suffix) for suffix in source_suffixes):
        return False, 'unsupported_source_type'
    if any(part in TEST_PATH_PARTS for part in lowered_parts):
        return False, 'test_path'
    if lowered_name.startswith('test_') or lowered_name.endswith('_test.py'):
        return False, 'test_path'
    if any(part in DOC_PATH_PARTS for part in lowered_parts):
        return False, 'docs_or_examples'
    if lowered_name.startswith(('changelog', 'release')):
        return False, 'metadata_or_release_note'
    if any(lowered_path.endswith(suffix) for suffix in METADATA_SUFFIXES):
        return False, 'metadata_or_release_note'
    return True, None


def _patch_set(patch_text: str):
    try:
        from unidiff import PatchSet
    except ImportError as exc:
        raise PatchLabelError('unidiff is required to parse SWE-bench patches') from exc
    try:
        return PatchSet(patch_text.splitlines(keepends=True))
    except Exception as exc:  # unidiff raises several parser-specific errors.
        raise PatchLabelError(f'could not parse unified diff: {exc}') from exc


def _patched_path(patched_file: object) -> str | None:
    is_removed = bool(getattr(patched_file, 'is_removed_file', False))
    source = normalize_diff_path(str(getattr(patched_file, 'source_file', '') or ''))
    target = normalize_diff_path(str(getattr(patched_file, 'target_file', '') or ''))
    return source if is_removed else (target or source)


def labels_from_patch(
    patch_text: str,
    *,
    source_suffixes: Sequence[str] = SOURCE_SUFFIXES,
) -> PatchLabels:
    if not patch_text or not patch_text.strip():
        raise PatchLabelError('patch is empty')

    source_files: list[str] = []
    hunks: list[HunkLabel] = []
    skipped: list[dict[str, str]] = []
    for patched_file in _patch_set(patch_text):
        path = _patched_path(patched_file)
        if path is None:
            skipped.append({'path': str(getattr(patched_file, 'path', '') or ''), 'reason': 'unsafe_path'})
            continue
        keep, reason = is_source_label_path(path, source_suffixes=source_suffixes)
        if not keep:
            skipped.append({'path': path, 'reason': reason or 'filtered'})
            continue
        if path not in source_files:
            source_files.append(path)
        for hunk in patched_file:
            hunks.append(
                HunkLabel(
                    path=path,
                    old_start=int(getattr(hunk, 'source_start', 0) or 0),
                    old_length=int(getattr(hunk, 'source_length', 0) or 0),
                    new_start=int(getattr(hunk, 'target_start', 0) or 0),
                    new_length=int(getattr(hunk, 'target_length', 0) or 0),
                )
            )
    return PatchLabels(gold_source_files=source_files, gold_hunks=hunks, skipped_files=skipped)


def has_usable_source_label(patch_text: str, *, source_suffixes: Iterable[str] = SOURCE_SUFFIXES) -> bool:
    try:
        return bool(labels_from_patch(patch_text, source_suffixes=tuple(source_suffixes)).gold_source_files)
    except PatchLabelError:
        return False

