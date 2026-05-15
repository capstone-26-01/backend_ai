from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any
import fcntl
import re
import shutil
import subprocess

from django.conf import settings


REPO_SEGMENT_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+$')
MAX_STDERR_CHARS = 1200


class RepoIngestionError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        command_category: str | None = None,
        stderr: str = '',
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.command_category = command_category
        self.stderr = stderr[:MAX_STDERR_CHARS]
        self.metadata = metadata or {}

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'code': self.code,
            'message': self.message,
        }
        if self.command_category:
            payload['command_category'] = self.command_category
        if self.metadata:
            payload['metadata'] = self.metadata
        return payload


def _is_safe_repo_segment(segment: str) -> bool:
    return bool(segment) and segment not in {'.', '..'} and REPO_SEGMENT_PATTERN.fullmatch(segment) is not None


def _repo_parts(repo_path: str) -> tuple[str, str]:
    parts = repo_path.split('/')
    if len(parts) != 2:
        raise ValueError('Unsafe repo path')
    owner, repo = parts
    if not _is_safe_repo_segment(owner) or not _is_safe_repo_segment(repo) or repo.endswith('.git'):
        raise ValueError('Unsafe repo path')
    return owner, repo


def _safe_subpath(base_dir: Path, *parts: str) -> Path:
    candidate = (base_dir / Path(*parts)).resolve(strict=False)
    resolved_base = base_dir.resolve()
    if candidate != resolved_base and resolved_base not in candidate.parents:
        raise ValueError('Path escaped base directory')
    return candidate


def _repo_clone_url(repo_path: str) -> str:
    return f'https://github.com/{repo_path}.git'


def _repo_dir(repo_path: str) -> Path:
    owner, repo = _repo_parts(repo_path)
    return _safe_subpath(settings.PLAYGROUND_DIR, owner, repo)


def _repo_lock_path(repo_path: str) -> Path:
    owner, repo = _repo_parts(repo_path)
    return _safe_subpath(settings.TEMP_DIR, 'locks', owner, f'{repo}.lock')


@contextmanager
def _repo_lock(repo_path: str) -> Iterator[None]:
    lock_path = _repo_lock_path(repo_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open('w', encoding='utf-8') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_path.unlink(missing_ok=True)
        if lock_path.parent.exists() and not any(lock_path.parent.iterdir()):
            lock_path.parent.rmdir()


def _setting_int(name: str, default: int) -> int:
    return int(getattr(settings, name, default))


def _git_timeout() -> int:
    return _setting_int('GITHUB_REPO_GIT_TIMEOUT_SECONDS', 30)


def _sanitize_stderr(stderr: str) -> str:
    sanitized = re.sub(r'https://[^@\s]+@github\.com/', 'https://github.com/', stderr)
    return sanitized[:MAX_STDERR_CHARS]


def _classify_git_error(args: tuple[str, ...], stderr: str, returncode: int) -> RepoIngestionError:
    category = args[0] if args else 'git'
    lowered = stderr.lower()
    if 'authentication failed' in lowered or 'could not read username' in lowered or 'permission denied' in lowered:
        return RepoIngestionError(
            'private_repo',
            '레포 접근 권한이 없거나 private repository입니다.',
            command_category=category,
            stderr=stderr,
            metadata={'returncode': returncode},
        )
    if 'repository not found' in lowered or ('not found' in lowered and category in {'clone', 'fetch'}):
        return RepoIngestionError(
            'repo_not_found',
            '레포를 찾을 수 없습니다.',
            command_category=category,
            stderr=stderr,
            metadata={'returncode': returncode},
        )
    return RepoIngestionError(
        'git_error',
        'Git 명령 실행 중 오류가 발생했습니다.',
        command_category=category,
        stderr=stderr,
        metadata={'returncode': returncode},
    )


def _run_git(*args: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ['git', *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=_git_timeout(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RepoIngestionError(
            'timeout',
            'Git 명령 시간이 초과되었습니다.',
            command_category=args[0] if args else 'git',
            stderr=_sanitize_stderr((exc.stderr or '') if isinstance(exc.stderr, str) else ''),
            metadata={'timeout_seconds': _git_timeout()},
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = _sanitize_stderr(exc.stderr or '')
        raise _classify_git_error(tuple(args), stderr, exc.returncode) from exc
    return result.stdout.strip()


def _run_git_raw(*args: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ['git', *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=_git_timeout(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RepoIngestionError(
            'timeout',
            'Git 명령 시간이 초과되었습니다.',
            command_category=args[0] if args else 'git',
            stderr=_sanitize_stderr((exc.stderr or '') if isinstance(exc.stderr, str) else ''),
            metadata={'timeout_seconds': _git_timeout()},
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = _sanitize_stderr(exc.stderr or '')
        raise _classify_git_error(tuple(args), stderr, exc.returncode) from exc
    return result.stdout


def _default_remote_ref(repo_dir: Path) -> str:
    try:
        return _run_git('symbolic-ref', '--short', 'refs/remotes/origin/HEAD', cwd=repo_dir)
    except RepoIngestionError as error:
        if error.code != 'git_error':
            raise
        branch_name = _run_git('rev-parse', '--abbrev-ref', 'HEAD', cwd=repo_dir)
        return f'origin/{branch_name}'


def _origin_matches(repo_dir: Path, repo_path: str) -> bool:
    try:
        remote_url = _run_git('remote', 'get-url', 'origin', cwd=repo_dir)
    except RepoIngestionError as error:
        if error.code != 'git_error':
            raise
        return False

    expected_urls = {
        _repo_clone_url(repo_path),
        _repo_clone_url(repo_path).removesuffix('.git'),
    }
    return remote_url in expected_urls


def _ensure_local_repo(repo_path: str) -> Path:
    repo_dir = _repo_dir(repo_path)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    if repo_dir.exists() and not _origin_matches(repo_dir, repo_path):
        shutil.rmtree(repo_dir)

    if not repo_dir.exists():
        _run_git('clone', '--depth', '1', _repo_clone_url(repo_path), str(repo_dir))
        return repo_dir

    _run_git('fetch', '--depth', '1', '--prune', 'origin', cwd=repo_dir)
    default_remote_ref = _default_remote_ref(repo_dir)
    default_branch = default_remote_ref.removeprefix('origin/')
    _run_git('checkout', default_branch, cwd=repo_dir)
    _run_git('reset', '--hard', default_remote_ref, cwd=repo_dir)
    _run_git('clean', '-fd', cwd=repo_dir)
    return repo_dir


def _enforce_snapshot_limits(files: list[str]) -> None:
    max_files = _setting_int('GITHUB_REPO_MAX_FILES', 5000)
    max_python_files = _setting_int('GITHUB_REPO_MAX_PYTHON_FILES', 1000)
    python_files = [file_path for file_path in files if file_path.endswith('.py')]

    if len(files) > max_files:
        raise RepoIngestionError(
            'too_large',
            '레포 파일 수가 허용 한도를 초과했습니다.',
            metadata={'limit': max_files, 'actual': len(files), 'limit_type': 'max_files'},
        )
    if len(python_files) > max_python_files:
        raise RepoIngestionError(
            'too_large',
            'Python 파일 수가 허용 한도를 초과했습니다.',
            metadata={'limit': max_python_files, 'actual': len(python_files), 'limit_type': 'max_python_files'},
        )


def _repo_snapshot(repo_path: str) -> tuple[Path, str, list[str]]:
    with _repo_lock(repo_path):
        repo_dir = _ensure_local_repo(repo_path)
        revision = _run_git('rev-parse', 'HEAD', cwd=repo_dir)
        files_output = _run_git('ls-tree', '-r', '--name-only', 'HEAD', cwd=repo_dir)
    files = [line for line in files_output.splitlines() if line]
    sorted_files = sorted(files)
    _enforce_snapshot_limits(sorted_files)
    return repo_dir, revision, sorted_files


def get_repo_snapshot_or_raise(repo_path: str) -> tuple[str, list[str]]:
    try:
        _repo_dir, revision, files = _repo_snapshot(repo_path)
    except ValueError as exc:
        raise RepoIngestionError('invalid_repo_path', '올바른 repo 경로가 아닙니다.') from exc
    return revision, files


def get_repo_snapshot(repo_path: str) -> tuple[str, list[str]] | None:
    try:
        return get_repo_snapshot_or_raise(repo_path)
    except RepoIngestionError:
        return None


def _normalize_repo_file_path(file_path: str) -> str | None:
    path = PurePosixPath(file_path)
    if path.is_absolute() or '..' in path.parts:
        return None
    return path.as_posix()


def get_file_tree_or_raise(repo_path: str) -> list[str]:
    snapshot = get_repo_snapshot_or_raise(repo_path)
    _revision, files = snapshot
    return files


def get_file_tree(repo_path: str) -> list[str] | None:
    try:
        return get_file_tree_or_raise(repo_path)
    except RepoIngestionError:
        return None


def _file_size_at_revision(repo_dir: Path, revision: str, file_path: str) -> int:
    return int(_run_git('cat-file', '-s', f'{revision}:{file_path}', cwd=repo_dir))


def get_file_content_or_raise(repo_path: str, file_path: str, revision: str | None = None) -> str | None:
    normalized_file_path = _normalize_repo_file_path(file_path)
    if normalized_file_path is None:
        raise RepoIngestionError('unsafe_path', '레포 파일 경로가 안전하지 않습니다.')

    try:
        repo_dir = _repo_dir(repo_path)
        if not repo_dir.exists() or revision is None:
            repo_dir, current_revision, _files = _repo_snapshot(repo_path)
            target_revision = revision or current_revision
        else:
            target_revision = revision
    except ValueError as exc:
        raise RepoIngestionError('invalid_repo_path', '올바른 repo 경로가 아닙니다.') from exc

    file_size = _file_size_at_revision(repo_dir, target_revision, normalized_file_path)
    max_file_size = _setting_int('GITHUB_REPO_MAX_SINGLE_FILE_BYTES', 300_000)
    if file_size > max_file_size:
        raise RepoIngestionError(
            'too_large',
            '분석 대상 파일이 허용 크기를 초과했습니다.',
            metadata={'limit': max_file_size, 'actual': file_size, 'limit_type': 'max_single_file_bytes', 'path': normalized_file_path},
        )
    return _run_git_raw('show', f'{target_revision}:{normalized_file_path}', cwd=repo_dir)


def get_file_content(repo_path: str, file_path: str, revision: str | None = None) -> str | None:
    try:
        return get_file_content_or_raise(repo_path, file_path, revision)
    except RepoIngestionError:
        return None


def get_repo_revision(repo_path: str) -> str | None:
    snapshot = get_repo_snapshot(repo_path)
    if snapshot is None:
        return None
    revision, _files = snapshot
    return revision
