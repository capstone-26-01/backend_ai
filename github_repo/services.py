from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import fcntl
import re
import shutil
import subprocess

from django.conf import settings


REPO_SEGMENT_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+$')


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


def _run_git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ['git', *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run_git_raw(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ['git', *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _default_remote_ref(repo_dir: Path) -> str:
    try:
        return _run_git('symbolic-ref', '--short', 'refs/remotes/origin/HEAD', cwd=repo_dir)
    except subprocess.CalledProcessError:
        branch_name = _run_git('rev-parse', '--abbrev-ref', 'HEAD', cwd=repo_dir)
        return f'origin/{branch_name}'


def _origin_matches(repo_dir: Path, repo_path: str) -> bool:
    try:
        remote_url = _run_git('remote', 'get-url', 'origin', cwd=repo_dir)
    except subprocess.CalledProcessError:
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
        _run_git('clone', _repo_clone_url(repo_path), str(repo_dir))
        return repo_dir

    _run_git('fetch', '--prune', 'origin', cwd=repo_dir)
    default_remote_ref = _default_remote_ref(repo_dir)
    default_branch = default_remote_ref.removeprefix('origin/')
    _run_git('checkout', default_branch, cwd=repo_dir)
    _run_git('reset', '--hard', default_remote_ref, cwd=repo_dir)
    _run_git('clean', '-fd', cwd=repo_dir)
    return repo_dir


def _repo_snapshot(repo_path: str) -> tuple[Path, str, list[str]]:
    with _repo_lock(repo_path):
        repo_dir = _ensure_local_repo(repo_path)
        revision = _run_git('rev-parse', 'HEAD', cwd=repo_dir)
        files_output = _run_git('ls-tree', '-r', '--name-only', 'HEAD', cwd=repo_dir)
    files = [line for line in files_output.splitlines() if line]
    return repo_dir, revision, sorted(files)


def get_repo_snapshot(repo_path: str) -> tuple[str, list[str]] | None:
    try:
        _repo_dir, revision, files = _repo_snapshot(repo_path)
    except (subprocess.CalledProcessError, ValueError):
        return None
    return revision, files


def _normalize_repo_file_path(file_path: str) -> str | None:
    path = PurePosixPath(file_path)
    if path.is_absolute() or '..' in path.parts:
        return None
    return path.as_posix()


def get_file_tree(repo_path: str) -> list[str] | None:
    snapshot = get_repo_snapshot(repo_path)
    if snapshot is None:
        return None
    _revision, files = snapshot
    return files


def get_file_content(repo_path: str, file_path: str, revision: str | None = None) -> str | None:
    normalized_file_path = _normalize_repo_file_path(file_path)
    if normalized_file_path is None:
        return None

    try:
        repo_dir = _repo_dir(repo_path)
        if not repo_dir.exists() or revision is None:
            repo_dir, current_revision, _files = _repo_snapshot(repo_path)
            target_revision = revision or current_revision
        else:
            target_revision = revision
    except (subprocess.CalledProcessError, ValueError):
        return None

    try:
        return _run_git_raw('show', f'{target_revision}:{normalized_file_path}', cwd=repo_dir)
    except subprocess.CalledProcessError:
        return None


def get_repo_revision(repo_path: str) -> str | None:
    snapshot = get_repo_snapshot(repo_path)
    if snapshot is None:
        return None
    revision, _files = snapshot
    return revision
