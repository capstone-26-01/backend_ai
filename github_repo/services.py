from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any
import fcntl
import os
import re
import shutil
import subprocess

from django.conf import settings
import requests


REPO_SEGMENT_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+$')
REVISION_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+$')
MAX_STDERR_CHARS = 1200
MAX_GITHUB_ERROR_BODY_CHARS = 1200
MAX_GITHUB_ISSUE_BODY_EXCERPT_CHARS = 240


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


class GithubIssueApiError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 502,
        upstream_status: int | None = None,
        metadata: dict[str, Any] | None = None,
        rate_limit: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.upstream_status = upstream_status
        self.metadata = metadata or {}
        self.rate_limit = rate_limit

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'code': self.code,
            'message': self.message,
        }
        if self.upstream_status is not None:
            payload['upstream_status'] = self.upstream_status
        if self.metadata:
            payload['metadata'] = self.metadata
        if self.rate_limit is not None:
            payload['rate_limit'] = self.rate_limit
        return payload


def _is_safe_repo_segment(segment: str) -> bool:
    return bool(segment) and segment not in {'.', '..'} and REPO_SEGMENT_PATTERN.fullmatch(segment) is not None


def _is_safe_revision(revision: str) -> bool:
    if not revision or len(revision) > 255 or revision.startswith('-'):
        return False
    return revision not in {'.', '..'} and REVISION_PATTERN.fullmatch(revision) is not None


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


def _github_api_timeout() -> int:
    return _setting_int('GITHUB_API_TIMEOUT_SECONDS', 10)


def _github_api_base_url() -> str:
    return str(getattr(settings, 'GITHUB_API_BASE_URL', 'https://api.github.com')).rstrip('/')


def _github_token() -> str:
    return str(getattr(settings, 'GITHUB_TOKEN', '') or os.getenv('GITHUB_TOKEN', '')).strip()


def _github_headers() -> dict[str, str]:
    headers = {
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    token = _github_token()
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return headers


def _response_header(response: Any, name: str) -> str:
    headers = getattr(response, 'headers', {}) or {}
    if hasattr(headers, 'get'):
        value = headers.get(name) or headers.get(name.lower()) or headers.get(name.upper())
        if value is not None:
            return str(value)
    lowered_name = name.lower()
    for key, value in dict(headers).items():
        if str(key).lower() == lowered_name:
            return str(value)
    return ''


def _int_response_header(response: Any, name: str) -> int | None:
    value = _response_header(response, name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _github_rate_limit(response: Any) -> dict[str, int | None] | None:
    rate_limit = {
        'limit': _int_response_header(response, 'X-RateLimit-Limit'),
        'remaining': _int_response_header(response, 'X-RateLimit-Remaining'),
        'reset': _int_response_header(response, 'X-RateLimit-Reset'),
        'used': _int_response_header(response, 'X-RateLimit-Used'),
    }
    if all(value is None for value in rate_limit.values()):
        return None
    return rate_limit


def _response_error_body(response: Any) -> str:
    text = str(getattr(response, 'text', '') or '')
    return text[:MAX_GITHUB_ERROR_BODY_CHARS]


def _github_error_from_response(
    response: Any,
    repo_path: str,
    *,
    resource: str = 'repository',
    issue_number: int | None = None,
) -> GithubIssueApiError:
    upstream_status = int(getattr(response, 'status_code', 0) or 0)
    rate_limit = _github_rate_limit(response)
    metadata = {'repo': repo_path, 'response_body': _response_error_body(response)}
    if issue_number is not None:
        metadata['issue_number'] = issue_number

    if upstream_status == 404:
        if resource in {'issue', 'comments'}:
            return GithubIssueApiError(
                'issue_not_found',
                'Issue를 찾을 수 없습니다.',
                status_code=404,
                upstream_status=upstream_status,
                metadata=metadata,
                rate_limit=rate_limit,
            )
        return GithubIssueApiError(
            'repo_not_found',
            '레포를 찾을 수 없거나 private repository입니다.',
            status_code=404,
            upstream_status=upstream_status,
            metadata=metadata,
            rate_limit=rate_limit,
        )
    if upstream_status == 403 and rate_limit and rate_limit.get('remaining') == 0:
        return GithubIssueApiError(
            'github_rate_limited',
            'GitHub API rate limit이 초과되었습니다.',
            status_code=429,
            upstream_status=upstream_status,
            metadata=metadata,
            rate_limit=rate_limit,
        )
    if upstream_status in {401, 403}:
        return GithubIssueApiError(
            'private_repo',
            '레포 접근 권한이 없거나 private repository입니다.',
            status_code=403,
            upstream_status=upstream_status,
            metadata=metadata,
            rate_limit=rate_limit,
        )
    return GithubIssueApiError(
        'github_issue_api_error',
        'GitHub issue API 호출 중 오류가 발생했습니다.',
        status_code=502,
        upstream_status=upstream_status,
        metadata=metadata,
        rate_limit=rate_limit,
    )


def _github_get_json(
    path: str,
    *,
    repo_path: str,
    params: dict[str, Any] | None = None,
    resource: str = 'repository',
    issue_number: int | None = None,
) -> tuple[Any, Any]:
    url = f'{_github_api_base_url()}{path}'
    try:
        response = requests.get(url, headers=_github_headers(), params=params, timeout=_github_api_timeout())
    except requests.RequestException as exc:
        raise GithubIssueApiError(
            'github_issue_api_unavailable',
            'GitHub issue API 호출 중 네트워크 오류가 발생했습니다.',
            status_code=502,
            metadata={'repo': repo_path, 'error': str(exc)[:MAX_GITHUB_ERROR_BODY_CHARS]},
        ) from exc

    if not getattr(response, 'ok', False):
        raise _github_error_from_response(response, repo_path, resource=resource, issue_number=issue_number)

    try:
        payload = response.json()
    except ValueError as exc:
        raise GithubIssueApiError(
            'github_issue_api_error',
            'GitHub issue API 응답을 해석할 수 없습니다.',
            status_code=502,
            upstream_status=int(getattr(response, 'status_code', 0) or 0),
            metadata={'repo': repo_path},
            rate_limit=_github_rate_limit(response),
        ) from exc
    return payload, response


def fetch_github_issue_list_page(repo_path: str, *, page: int = 1, per_page: int = 30, state: str = 'open') -> tuple[Any, Any]:
    owner, repo = _repo_parts(repo_path)
    return _github_get_json(
        f'/repos/{owner}/{repo}/issues',
        repo_path=repo_path,
        params={'state': state, 'page': page, 'per_page': per_page},
    )


def fetch_github_issue_detail(repo_path: str, issue_number: int) -> tuple[Any, Any]:
    owner, repo = _repo_parts(repo_path)
    return _github_get_json(
        f'/repos/{owner}/{repo}/issues/{issue_number}',
        repo_path=repo_path,
        resource='issue',
        issue_number=issue_number,
    )


def fetch_github_issue_comments_page(
    repo_path: str,
    issue_number: int,
    *,
    page: int = 1,
    per_page: int = 30,
) -> tuple[Any, Any]:
    owner, repo = _repo_parts(repo_path)
    return _github_get_json(
        f'/repos/{owner}/{repo}/issues/{issue_number}/comments',
        repo_path=repo_path,
        params={'page': page, 'per_page': per_page},
        resource='comments',
        issue_number=issue_number,
    )


def _string(value: Any) -> str:
    if value is None:
        return ''
    return str(value)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_github_user(user: Any) -> dict[str, str] | None:
    if not isinstance(user, dict):
        return None
    return {
        'login': _string(user.get('login')),
        'avatar_url': _string(user.get('avatar_url')),
        'html_url': _string(user.get('html_url')),
    }


def _normalize_github_label(label: Any) -> dict[str, str | None]:
    if not isinstance(label, dict):
        return {'name': '', 'color': '', 'description': None}
    description = label.get('description')
    return {
        'name': _string(label.get('name')),
        'color': _string(label.get('color')),
        'description': None if description is None else str(description),
    }


def _normalize_github_user_list(users: Any) -> list[dict[str, str]]:
    if not isinstance(users, list):
        return []
    normalized = [_normalize_github_user(user) for user in users]
    return [user for user in normalized if user is not None]


def _normalize_github_label_list(labels: Any) -> list[dict[str, str | None]]:
    if not isinstance(labels, list):
        return []
    return [_normalize_github_label(label) for label in labels]


def _body_excerpt(body: Any) -> tuple[str, bool]:
    body_text = _string(body)
    excerpt = body_text[:MAX_GITHUB_ISSUE_BODY_EXCERPT_CHARS]
    return excerpt, len(body_text) > MAX_GITHUB_ISSUE_BODY_EXCERPT_CHARS


def normalize_github_issue(repo_path: str, issue: Any) -> dict[str, Any]:
    if not isinstance(issue, dict):
        raise GithubIssueApiError(
            'github_issue_api_error',
            'GitHub issue API 응답 형식이 올바르지 않습니다.',
            status_code=502,
            metadata={'repo': repo_path, 'payload_type': type(issue).__name__},
        )

    number = _int(issue.get('number'))
    body_excerpt, body_truncated = _body_excerpt(issue.get('body'))
    return {
        'key': f'github:{repo_path}#{number}',
        'number': number,
        'title': _string(issue.get('title')),
        'state': _string(issue.get('state') or 'open'),
        'html_url': _string(issue.get('html_url') or f'https://github.com/{repo_path}/issues/{number}'),
        'author': _normalize_github_user(issue.get('user')),
        'labels': _normalize_github_label_list(issue.get('labels')),
        'assignees': _normalize_github_user_list(issue.get('assignees')),
        'comments_count': _int(issue.get('comments')),
        'created_at': _string(issue.get('created_at')),
        'updated_at': _string(issue.get('updated_at')),
        'body_excerpt': body_excerpt,
        'body_truncated': body_truncated,
        'locked': bool(issue.get('locked')),
        'is_pull_request': isinstance(issue.get('pull_request'), dict),
    }


def normalize_github_issue_comment(comment: Any) -> dict[str, Any]:
    if not isinstance(comment, dict):
        raise GithubIssueApiError(
            'github_issue_api_error',
            'GitHub issue comment API 응답 형식이 올바르지 않습니다.',
            status_code=502,
            metadata={'payload_type': type(comment).__name__},
        )
    return {
        'id': _int(comment.get('id')),
        'author': _normalize_github_user(comment.get('user')),
        'body': _string(comment.get('body')),
        'created_at': _string(comment.get('created_at')),
        'updated_at': _string(comment.get('updated_at')),
        'html_url': _string(comment.get('html_url')),
    }


def get_github_issue_detail_response(repo_path: str, issue_number: int) -> dict[str, Any]:
    payload, _response = fetch_github_issue_detail(repo_path, issue_number)
    if not isinstance(payload, dict):
        raise GithubIssueApiError(
            'github_issue_api_error',
            'GitHub issue API 응답 형식이 올바르지 않습니다.',
            status_code=502,
            metadata={'repo': repo_path, 'issue_number': issue_number, 'payload_type': type(payload).__name__},
        )
    if _issue_is_pull_request(payload):
        raise GithubIssueApiError(
            'issue_not_found',
            'Pull Request는 issue map 대상으로 지원하지 않습니다.',
            status_code=404,
            metadata={'repo': repo_path, 'issue_number': issue_number},
        )
    issue = normalize_github_issue(repo_path, payload)
    issue['body'] = _string(payload.get('body'))
    return issue


def get_github_issue_comments_response(
    repo_path: str,
    issue_number: int,
    *,
    max_comments: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_comments = max(0, min(max_comments, 100))
    if max_comments == 0:
        return [], []

    payload, response = fetch_github_issue_comments_page(repo_path, issue_number, page=1, per_page=max_comments)
    if not isinstance(payload, list):
        raise GithubIssueApiError(
            'github_issue_api_error',
            'GitHub issue comment API 응답 형식이 올바르지 않습니다.',
            status_code=502,
            metadata={'repo': repo_path, 'issue_number': issue_number, 'payload_type': type(payload).__name__},
            rate_limit=_github_rate_limit(response),
        )
    comments = [normalize_github_issue_comment(comment) for comment in payload[:max_comments]]
    warnings = []
    if _has_next_page(response):
        warnings.append(
            {
                'code': 'github_comments_truncated',
                'message': 'GitHub issue comment가 max_comments 한도를 초과해 일부만 사용했습니다.',
                'max_comments': max_comments,
            }
        )
    return comments, warnings


def _issue_is_pull_request(issue: Any) -> bool:
    return isinstance(issue, dict) and isinstance(issue.get('pull_request'), dict)


def _has_next_page(response: Any) -> bool:
    link_header = _response_header(response, 'Link')
    return bool(re.search(r'rel="?next"?', link_header))


def get_github_issue_list_response(repo_path: str, *, page: int = 1, per_page: int = 30, state: str = 'open') -> dict[str, Any]:
    payload, response = fetch_github_issue_list_page(repo_path, page=page, per_page=per_page, state=state)
    if not isinstance(payload, list):
        raise GithubIssueApiError(
            'github_issue_api_error',
            'GitHub issue API 응답 형식이 올바르지 않습니다.',
            status_code=502,
            metadata={'repo': repo_path, 'payload_type': type(payload).__name__},
            rate_limit=_github_rate_limit(response),
        )

    issue_payloads = [issue for issue in payload if not _issue_is_pull_request(issue)]
    normalized_issues = [normalize_github_issue(repo_path, issue) for issue in issue_payloads]
    has_next_page = _has_next_page(response)
    return {
        'repo': repo_path,
        'repository': {
            'full_name': repo_path,
            'html_url': f'https://github.com/{repo_path}',
            'api_url': f'{_github_api_base_url()}/repos/{repo_path}',
        },
        'provider': 'github',
        'source': 'github',
        'mock': False,
        'state': state,
        'page': page,
        'per_page': per_page,
        'has_next_page': has_next_page,
        'next_page': page + 1 if has_next_page else None,
        'issues': normalized_issues,
        'rate_limit': _github_rate_limit(response),
        'warnings': [],
    }


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
    if category == 'fetch' and (
        "couldn't find remote ref" in lowered
        or 'server does not allow request for unadvertised object' in lowered
        or 'not our ref' in lowered
    ):
        return RepoIngestionError(
            'revision_not_found',
            'revision을 찾을 수 없습니다.',
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


def _revision_fetch_depth() -> int:
    return max(1, _setting_int('GITHUB_REPO_REVISION_FETCH_DEPTH', 50))


def _resolve_commit(repo_dir: Path, revision: str) -> str | None:
    try:
        return _run_git('rev-parse', '--verify', f'{revision}^{{commit}}', cwd=repo_dir)
    except RepoIngestionError as error:
        if error.code in {'git_error', 'revision_not_found'}:
            return None
        raise


def _fetch_revision_candidate(repo_dir: Path, revision: str) -> str | None:
    try:
        _run_git('fetch', '--depth', '1', 'origin', revision, cwd=repo_dir)
    except RepoIngestionError as error:
        if error.code not in {'git_error', 'revision_not_found'}:
            raise
        return None
    return _resolve_commit(repo_dir, 'FETCH_HEAD')


def _fetch_default_history_for_revision(repo_dir: Path, revision: str) -> str | None:
    default_remote_ref = _default_remote_ref(repo_dir)
    default_branch = default_remote_ref.removeprefix('origin/')
    try:
        _run_git('fetch', f'--deepen={_revision_fetch_depth()}', 'origin', default_branch, cwd=repo_dir)
    except RepoIngestionError as error:
        if error.code not in {'git_error', 'revision_not_found'}:
            raise
    return _resolve_commit(repo_dir, revision)


def _ensure_revision_available(repo_dir: Path, revision: str) -> str:
    resolved_revision = _resolve_commit(repo_dir, revision)
    if resolved_revision is not None:
        return resolved_revision

    resolved_revision = _fetch_revision_candidate(repo_dir, revision)
    if resolved_revision is not None:
        return resolved_revision

    resolved_revision = _fetch_default_history_for_revision(repo_dir, revision)
    if resolved_revision is not None:
        return resolved_revision

    raise RepoIngestionError(
        'revision_not_found',
        'revision을 찾을 수 없습니다.',
        command_category='fetch',
        metadata={'revision': revision},
    )


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


def _repo_snapshot_at_revision(repo_path: str, revision: str) -> tuple[Path, str, list[str]]:
    if not _is_safe_revision(revision):
        raise ValueError('Unsafe revision')

    with _repo_lock(repo_path):
        repo_dir = _ensure_local_repo(repo_path)
        target_revision = _ensure_revision_available(repo_dir, revision)
        files_output = _run_git('ls-tree', '-r', '--name-only', target_revision, cwd=repo_dir)
    files = [line for line in files_output.splitlines() if line]
    sorted_files = sorted(files)
    _enforce_snapshot_limits(sorted_files)
    return repo_dir, target_revision, sorted_files


def get_repo_snapshot_or_raise(repo_path: str) -> tuple[str, list[str]]:
    try:
        _repo_dir, revision, files = _repo_snapshot(repo_path)
    except ValueError as exc:
        raise RepoIngestionError('invalid_repo_path', '올바른 repo 경로가 아닙니다.') from exc
    return revision, files


def get_repo_snapshot_at_revision_or_raise(repo_path: str, revision: str) -> tuple[str, list[str]]:
    try:
        _repo_dir, target_revision, files = _repo_snapshot_at_revision(repo_path, revision)
    except ValueError as exc:
        raise RepoIngestionError('invalid_repo_path', '올바른 repo 경로 또는 revision이 아닙니다.') from exc
    return target_revision, files


def get_repo_snapshot_at_revision(repo_path: str, revision: str) -> tuple[str, list[str]] | None:
    try:
        return get_repo_snapshot_at_revision_or_raise(repo_path, revision)
    except RepoIngestionError:
        return None


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
