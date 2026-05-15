from rest_framework import serializers
from urllib.parse import urlparse
import re


REPO_SEGMENT_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+$')
REVISION_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+$')
UNSAFE_REF_PATTERN = re.compile(r'[\s\\~^:?*\[\]\x00-\x1f]')


def _is_safe_repo_segment(segment: str) -> bool:
    return bool(segment) and segment not in {'.', '..'} and REPO_SEGMENT_PATTERN.fullmatch(segment) is not None


def is_safe_revision(revision: str) -> bool:
    return bool(revision) and revision not in {'.', '..'} and REVISION_PATTERN.fullmatch(revision) is not None


def is_safe_ref(ref: str) -> bool:
    if not ref or len(ref) > 255:
        return False
    if ref.startswith(('/', '-')) or ref.endswith('/') or ref.endswith('.lock'):
        return False
    if '..' in ref or '@{' in ref or '//' in ref or UNSAFE_REF_PATTERN.search(ref):
        return False
    return all(part not in {'.', '..'} for part in ref.split('/'))


def extract_repo_path(repo_url):
    parsed = urlparse(repo_url)
    if parsed.scheme not in {'http', 'https'} or parsed.netloc != 'github.com':
        return None

    path_parts = [part for part in parsed.path.strip('/').split('/') if part]
    if len(path_parts) != 2:
        return None

    owner, repo = path_parts
    if repo.endswith('.git'):
        return None

    if not _is_safe_repo_segment(owner) or not _is_safe_repo_segment(repo):
        return None

    return f'{owner}/{repo}'


class RepoUrlSerializer(serializers.Serializer):
    repo_url = serializers.CharField()

    def validate_repo_url(self, value):
        repo_path = extract_repo_path(value)
        if not repo_path:
            raise serializers.ValidationError('올바른 GitHub URL이 아닙니다')
        return repo_path  # 검증 후 repo_path로 변환해서 반환


class AnalysisRequestSerializer(RepoUrlSerializer):
    revision = serializers.CharField(required=False)

    def validate_revision(self, value):
        if not is_safe_revision(value):
            raise serializers.ValidationError('올바른 revision이 아닙니다')
        return value


class QASerializer(serializers.Serializer):
    repo_url = serializers.CharField()
    question = serializers.CharField()
    revision = serializers.CharField(required=False)

    def validate_repo_url(self, value):
        repo_path = extract_repo_path(value)
        if not repo_path:
            raise serializers.ValidationError('올바른 GitHub URL이 아닙니다')
        return repo_path

    def validate_revision(self, value):
        if not is_safe_revision(value):
            raise serializers.ValidationError('올바른 revision이 아닙니다')
        return value
