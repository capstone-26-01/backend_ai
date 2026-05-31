from rest_framework import serializers
from pathlib import PurePosixPath
from urllib.parse import urlparse
import re


REPO_SEGMENT_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+$')
REVISION_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+$')
GRAPH_ID_PATTERN = re.compile(r'^[A-Za-z0-9_./:-]+$')
SHARE_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]+$')
UNSAFE_REF_PATTERN = re.compile(r'[\s\\~^:?*\[\]\x00-\x1f]')


def _is_safe_repo_segment(segment: str) -> bool:
    return bool(segment) and segment not in {'.', '..'} and REPO_SEGMENT_PATTERN.fullmatch(segment) is not None


def is_safe_revision(revision: str) -> bool:
    if not revision or len(revision) > 255 or revision.startswith('-'):
        return False
    return revision not in {'.', '..'} and REVISION_PATTERN.fullmatch(revision) is not None


def is_safe_ref(ref: str) -> bool:
    if not ref or len(ref) > 255:
        return False
    if ref.startswith(('/', '-')) or ref.endswith('/') or ref.endswith('.lock'):
        return False
    if '..' in ref or '@{' in ref or '//' in ref or UNSAFE_REF_PATTERN.search(ref):
        return False
    return all(part not in {'.', '..'} for part in ref.split('/'))


def is_safe_graph_id(graph_id: str) -> bool:
    if not graph_id or len(graph_id) > 512 or GRAPH_ID_PATTERN.fullmatch(graph_id) is None:
        return False
    normalized_parts = graph_id.replace('::', '/').split('/')
    return all(part not in {'', '.', '..'} for part in normalized_parts)


def is_safe_repo_file_path(file_path: str) -> bool:
    if not file_path or len(file_path) > 1024 or '\\' in file_path or '\x00' in file_path:
        return False
    path = PurePosixPath(file_path)
    if path.is_absolute():
        return False
    return all(part not in {'', '.', '..'} for part in path.parts)


def is_safe_share_id(share_id: str) -> bool:
    return 16 <= len(share_id) <= 128 and SHARE_ID_PATTERN.fullmatch(share_id) is not None


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


class DiffRequestSerializer(RepoUrlSerializer):
    base = serializers.CharField()
    head = serializers.CharField(required=False)

    def validate_base(self, value):
        if not is_safe_revision(value):
            raise serializers.ValidationError('올바른 base revision이 아닙니다')
        return value

    def validate_head(self, value):
        if not is_safe_revision(value):
            raise serializers.ValidationError('올바른 head revision이 아닙니다')
        return value


class AnalysisDiffRequestSerializer(serializers.Serializer):
    base = serializers.IntegerField(min_value=1)


class ShareCreateSerializer(RepoUrlSerializer):
    mode = serializers.ChoiceField(choices=('fixed', 'latest'), required=False, default='fixed')
    revision = serializers.CharField(required=False)
    title = serializers.CharField(required=False, allow_blank=True, max_length=255)
    expires_at = serializers.DateTimeField(required=False, allow_null=True)

    def validate_revision(self, value):
        if not is_safe_revision(value):
            raise serializers.ValidationError('올바른 revision이 아닙니다')
        return value

    def validate(self, attrs):
        if attrs.get('mode') == 'latest' and attrs.get('revision'):
            raise serializers.ValidationError({'revision': ['latest share에는 revision을 지정할 수 없습니다']})
        return attrs


class QASerializer(serializers.Serializer):
    repo_url = serializers.CharField(required=False)
    question = serializers.CharField()
    revision = serializers.CharField(required=False)
    analysis_id = serializers.IntegerField(required=False, min_value=1)
    selected_node_id = serializers.CharField(required=False)
    selected_file_path = serializers.CharField(required=False)
    max_context_files = serializers.IntegerField(required=False, min_value=1, max_value=10, default=4)

    def validate_repo_url(self, value):
        repo_path = extract_repo_path(value)
        if not repo_path:
            raise serializers.ValidationError('올바른 GitHub URL이 아닙니다')
        return repo_path

    def validate_revision(self, value):
        if not is_safe_revision(value):
            raise serializers.ValidationError('올바른 revision이 아닙니다')
        return value

    def validate_selected_node_id(self, value):
        if not is_safe_graph_id(value):
            raise serializers.ValidationError('올바른 selected_node_id가 아닙니다')
        return value

    def validate_selected_file_path(self, value):
        if not is_safe_repo_file_path(value):
            raise serializers.ValidationError('올바른 selected_file_path가 아닙니다')
        return value

    def validate(self, attrs):
        if not attrs.get('repo_url') and not attrs.get('analysis_id'):
            raise serializers.ValidationError({'repo_url': ['repo_url 또는 analysis_id가 필요합니다']})
        return attrs


class SummaryRequestSerializer(serializers.Serializer):
    analysis_id = serializers.IntegerField(min_value=1)
    kind = serializers.ChoiceField(
        choices=('repo_overview', 'onboarding_guide'),
        required=False,
        default='repo_overview',
    )


class NodeSummaryRequestSerializer(serializers.Serializer):
    analysis_id = serializers.IntegerField(min_value=1)
    node_id = serializers.CharField()

    def validate_node_id(self, value):
        if not is_safe_graph_id(value):
            raise serializers.ValidationError('올바른 node_id가 아닙니다')
        return value


class IssueListRequestSerializer(RepoUrlSerializer):
    page = serializers.IntegerField(required=False, min_value=1, default=1)
    per_page = serializers.IntegerField(required=False, min_value=1, max_value=100, default=30)
    state = serializers.ChoiceField(choices=('open',), required=False, default='open')
    mock = serializers.BooleanField(
        required=False,
        default=False,
        help_text='true이면 live GitHub 대신 프런트엔드/테스트용 mock issue 목록을 반환합니다.',
    )


class IssueRelatedNodesRequestSerializer(serializers.Serializer):
    analysis_id = serializers.IntegerField(min_value=1)
    issue_number = serializers.IntegerField(min_value=1)
    max_nodes = serializers.IntegerField(required=False, min_value=1, max_value=20, default=8)
    mock = serializers.BooleanField(required=False, default=False, help_text='true이면 live GitHub issue detail/comment 대신 기존 mock issue map을 반환합니다.')
    include_comments = serializers.BooleanField(required=False, default=True, help_text='true이면 GitHub issue comments를 함께 읽어 ranking evidence에 반영합니다.')
    max_context_files = serializers.IntegerField(required=False, min_value=1, max_value=10, default=4, help_text='code_context에 포함할 최대 파일 수입니다.')


class IssueAuthorSerializer(serializers.Serializer):
    login = serializers.CharField(help_text='GitHub 사용자 login입니다.')
    avatar_url = serializers.URLField(help_text='사용자 avatar 이미지 URL입니다.')
    html_url = serializers.URLField(help_text='GitHub 사용자 프로필 URL입니다.')


class IssueLabelSerializer(serializers.Serializer):
    name = serializers.CharField(help_text='GitHub issue label 이름입니다.')
    color = serializers.CharField(help_text='GitHub label 색상 hex 값입니다. # 없이 내려옵니다.')
    description = serializers.CharField(allow_blank=True, allow_null=True, help_text='GitHub label 설명입니다. 없으면 null 또는 빈 문자열입니다.')


class IssueListItemSerializer(serializers.Serializer):
    key = serializers.CharField(help_text='프런트엔드 list key로 쓰기 좋은 안정 식별자입니다. 형식: github:{owner}/{repo}#{number}')
    number = serializers.IntegerField(help_text='GitHub issue 번호입니다. 관련 노드 추천 API의 issue_number로 그대로 넘깁니다.')
    title = serializers.CharField(help_text='Issue 제목입니다.')
    state = serializers.CharField(help_text='현재는 open issue만 반환합니다.')
    html_url = serializers.URLField(help_text='GitHub issue 페이지 URL입니다.')
    author = IssueAuthorSerializer(allow_null=True, help_text='Issue 작성자 정보입니다. 삭제된 사용자 등으로 알 수 없으면 null입니다.')
    labels = IssueLabelSerializer(many=True, help_text='Issue label 목록입니다.')
    assignees = IssueAuthorSerializer(many=True, help_text='Issue assignee 목록입니다.')
    comments_count = serializers.IntegerField(help_text='Issue comment 수입니다.')
    created_at = serializers.DateTimeField(help_text='Issue 생성 시각입니다.')
    updated_at = serializers.DateTimeField(help_text='Issue 마지막 수정 시각입니다.')
    body_excerpt = serializers.CharField(help_text='목록 화면 preview용 본문 요약입니다. 전체 body가 아닙니다.')
    body_truncated = serializers.BooleanField(help_text='true면 body_excerpt가 원문 일부만 담은 preview입니다.')
    locked = serializers.BooleanField(help_text='GitHub issue conversation 잠금 여부입니다.')
    is_pull_request = serializers.BooleanField(help_text='항상 false입니다. 실제 구현에서도 PR은 제외하고 issue만 반환합니다.')


class IssueRepositorySerializer(serializers.Serializer):
    full_name = serializers.CharField(help_text='정규화된 owner/repo 형식의 레포 이름입니다.')
    html_url = serializers.URLField(help_text='GitHub 레포 웹 URL입니다.')
    api_url = serializers.URLField(help_text='GitHub REST API 레포 URL입니다.')


class IssueRateLimitSerializer(serializers.Serializer):
    limit = serializers.IntegerField(allow_null=True, required=False, help_text='GitHub REST API rate limit 한도입니다.')
    remaining = serializers.IntegerField(allow_null=True, required=False, help_text='현재 남은 GitHub REST API 요청 수입니다.')
    reset = serializers.IntegerField(allow_null=True, required=False, help_text='GitHub rate limit reset epoch timestamp입니다.')
    used = serializers.IntegerField(allow_null=True, required=False, help_text='현재 window에서 사용한 요청 수입니다.')


class IssueListResponseSerializer(serializers.Serializer):
    repo = serializers.CharField(help_text='정규화된 owner/repo 형식의 레포 이름입니다.')
    repository = IssueRepositorySerializer(required=False, help_text='live GitHub 응답에서 제공되는 레포 metadata입니다.')
    provider = serializers.CharField(help_text='현재 provider입니다. GitHub repo만 지원하므로 github입니다.')
    source = serializers.CharField(help_text='현재 데이터 출처입니다. github이면 live GitHub, mock이면 실제 GitHub 조회가 아닙니다.')
    mock = serializers.BooleanField(help_text='true면 프런트엔드/테스트용 mock 응답입니다.')
    state = serializers.CharField(help_text='조회한 issue 상태입니다. 현재는 open만 지원합니다.')
    page = serializers.IntegerField(help_text='현재 페이지 번호입니다.')
    per_page = serializers.IntegerField(help_text='페이지당 issue 수입니다.')
    has_next_page = serializers.BooleanField(help_text='다음 페이지 존재 여부입니다.')
    next_page = serializers.IntegerField(allow_null=True, help_text='다음 페이지 번호입니다. 없으면 null입니다.')
    issues = IssueListItemSerializer(many=True, help_text='Open issue 목록입니다.')
    rate_limit = IssueRateLimitSerializer(allow_null=True, required=False, help_text='live GitHub 응답의 rate limit header 요약입니다.')
    warnings = serializers.JSONField(required=False, help_text='비치명적 경고 목록입니다. 현재 기본값은 빈 배열입니다.')


class IssueRefSerializer(serializers.Serializer):
    key = serializers.CharField(help_text='Issue 안정 식별자입니다. 형식: github:{owner}/{repo}#{number}')
    number = serializers.IntegerField(help_text='GitHub issue 번호입니다.')
    title = serializers.CharField(help_text='Issue 제목입니다.')
    state = serializers.CharField(help_text='Issue 상태입니다.')
    html_url = serializers.URLField(help_text='GitHub issue 페이지 URL입니다.')
    labels = IssueLabelSerializer(many=True, help_text='선택 issue label 목록입니다.')
    comments_count = serializers.IntegerField(help_text='선택 issue comment 수입니다.')
    updated_at = serializers.DateTimeField(help_text='선택 issue 마지막 수정 시각입니다.')
    body_excerpt = serializers.CharField(help_text='선택 issue preview 본문입니다.')


class IssueRelatedNodeSerializer(serializers.Serializer):
    id = serializers.CharField(help_text='Graph node ID입니다. /api/graph/ 응답의 nodes[].id와 같습니다.')
    kind = serializers.CharField(allow_blank=True, help_text='file, module, class, function, method, external 등 node 종류입니다.')
    label = serializers.CharField(allow_blank=True, help_text='화면에 표시하기 좋은 node 이름입니다.')
    path = serializers.CharField(allow_null=True, help_text='연결된 repo 내부 파일 경로입니다. 없으면 null입니다.')
    start_line = serializers.IntegerField(allow_null=True, help_text='파일 내 시작 줄입니다. 알 수 없으면 null입니다.')
    end_line = serializers.IntegerField(allow_null=True, help_text='파일 내 끝 줄입니다. 알 수 없으면 null입니다.')
    metadata = serializers.JSONField(help_text='Graph node 원본 metadata입니다.')


class IssueRelatedEvidenceSerializer(serializers.Serializer):
    type = serializers.CharField(help_text='근거 종류입니다. mock 응답에서는 mock 또는 graph_metadata입니다.')
    message = serializers.CharField(help_text='프런트엔드에서 표시할 수 있는 짧은 근거 설명입니다.')


class IssueRelatedNodeCandidateSerializer(serializers.Serializer):
    rank = serializers.IntegerField(help_text='추천 순위입니다. 1부터 시작합니다.')
    score = serializers.FloatField(help_text='0-1 범위의 관련도 점수입니다. mock에서는 deterministic placeholder입니다.')
    node_id = serializers.CharField(help_text='추천된 graph node ID입니다.')
    node = IssueRelatedNodeSerializer(help_text='프런트엔드 표시용 node 요약입니다.')
    reason = serializers.CharField(help_text='왜 이 node가 추천되었는지에 대한 짧은 설명입니다.')
    evidence = IssueRelatedEvidenceSerializer(many=True, help_text='추천 근거 목록입니다.')


class IssueRelatedNodesResponseSerializer(serializers.Serializer):
    analysis_id = serializers.IntegerField(help_text='추천 기준 분석 run ID입니다.')
    repo = serializers.CharField(help_text='분석 run이 속한 owner/repo입니다.')
    revision = serializers.CharField(help_text='분석 run의 git commit SHA입니다.')
    provider = serializers.CharField(help_text='현재 provider입니다. GitHub repo만 지원하므로 github입니다.')
    source = serializers.CharField(help_text='현재 데이터 출처입니다. mock이면 실제 GitHub issue/LLM 조회가 아닙니다.')
    mock = serializers.BooleanField(help_text='true면 프런트엔드 선작업용 mock 응답입니다.')
    issue = IssueRefSerializer(help_text='프런트엔드가 선택한 issue 정보입니다.')
    selected_node_ids = serializers.ListField(child=serializers.CharField(), help_text='그래프에서 바로 highlight할 node ID 목록입니다.')
    candidates = IssueRelatedNodeCandidateSerializer(many=True, help_text='관련도가 높다고 판단된 graph node 후보 목록입니다.')
    limits = serializers.JSONField(help_text='요청에서 적용된 max_nodes 등 제한값입니다.')
    warnings = serializers.JSONField(help_text='추천 생성 중 발생한 경고 목록입니다.')
    overview_graph = serializers.JSONField(required=False, help_text='레포 전체를 빠르게 이해하기 위한 축약 overview graph입니다.')
    focus_graph = serializers.JSONField(required=False, help_text='선택 issue 해결에 직접 관련된 후보/주변 노드 중심 graph입니다.')
    hypotheses = serializers.JSONField(required=False, help_text='Issue가 어디서 시작됐을 가능성이 높은지에 대한 deterministic 가설 목록입니다.')
    investigation_path = serializers.JSONField(required=False, help_text='첫 기여자가 순서대로 확인하면 좋은 node/file 조사 경로입니다.')
    code_context = serializers.JSONField(required=False, help_text='관련 후보 파일의 bounded code excerpt입니다.')
    confidence = serializers.JSONField(required=False, help_text='Deterministic ranking confidence 요약입니다.')
