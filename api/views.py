import html
import importlib
import json
import logging
from typing import Any, cast

from django.conf import settings
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators.clickjacking import xframe_options_exempt
from rest_framework.decorators import api_view
from rest_framework.decorators import renderer_classes
from rest_framework.decorators import throttle_classes
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiExample, extend_schema, OpenApiParameter, inline_serializer
from rest_framework import serializers

from github_repo.services import GithubIssueApiError, RepoIngestionError, get_file_tree_or_raise
from llm.services import answer_question, stream_answer_question
from .readme_svg import (
    DEFAULT_NODE_LIMIT,
    DEFAULT_SVG_HEIGHT,
    DEFAULT_SVG_WIDTH,
    MAX_NODE_LIMIT,
    MAX_SVG_HEIGHT,
    MAX_SVG_WIDTH,
    MIN_NODE_LIMIT,
    MIN_SVG_HEIGHT,
    MIN_SVG_WIDTH,
    normalize_svg_options,
    render_share_graph_svg,
)
from .renderers import SseRenderer, SvgRenderer
from .throttles import ShareCreateRateThrottle
from .serializers import (
    AnalysisDiffRequestSerializer,
    AnalysisRequestSerializer,
    DiffRequestSerializer,
    IssueListRequestSerializer,
    IssueListResponseSerializer,
    IssueRelatedNodesRequestSerializer,
    IssueRelatedNodesResponseSerializer,
    NodeSummaryRequestSerializer,
    RepoUrlSerializer,
    QASerializer,
    ShareCreateSerializer,
    SummaryRequestSerializer,
    is_safe_share_id,
    is_safe_revision,
)

logger = logging.getLogger(__name__)
api_services = importlib.import_module('api.services')


def get_repo_analysis(repo_path: str, revision: str | None = None, analysis_profile: str | None = None):
    return api_services.get_repo_analysis(repo_path, revision, analysis_profile=analysis_profile)


def get_analysis_response(repo_path: str, revision: str | None = None, analysis_profile: str | None = None):
    return api_services.get_analysis_response(repo_path, revision, analysis_profile=analysis_profile)


def get_analysis_response_by_id(analysis_id: int):
    return api_services.get_analysis_response_by_id(analysis_id)


def get_analysis_run_by_revision(repo_path: str, revision: str, analysis_profile: str | None = None):
    return api_services.get_analysis_run_by_revision(repo_path, revision, analysis_profile=analysis_profile)


def build_tree_response(payload, analysis_run=None):
    return api_services.build_tree_response(payload, analysis_run)


def build_graph_response(payload, analysis_run=None):
    return api_services.build_graph_response(payload, analysis_run)


def get_or_create_summary_response(analysis_id: int, kind: str):
    return api_services.get_or_create_summary_response(analysis_id, kind)


def get_or_create_node_summary_response(analysis_id: int, node_id: str):
    return api_services.get_or_create_node_summary_response(analysis_id, node_id)


def get_mock_issue_list_response(repo_path: str, **kwargs):
    return api_services.get_mock_issue_list_response(repo_path, **kwargs)


def get_live_issue_list_response(repo_path: str, **kwargs):
    return api_services.get_live_issue_list_response(repo_path, **kwargs)


def get_mock_issue_related_nodes_response(analysis_id: int, issue_number: int, **kwargs):
    return api_services.get_mock_issue_related_nodes_response(analysis_id, issue_number, **kwargs)


def get_issue_map_response(analysis_id: int, issue_number: int, **kwargs):
    return api_services.get_issue_map_response(analysis_id, issue_number, **kwargs)


def get_diff_response(repo_path: str, base_revision: str, head_revision: str | None = None, analysis_profile: str | None = None):
    return api_services.get_diff_response(repo_path, base_revision, head_revision, analysis_profile=analysis_profile)


def get_diff_response_by_analysis_id(head_analysis_id: int, base_analysis_id: int):
    return api_services.get_diff_response_by_analysis_id(head_analysis_id, base_analysis_id)


def create_share_response(repo_path: str, **kwargs):
    return api_services.create_share_response(repo_path, **kwargs)


def get_share_response(share_id: str):
    return api_services.get_share_response(share_id)


def _repo_ingestion_error_response(error: RepoIngestionError) -> Response:
    status_by_code = {
        'invalid_repo_path': 400,
        'unsafe_path': 400,
        'repo_not_found': 404,
        'private_repo': 404,
        'timeout': 504,
        'too_large': 413,
        'revision_not_found': 404,
        'git_error': 502,
    }
    logger.warning('Repo ingestion failed: %s', error.as_dict())
    return Response(
        {
            'error': error.message,
            'code': error.code,
            'detail': error.as_dict(),
        },
        status=status_by_code.get(error.code, 502),
    )


def _github_issue_api_error_response(error: GithubIssueApiError) -> Response:
    logger.warning('GitHub issue API failed: %s', error.as_dict())
    return Response(error.as_dict(), status=error.status_code)


def _issue_map_error_response(error: Exception) -> Response:
    if isinstance(error, api_services.IssueMapResponseError):
        return Response(error.as_dict(), status=error.status_code)
    raise error


def _summary_error_response(error: Exception) -> Response:
    if isinstance(error, api_services.SummaryInputError):
        return Response({'error': str(error), 'code': 'summary_input_error'}, status=400)
    if isinstance(error, api_services.SummaryUnavailable):
        return Response({'error': '요약을 생성할 수 없습니다', 'code': 'summary_unavailable', 'detail': str(error)}, status=503)
    raise error


def _diff_error_response(error: Exception) -> Response:
    if isinstance(error, api_services.GraphDiffInputError):
        return Response({'error': str(error), 'code': 'diff_input_error'}, status=400)
    raise error


def _share_error_response(error: Exception) -> Response:
    if isinstance(error, api_services.ShareInputError):
        return Response({'error': str(error), 'code': 'share_input_error'}, status=400)
    raise error


def _frontend_share_url(request, share_id: str) -> str:
    base_url = getattr(settings, 'FRONTEND_BASE_URL', '').rstrip('/')
    if base_url:
        return f'{base_url}/share/{share_id}'
    return request.build_absolute_uri(f'/share/{share_id}')


def _share_payload_with_links(request, payload: dict[str, Any]) -> dict[str, Any]:
    response_payload = dict(payload)
    share_id = str(response_payload['share_id'])
    repo_name = html.escape(str(response_payload['repo']))
    share_url = request.build_absolute_uri(f'/api/share/{share_id}/')
    embed_url = request.build_absolute_uri(f'/api/embed/{share_id}/')
    readme_svg_url = request.build_absolute_uri(f'/api/share/{share_id}/graph.svg')
    frontend_share_url = _frontend_share_url(request, share_id)
    response_payload['urls'] = {
        'share': share_url,
        'embed': embed_url,
        'readme_svg': readme_svg_url,
        'frontend_share': frontend_share_url,
    }
    response_payload['snippets'] = {
        'markdown_link': f'[{response_payload["repo"]} graph]({frontend_share_url})',
        'markdown_image_link': f'[![{response_payload["repo"]} code graph]({readme_svg_url})]({frontend_share_url})',
        'html_image_link': f'<a href="{html.escape(frontend_share_url)}"><img src="{html.escape(readme_svg_url)}" alt="{repo_name} code graph" /></a>',
        'html_iframe': f'<iframe src="{embed_url}" width="100%" height="640" loading="lazy" title="{repo_name} graph"></iframe>',
        'github_readme_note': 'GitHub README는 iframe을 렌더링하지 않으므로 markdown_image_link를 사용하세요.',
    }
    return response_payload


def _svg_response(svg: str) -> HttpResponse:
    response_obj = HttpResponse(svg, content_type='image/svg+xml; charset=utf-8')
    response_obj['Cache-Control'] = 'public, max-age=300, stale-while-revalidate=3600'
    response_obj['X-Content-Type-Options'] = 'nosniff'
    return response_obj


def _sse_event(event_name: str, data: Any) -> str:
    return f'event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


def _accepts_event_stream(request) -> bool:
    return 'text/event-stream' in str(request.headers.get('Accept', '')).lower()


def _qa_stream_requested(request, validated_data: dict[str, Any]) -> bool:
    return bool(validated_data.get('stream')) or _accepts_event_stream(request)


def _qa_streaming_response(
    repo_path: str,
    analysis: dict[str, Any],
    question: str,
    *,
    selected_node_id: str | None = None,
    selected_file_path: str | None = None,
    max_context_files: int = 4,
    model: str | None = None,
) -> StreamingHttpResponse:
    def stream():
        answer_stream = None
        try:
            answer_stream = stream_answer_question(
                repo_path,
                analysis,
                question,
                selected_node_id=selected_node_id,
                selected_file_path=selected_file_path,
                max_context_files=max_context_files,
                model=model,
            )
            for item in answer_stream:
                if not isinstance(item, dict):
                    continue
                event_name = str(item.get('event') or 'message')
                yield _sse_event(event_name, item.get('data') or {})
        except Exception as error:
            logger.exception('QA streaming failed: %s', error)
            yield _sse_event(
                'error',
                {
                    'error': 'QA 스트리밍 중 오류가 발생했습니다.',
                    'code': 'qa_stream_error',
                    'detail': str(error),
                },
            )
        finally:
            if answer_stream is not None:
                close = getattr(answer_stream, 'close', None)
                if callable(close):
                    close()

    response_obj = StreamingHttpResponse(stream(), content_type='text/event-stream; charset=utf-8')
    response_obj['Cache-Control'] = 'no-cache'
    response_obj['X-Accel-Buffering'] = 'no'
    response_obj['Vary'] = 'Accept'
    return response_obj


def _query_int(request, name: str) -> int | None:
    value = request.GET.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_ANALYSIS_REQUEST_SCHEMA = inline_serializer(
    name='AnalysisRequest',
    fields={
        'repo_url': serializers.CharField(help_text='분석할 public GitHub 레포 URL입니다. 예: https://github.com/psf/requests'),
        'revision': serializers.CharField(required=False, help_text='선택. 특정 commit/ref를 고정할 때 사용합니다. 생략하면 최신 HEAD를 분석합니다.'),
        'analysis_profile': serializers.CharField(required=False, help_text='선택. python-v1 또는 multi-lang-js-ts-v1. 생략하면 서버 기본 profile을 사용합니다.'),
    },
)
_ANALYSIS_RESPONSE_SCHEMA = inline_serializer(
    name='AnalysisResponse',
    fields={
        'analysis_id': serializers.IntegerField(allow_null=True, help_text='이 분석 run의 ID입니다. summary, node-summary, qa에서 재사용합니다.'),
        'repo': serializers.CharField(help_text='정규화된 owner/repo 형식의 레포 이름입니다.'),
        'revision': serializers.CharField(help_text='실제로 분석된 git commit SHA입니다. tree/graph/qa 재조회 시 같은 결과를 고정하는 데 사용합니다.'),
        'analysis_profile': serializers.CharField(help_text='분석 profile ID입니다.'),
        'status': serializers.CharField(help_text='분석 상태입니다. succeeded면 artifact를 사용할 수 있습니다.'),
        'artifact': serializers.JSONField(allow_null=True, help_text='tree, nodes, edges, entrypoints, key_modules 등을 포함한 전체 분석 결과입니다.'),
        'warnings': serializers.JSONField(help_text='분석 중 일부 파일 제외 등 사용자에게 보여줄 수 있는 경고 목록입니다.'),
    },
)
_ANALYSIS_DETAIL_RESPONSE_SCHEMA = inline_serializer(
    name='AnalysisByIdResponse',
    fields={
        'analysis_id': serializers.IntegerField(help_text='조회한 분석 run ID입니다.'),
        'repo': serializers.CharField(help_text='정규화된 owner/repo 형식의 레포 이름입니다.'),
        'revision': serializers.CharField(allow_blank=True, help_text='분석된 git commit SHA입니다. 실패/진행 중이면 비어 있을 수 있습니다.'),
        'analysis_profile': serializers.CharField(help_text='분석 profile ID입니다.'),
        'status': serializers.CharField(help_text='started, succeeded, failed 중 하나입니다.'),
        'artifact': serializers.JSONField(allow_null=True, help_text='성공한 분석의 전체 artifact입니다. 실패/진행 중이면 null입니다.'),
        'warnings': serializers.JSONField(help_text='분석 경고 목록입니다.'),
        'error': serializers.JSONField(required=False, allow_null=True, help_text='실패한 분석 run의 에러 정보입니다.'),
    },
)
_DIFF_RESPONSE_SCHEMA = inline_serializer(
    name='GraphDiffResponse',
    fields={
        'repo': serializers.CharField(help_text='비교 대상 owner/repo입니다.'),
        'base': serializers.JSONField(help_text='기준 분석 결과의 revision/analysis 정보입니다.'),
        'head': serializers.JSONField(help_text='대상 분석 결과의 revision/analysis 정보입니다.'),
        'diff': serializers.JSONField(help_text='추가/삭제/변경된 노드와 엣지 요약입니다.'),
        'warnings': serializers.JSONField(help_text='비교 중 발생한 경고 목록입니다.'),
    },
)
_SHARE_CREATE_SCHEMA = inline_serializer(
    name='ShareCreateRequest',
    fields={
        'repo_url': serializers.CharField(help_text='공유할 public GitHub 레포 URL입니다.'),
        'mode': serializers.ChoiceField(choices=('fixed', 'latest'), required=False, help_text='fixed는 현재 revision 고정, latest는 조회 시 최신 HEAD로 갱신합니다. 기본값 fixed.'),
        'revision': serializers.CharField(required=False, help_text='fixed 공유에서 특정 revision을 지정할 때 사용합니다. latest 모드에서는 사용할 수 없습니다.'),
        'analysis_profile': serializers.CharField(required=False, help_text='선택. python-v1 또는 multi-lang-js-ts-v1. 생략하면 서버 기본 profile을 사용합니다.'),
        'title': serializers.CharField(required=False, allow_blank=True, help_text='공유 화면/SVG에 표시할 제목입니다. 생략하면 repo 이름을 사용합니다.'),
        'expires_at': serializers.DateTimeField(required=False, allow_null=True, help_text='선택. 공유 링크 만료 시간입니다.'),
    },
)
_SHARE_RESPONSE_SCHEMA = inline_serializer(
    name='ShareResponse',
    fields={
        'share_id': serializers.CharField(help_text='공유 링크 식별자입니다. /api/share/{share_id}/ 등에 사용합니다.'),
        'mode': serializers.CharField(help_text='fixed 또는 latest입니다.'),
        'title': serializers.CharField(allow_blank=True, help_text='공유 제목입니다.'),
        'repo': serializers.CharField(help_text='정규화된 owner/repo 형식의 레포 이름입니다.'),
        'repository': serializers.JSONField(help_text='owner, name, full_name 등 레포 메타데이터입니다.'),
        'ref': serializers.CharField(help_text='공유가 가리키는 ref입니다. fixed면 revision, latest면 HEAD입니다.'),
        'revision': serializers.CharField(help_text='현재 응답이 실제로 사용하는 분석 revision입니다.'),
        'analysis_profile': serializers.CharField(help_text='분석 profile ID입니다.'),
        'analysis_id': serializers.IntegerField(help_text='현재 공유 응답에 연결된 분석 run ID입니다.'),
        'graph': serializers.JSONField(help_text='공유 화면에 사용할 공개 graph payload입니다. file_contents는 포함하지 않습니다.'),
        'is_active': serializers.BooleanField(help_text='공유 링크 활성 여부입니다.'),
        'created_at': serializers.CharField(help_text='공유 링크 생성 시각입니다.'),
        'expires_at': serializers.CharField(allow_null=True, help_text='공유 링크 만료 시각입니다. 없으면 null입니다.'),
        'warnings': serializers.JSONField(help_text='분석 경고 목록입니다.'),
        'urls': serializers.JSONField(help_text='share, embed, readme_svg, frontend_share URL 모음입니다.'),
        'snippets': serializers.JSONField(help_text='README/HTML에 바로 붙일 수 있는 markdown/html snippet 모음입니다.'),
    },
)
_README_GRAPH_SVG_DESCRIPTION = f'''
GitHub 레포 URL 하나로 README용 코드 구조 SVG를 생성합니다.

프런트엔드 연결:
```ts
const src = `https://gitstarter.kro.kr/api/readme-graph.svg?${{new URLSearchParams({{ url: repoUrl }})}}`;
```
`repoUrl`은 사용자가 입력한 원본 GitHub URL입니다. `URLSearchParams`가 인코딩을 처리합니다.

Swagger에서 테스트할 때:
- `url`에는 원본 URL을 그대로 입력하세요. 예: `https://github.com/psf/requests`
- 응답은 `image/svg+xml`입니다. 새 탭에서 열거나 `<img src>`에 넣어 확인하세요.

README 또는 `<img src>`에 붙일 때:
```md
![Codebase structure](https://gitstarter.kro.kr/api/readme-graph.svg?url=https%3A%2F%2Fgithub.com%2Fpsf%2Frequests)
```

알아둘 점:
- 공식 파라미터는 `url`입니다. `share_id`는 필요하지 않습니다.
- 첫 요청은 레포 분석 때문에 느릴 수 있고, 이후에는 분석 캐시를 재사용합니다.
- public GitHub 레포만 지원합니다.
- 주요 실패: 400 잘못된 URL, 404 private/not found, 413 너무 큰 레포, 504 timeout.
'''
_ANALYSIS_GET_DESCRIPTION = '''
레포를 분석하고 프런트엔드가 재사용할 기본 분석 결과를 반환합니다.

프런트엔드 권장 흐름:
1. 사용자가 GitHub URL을 입력하면 먼저 이 API를 호출합니다.
2. 응답의 `analysis_id`는 `/api/summary/`, `/api/node-summary/`, `/api/qa/`에서 재사용합니다.
3. 응답의 `revision`은 실제 분석된 git commit SHA입니다. 같은 커밋을 다시 조회할 때
   `/api/tree/`, `/api/graph/`, `/api/qa/`의 `revision`으로 넘기면 같은 분석 결과를 고정해서 볼 수 있습니다.

캐시 의미:
- 서버는 `repo + revision(commit SHA)` 기준으로 분석 결과를 저장합니다.
- 같은 repo/revision 요청은 다시 clone/parse하지 않고 저장된 분석 artifact를 반환합니다.
- `revision`을 생략하면 현재 HEAD를 분석합니다. HEAD가 바뀌면 새 분석 결과가 만들어질 수 있습니다.

주요 실패: 400 잘못된 URL/revision, 404 not found/private/revision 없음, 413 너무 큰 레포, 504 timeout.
'''
_ANALYSIS_POST_DESCRIPTION = '''
GET `/api/analysis/`와 같은 분석을 POST body로 요청합니다.

폼/query string보다 JSON body가 편한 클라이언트에서 사용하세요.
응답의 `analysis_id`와 `revision` 재사용 방식은 GET과 같습니다.
'''
_ANALYSIS_DETAIL_DESCRIPTION = '''
이미 생성된 분석 run을 `analysis_id`로 다시 조회합니다.

프런트엔드가 이전 응답에서 받은 `analysis_id`를 보관하고 있다면 repo URL 없이도 같은 분석 artifact를 불러올 수 있습니다.
'''
_TREE_DESCRIPTION = '''
레포 파일/심볼 트리를 반환합니다.

초기 화면에서 사이드바 파일 트리나 구조 트리를 그릴 때 사용합니다.
`revision`을 생략하면 최신 HEAD 기준으로 분석하고, 특정 커밋을 고정하려면 `/api/analysis/` 응답의 `revision`을 넣으세요.
분석 캐시는 `/api/analysis/`, `/api/graph/`, `/api/qa/`와 공유됩니다.
'''
_GRAPH_DESCRIPTION = '''
레포의 코드 그래프 노드/엣지를 반환합니다.

캔버스/그래프 뷰를 그릴 때 사용합니다. 응답의 `nodes[].id`는 `/api/node-summary/`의 `node_id`,
`/api/qa/`의 `selected_node_id`로 재사용할 수 있습니다.
`revision`은 `/api/analysis/` 응답 값을 넣으면 같은 분석 결과를 고정합니다.
'''
_SHARE_CREATE_DESCRIPTION = '''
현재 분석 결과를 공유 가능한 링크로 만듭니다.

프런트엔드에서 "공유하기" 버튼을 만들 때 사용합니다.
응답의 `urls.readme_svg`와 `snippets.markdown_image_link`는 README에 바로 붙일 수 있습니다.
`mode=fixed`는 현재 revision에 고정되고, `mode=latest`는 조회 시 최신 HEAD 분석으로 갱신됩니다.
'''
_SHARE_DETAIL_DESCRIPTION = '''
share_id로 공개 공유 데이터를 다시 조회합니다.

공유 상세 페이지를 열 때 사용합니다. 응답에는 graph 데이터와 embed/readme 링크가 포함됩니다.
'''
_SHARE_GRAPH_SVG_DESCRIPTION = '''
이미 생성된 share_id를 기준으로 README-safe SVG 이미지를 반환합니다.

새 연결에는 `/api/readme-graph.svg?url=...`가 더 단순합니다.
이 엔드포인트는 `/api/share/` 응답의 `urls.readme_svg`를 그대로 렌더링할 때 사용합니다.
'''
_EMBED_DESCRIPTION = '''
share_id 기반 HTML embed 페이지입니다.

일반 웹페이지 iframe에는 사용할 수 있지만, GitHub README는 iframe을 렌더링하지 않습니다.
GitHub README에는 `snippets.markdown_image_link` 또는 `/api/readme-graph.svg`를 사용하세요.
'''
_DIFF_BY_REVISION_DESCRIPTION = '''
같은 레포의 두 revision 분석 결과를 비교합니다.

`base`는 필수이고 `head`를 생략하면 최신 HEAD와 비교합니다.
프런트엔드에서 변경 전/후 그래프 차이를 보여줄 때 사용합니다.
'''
_DIFF_BY_ID_DESCRIPTION = '''
두 분석 run ID를 비교합니다.

현재 path의 `analysis_id`가 head이고, query의 `base`가 비교 기준 분석 run ID입니다.
'''
_SUMMARY_DESCRIPTION = '''
분석 결과 전체에 대한 LLM 요약을 생성하거나 캐시된 요약을 반환합니다.

반드시 `/api/analysis/`에서 받은 `analysis_id`를 먼저 넣어야 합니다.
`kind=repo_overview`는 짧은 전체 설명, `kind=onboarding_guide`는 온보딩 관점 설명입니다.
'''
_NODE_SUMMARY_DESCRIPTION = '''
그래프의 특정 노드 하나를 요약합니다.

`analysis_id`는 `/api/analysis/` 응답에서, `node_id`는 `/api/graph/` 응답의 `nodes[].id`에서 가져오세요.
'''
_QA_DESCRIPTION = '''
레포 분석 결과를 바탕으로 질문에 답합니다.

권장 사용법:
- 이미 `/api/analysis/`를 호출했다면 `analysis_id`를 보내세요. repo URL 재분석을 피할 수 있습니다.
- 특정 그래프 노드를 선택한 상태라면 `selected_node_id`를 함께 보내면 답변 범위가 좁아집니다.
- `repo_url`만 보내도 동작하지만, 프런트엔드에서는 `analysis_id` 재사용이 더 안정적입니다.
- 기본은 기존 JSON 응답입니다. `Accept: text/event-stream` 또는 `stream=true`를 보내면 같은 endpoint에서 SSE를 반환합니다.
- SSE 완료 신호는 항상 `event: final`입니다. `[DONE]` sentinel이나 별도 end event는 없습니다.
- 답변 본문은 `event: token` / `data: {"text": "..."}` 형식으로 전송되며, 최종 전체 답변은 `final.data.answer`에도 포함됩니다.
- Pi harness QA 경로에서는 답변 token 전송 전에 진행 이벤트가 추가로 올 수 있습니다: `harness_start`, `harness_usage`, `harness_tool_call`, `harness_tool_result`.
- 실패나 timeout은 스트림 종료 전에 `event: error` / `data: {"error": "...", "code": "..."}` 형식으로 전송됩니다.
- LLM 호출은 OpenCode Zen만 사용합니다. `model`을 보내면 해당 Zen 모델을 요청하고, `config.yaml`의 `opencode.allowed_models`가 설정되어 있으면 그 목록 안에서만 허용됩니다.
'''
_REPO_FILES_DESCRIPTION = '''
GitHub 레포의 파일 경로 목록만 빠르게 반환합니다.

코드 그래프/요약이 필요 없는 단순 파일 목록 UI에 사용합니다.
구조 분석이 필요한 화면에서는 `/api/tree/` 또는 `/api/analysis/`를 사용하세요.
'''
_ISSUES_LIST_DESCRIPTION = '''
GitHub open issue 목록 API입니다.

프런트엔드 권장 흐름:
1. 사용자가 GitHub URL을 입력하면 `/api/analysis/`로 먼저 분석을 만들고 `analysis_id`를 보관합니다.
2. 같은 URL로 이 API를 호출해 open issue 목록을 표시합니다.
3. 사용자가 issue를 선택하면 응답의 `number`를 `/api/issues/related-nodes/`의 `issue_number`로 보냅니다.

현재 동작:
- 기본값은 live GitHub REST API 조회입니다.
- `mock=true`이면 안정적인 프런트엔드/테스트용 mock issue 목록을 반환합니다.
- `source=github`, `mock=false`이면 live GitHub 응답입니다.
- `key`, `number`, `title`, `labels`, `assignees`, `body_excerpt`, `body_truncated`, `locked` 필드는 유지하는 계약입니다.
- live 응답에는 `repository`, `rate_limit`, `warnings`가 추가될 수 있습니다.
- GitHub Pull Request는 issue 목록에서 제외하므로 `is_pull_request=false`만 반환합니다.

프런트엔드 예시:
```ts
const params = new URLSearchParams({ url: repoUrl, page: "1", per_page: "30" });
const res = await fetch(`${API_BASE}/api/issues/?${params}`);
```

주요 실패: 400 잘못된 GitHub URL 또는 pagination 값, 404 레포 없음/private, 429 GitHub rate limit, 502 upstream 오류.
'''
_ISSUE_RELATED_NODES_DESCRIPTION = '''
선택한 GitHub issue를 해결하는 데 관련 있어 보이는 contributor graph와 code context를 반환합니다.

프런트엔드 권장 흐름:
1. `/api/analysis/` 응답의 `analysis_id`를 보관합니다.
2. `/api/issues/`에서 받은 issue의 `number`를 사용자가 선택한 issue로 저장합니다.
3. 이 API에 `analysis_id`, `issue_number`, 선택적으로 `max_nodes`, `include_comments`, `max_context_files`를 POST합니다.
4. 응답의 `start_here`는 초보자용 "먼저 열 파일/노드" 패널에 바로 쓰고, `selected_node_ids`와 `focus_graph.highlight_node_ids`는 graph highlight에 사용합니다.
5. `candidates[].node_id`는 `/api/node-summary/`의 `node_id`로 재사용할 수 있습니다.

현재 동작:
- 기본값은 live GitHub issue detail/comment를 읽고 deterministic evidence/ranking으로 seed 후보를 만듭니다.
- `ISSUE_HARNESS_ENABLED=true` 또는 `ISSUE_MAP_LLM_ENABLED=true`이면 Pi 기반 bounded issue harness가 artifact 도구로 repo graph/code를 조사합니다.
- harness는 built-in filesystem/shell/network tools 없이 backend가 넘긴 bounded graph/file_contents만 읽고, 실패하면 deterministic ranking으로 폴백합니다.
- `mock=true`이면 기존 프런트엔드 선작업용 mock 응답을 반환합니다.
- 반환되는 `node_id`는 실제 `/api/graph/`의 `nodes[].id`와 연결됩니다.
- 새 필드는 additive입니다: `overview_graph`, `focus_graph`, `hypotheses`, `investigation_path`, `start_here`, `next_steps`, `avoid`, `guidance_summary`, `code_context`, `confidence`, `harness`.

요청 예시:
```json
{
  "analysis_id": 123,
  "issue_number": 42,
  "max_nodes": 8,
  "include_comments": true,
  "max_context_files": 4
}
```

주요 실패: 400 잘못된 body, 404 분석 결과 또는 mock issue 번호 없음.
'''
_ISSUES_LIST_RESPONSE_EXAMPLE = {
    'repo': 'owner/repo',
    'provider': 'github',
    'source': 'mock',
    'mock': True,
    'state': 'open',
    'page': 1,
    'per_page': 30,
    'has_next_page': False,
    'next_page': None,
    'issues': [
        {
            'key': 'github:owner/repo#42',
            'number': 42,
            'title': 'Repository analysis fails on large Python projects',
            'state': 'open',
            'html_url': 'https://github.com/owner/repo/issues/42',
            'author': {
                'login': 'octocat',
                'avatar_url': 'https://github.com/octocat.png',
                'html_url': 'https://github.com/octocat',
            },
            'labels': [
                {'name': 'bug', 'color': 'd73a4a', 'description': 'Something is not working'},
                {'name': 'analysis', 'color': '1d76db', 'description': 'Repository analysis flow'},
            ],
            'assignees': [
                {
                    'login': 'hubot',
                    'avatar_url': 'https://github.com/hubot.png',
                    'html_url': 'https://github.com/hubot',
                }
            ],
            'comments_count': 3,
            'created_at': '2026-05-20T10:00:00Z',
            'updated_at': '2026-05-23T12:30:00Z',
            'body_excerpt': 'Repository analysis fails when the project has many Python files or the parser exceeds configured limits.',
            'body_truncated': True,
            'locked': False,
            'is_pull_request': False,
        },
        {
            'key': 'github:owner/repo#156',
            'number': 156,
            'title': 'Empty state should work when an issue has no labels',
            'state': 'open',
            'html_url': 'https://github.com/owner/repo/issues/156',
            'author': {
                'login': 'minimal-reporter',
                'avatar_url': 'https://github.com/minimal-reporter.png',
                'html_url': 'https://github.com/minimal-reporter',
            },
            'labels': [],
            'assignees': [],
            'comments_count': 0,
            'created_at': '2026-05-14T07:30:00Z',
            'updated_at': '2026-05-14T07:30:00Z',
            'body_excerpt': '',
            'body_truncated': False,
            'locked': False,
            'is_pull_request': False,
        },
        {
            'key': 'github:owner/repo#181',
            'number': 181,
            'title': 'Deleted author issue should not crash rendering',
            'state': 'open',
            'html_url': 'https://github.com/owner/repo/issues/181',
            'author': None,
            'labels': [
                {'name': 'edge-case', 'color': 'ededed', 'description': 'Mock data for nullable GitHub fields'},
            ],
            'assignees': [],
            'comments_count': 1,
            'created_at': '2026-05-10T03:05:00Z',
            'updated_at': '2026-05-15T19:25:00Z',
            'body_excerpt': 'GitHub can return nullable user-like data in some historical or deleted-user cases.',
            'body_truncated': False,
            'locked': False,
            'is_pull_request': False,
        },
        {
            'key': 'github:owner/repo#209',
            'number': 209,
            'title': 'Locked conversation still needs related node suggestions',
            'state': 'open',
            'html_url': 'https://github.com/owner/repo/issues/209',
            'author': {
                'login': 'security-reviewer',
                'avatar_url': 'https://github.com/security-reviewer.png',
                'html_url': 'https://github.com/security-reviewer',
            },
            'labels': [
                {'name': 'security', 'color': 'ee0701', 'description': 'Security-sensitive behavior'},
                {'name': 'backend', 'color': '0052cc', 'description': 'Backend implementation'},
            ],
            'assignees': [
                {
                    'login': 'backend-owner',
                    'avatar_url': 'https://github.com/backend-owner.png',
                    'html_url': 'https://github.com/backend-owner',
                }
            ],
            'comments_count': 8,
            'created_at': '2026-05-08T14:00:00Z',
            'updated_at': '2026-05-24T06:55:00Z',
            'body_excerpt': 'Locked issues should remain selectable, but the frontend may show a lock badge while still requesting related nodes.',
            'body_truncated': False,
            'locked': True,
            'is_pull_request': False,
        },
    ],
}
_ISSUE_RELATED_NODES_REQUEST_EXAMPLE = {
    'analysis_id': 123,
    'issue_number': 42,
    'max_nodes': 8,
    'include_comments': True,
    'max_context_files': 4,
}
_ISSUE_RELATED_NODES_RESPONSE_EXAMPLE = {
    'analysis_id': 123,
    'repo': 'owner/repo',
    'revision': 'abc123',
    'provider': 'github',
    'source': 'mock',
    'mock': True,
    'cached': False,
    'cache_key': 'issue_map:42:v2:comments_false:ctx_4:nodes_8:harness_off',
    'cache_version': 'v2',
    'issue': {
        'key': 'github:owner/repo#42',
        'number': 42,
        'title': 'Repository analysis fails on large Python projects',
        'state': 'open',
        'html_url': 'https://github.com/owner/repo/issues/42',
        'labels': [
            {'name': 'bug', 'color': 'd73a4a', 'description': 'Something is not working'},
            {'name': 'analysis', 'color': '1d76db', 'description': 'Repository analysis flow'},
        ],
        'comments_count': 3,
        'updated_at': '2026-05-23T12:30:00Z',
        'body_excerpt': 'Repository analysis fails when the project has many Python files or the parser exceeds configured limits.',
    },
    'selected_node_ids': ['api/views.py::analysis', 'api/services.py::get_repo_analysis'],
    'candidates': [
        {
            'rank': 1,
            'score': 1.0,
            'node_id': 'api/views.py::analysis',
            'node': {
                'id': 'api/views.py::analysis',
                'kind': 'function',
                'label': 'analysis',
                'path': 'api/views.py',
                'start_line': 180,
                'end_line': 220,
                'metadata': {},
            },
            'reason': 'Mock candidate based on issue title/body tokens and graph node metadata. 실제 구현에서는 GitHub issue 본문/comment, deterministic seed ranking, bounded issue harness를 사용합니다.',
            'evidence': [
                {'type': 'mock', 'message': '프런트엔드 graph highlight 연동을 위한 임시 추천입니다.'},
                {'type': 'graph_metadata', 'message': 'Graph node path: api/views.py'},
                {'type': 'node_kind_priority', 'message': 'function node를 file node보다 우선 추천했습니다.'},
            ],
        }
    ],
    'limits': {'max_nodes': 8},
    'warnings': [],
    'overview_graph': {'nodes': [], 'edges': [], 'node_ids': [], 'limits': {'node_limit': 80}},
    'focus_graph': {'nodes': [], 'edges': [], 'node_ids': [], 'highlight_node_ids': []},
    'hypotheses': [],
    'investigation_path': [],
    'start_here': {
        'node_id': 'api/views.py::analysis',
        'path': 'api/views.py',
        'start_line': 180,
        'end_line': 220,
        'label': 'analysis',
        'kind': 'function',
        'why': 'Issue evidence points to this API entrypoint.',
        'confidence': 1.0,
    },
    'next_steps': [
        {
            'node_id': 'api/services.py::get_repo_analysis',
            'path': 'api/services.py',
            'start_line': 620,
            'end_line': 710,
            'label': 'get_repo_analysis',
            'kind': 'function',
            'action': 'inspect',
            'why': 'This service path performs repository analysis work.',
        }
    ],
    'avoid': [],
    'guidance_summary': {
        'mode': 'deterministic',
        'message': 'Start with api/views.py::analysis, then inspect api/services.py::get_repo_analysis.',
        'warning_codes': [],
    },
    'code_context': {'files': [], 'file_count': 0, 'max_context_files': 4, 'truncated': False},
    'confidence': {'level': 'medium', 'score': 0.7, 'reasons': []},
}


@extend_schema(
    methods=['GET'],
    operation_id='analysis_retrieve_by_url',
    description=_ANALYSIS_GET_DESCRIPTION,
    parameters=[
        OpenApiParameter(name='url', description='필수. 분석할 public GitHub 레포 URL. 예: https://github.com/psf/requests', required=True, type=str),
        OpenApiParameter(name='revision', description='선택. 특정 commit/ref를 고정할 때 입력합니다. 보통은 생략하고, 재조회 시 응답의 revision을 재사용합니다.', required=False, type=str),
        OpenApiParameter(name='analysis_profile', description='선택. python-v1 또는 multi-lang-js-ts-v1. 생략하면 서버 기본 profile을 사용합니다.', required=False, type=str),
    ],
    responses=_ANALYSIS_RESPONSE_SCHEMA,
)
@extend_schema(
    methods=['POST'],
    operation_id='analysis_create',
    description=_ANALYSIS_POST_DESCRIPTION,
    request=_ANALYSIS_REQUEST_SCHEMA,
    responses=_ANALYSIS_RESPONSE_SCHEMA,
)
@api_view(['GET', 'POST'])
def analysis(request):
    if request.method == 'GET':
        request_data = {'repo_url': request.GET.get('url')}
        if request.GET.get('revision') is not None:
            request_data['revision'] = request.GET.get('revision')
        if request.GET.get('analysis_profile') is not None:
            request_data['analysis_profile'] = request.GET.get('analysis_profile')
        serializer = AnalysisRequestSerializer(data=request_data)
    else:
        serializer = AnalysisRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, str], serializer.validated_data)
    repo_path = validated_data['repo_url']
    revision = validated_data.get('revision')
    analysis_profile = validated_data.get('analysis_profile')
    try:
        response = get_analysis_response(repo_path, revision, analysis_profile=analysis_profile)
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)

    if response is None:
        return Response({'error': '분석 결과를 찾을 수 없습니다'}, status=404)
    return Response(response)


@extend_schema(
    operation_id='analysis_retrieve_by_id',
    description=_ANALYSIS_DETAIL_DESCRIPTION,
    responses=_ANALYSIS_DETAIL_RESPONSE_SCHEMA,
)
@api_view(['GET'])
def analysis_detail(request, analysis_id: int):
    response = get_analysis_response_by_id(analysis_id)
    if response is None:
        return Response({'error': '분석 결과를 찾을 수 없습니다'}, status=404)
    return Response(response)


@extend_schema(
    operation_id='analysis_diff_by_id',
    description=_DIFF_BY_ID_DESCRIPTION,
    parameters=[
        OpenApiParameter(name='base', description='필수. 비교 기준이 되는 analysis_id. path의 analysis_id는 비교 대상 head입니다.', required=True, type=int),
    ],
    responses=_DIFF_RESPONSE_SCHEMA,
)
@api_view(['GET'])
def analysis_diff(request, analysis_id: int):
    serializer = AnalysisDiffRequestSerializer(data=request.GET)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, Any], serializer.validated_data)
    try:
        response = get_diff_response_by_analysis_id(analysis_id, int(validated_data['base']))
    except Exception as error:
        return _diff_error_response(error)
    if response is None:
        return Response({'error': '비교할 분석 결과를 찾을 수 없습니다'}, status=404)
    return Response(response)


@extend_schema(
    operation_id='analysis_diff_by_revision',
    description=_DIFF_BY_REVISION_DESCRIPTION,
    parameters=[
        OpenApiParameter(name='url', description='필수. 비교할 public GitHub 레포 URL.', required=True, type=str),
        OpenApiParameter(name='base', description='필수. 기준 revision 또는 commit SHA.', required=True, type=str),
        OpenApiParameter(name='head', description='선택. 대상 revision. 생략하면 최신 HEAD와 비교합니다.', required=False, type=str),
        OpenApiParameter(name='analysis_profile', description='선택. python-v1 또는 multi-lang-js-ts-v1. 생략하면 서버 기본 profile을 사용합니다.', required=False, type=str),
    ],
    responses=_DIFF_RESPONSE_SCHEMA,
)
@api_view(['GET'])
def graph_diff(request):
    request_data = {
        'repo_url': request.GET.get('url'),
        'base': request.GET.get('base'),
    }
    if request.GET.get('head') is not None:
        request_data['head'] = request.GET.get('head')
    if request.GET.get('analysis_profile') is not None:
        request_data['analysis_profile'] = request.GET.get('analysis_profile')
    serializer = DiffRequestSerializer(data=request_data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, str], serializer.validated_data)
    try:
        response = get_diff_response(
            validated_data['repo_url'],
            validated_data['base'],
            validated_data.get('head'),
            analysis_profile=validated_data.get('analysis_profile'),
        )
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)
    except Exception as error:
        return _diff_error_response(error)
    if response is None:
        return Response({'error': '비교할 분석 결과를 찾을 수 없습니다'}, status=404)
    return Response(response)


@extend_schema(
    methods=['POST'],
    operation_id='share_create',
    description=_SHARE_CREATE_DESCRIPTION,
    request=_SHARE_CREATE_SCHEMA,
    responses={201: _SHARE_RESPONSE_SCHEMA},
)
@api_view(['POST'])
@throttle_classes([ShareCreateRateThrottle])
def share(request):
    serializer = ShareCreateSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, Any], serializer.validated_data)
    try:
        response = create_share_response(
            str(validated_data['repo_url']),
            mode=str(validated_data.get('mode', 'fixed')),
            revision=validated_data.get('revision'),
            analysis_profile=str(validated_data.get('analysis_profile') or ''),
            title=str(validated_data.get('title') or ''),
            expires_at=validated_data.get('expires_at'),
        )
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)
    except Exception as error:
        return _share_error_response(error)
    if response is None:
        return Response({'error': '공유할 분석 결과를 만들 수 없습니다'}, status=404)
    return Response(_share_payload_with_links(request, response), status=201)


@extend_schema(
    operation_id='readme_graph_svg_by_repo_url',
    description=_README_GRAPH_SVG_DESCRIPTION,
    parameters=[
        OpenApiParameter(
            name='url',
            description='필수 공식 파라미터. Swagger에는 원본 GitHub URL을 그대로 입력하세요. FE 코드에서는 URLSearchParams 또는 encodeURIComponent로 인코딩하세요.',
            required=True,
            type=str,
        ),
        OpenApiParameter(
            name='revision',
            description='선택. 특정 revision/ref를 고정해서 분석할 때만 입력합니다. 생략하면 최신 HEAD를 사용합니다.',
            required=False,
            type=str,
        ),
        OpenApiParameter(name='width', description=f'선택. SVG 너비 px ({MIN_SVG_WIDTH}-{MAX_SVG_WIDTH}). 기본값 권장.', required=False, type=int),
        OpenApiParameter(name='height', description=f'선택. SVG 높이 px ({MIN_SVG_HEIGHT}-{MAX_SVG_HEIGHT}). 기본값 권장.', required=False, type=int),
        OpenApiParameter(name='limit', description=f'선택. 모듈 선별 한도 ({MIN_NODE_LIMIT}-{MAX_NODE_LIMIT}). 고급 옵션이며 기본값 권장.', required=False, type=int),
        OpenApiParameter(name='theme', description='선택. light 또는 dark.', required=False, type=str),
        OpenApiParameter(name='title', description='선택. SVG 상단 제목을 직접 지정할 때 사용합니다.', required=False, type=str),
        OpenApiParameter(name='analysis_profile', description='선택. python-v1 또는 multi-lang-js-ts-v1. 생략하면 서버 기본 profile을 사용합니다.', required=False, type=str),
    ],
    responses={(200, 'image/svg+xml'): OpenApiTypes.STR},
)
@api_view(['GET', 'HEAD'])
@renderer_classes([JSONRenderer, SvgRenderer])
def readme_graph_svg(request):
    request_data = {'repo_url': request.GET.get('url') or request.GET.get('repo_url')}
    if request.GET.get('revision') is not None:
        request_data['revision'] = request.GET.get('revision')
    if request.GET.get('analysis_profile') is not None:
        request_data['analysis_profile'] = request.GET.get('analysis_profile')
    serializer = AnalysisRequestSerializer(data=request_data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, str], serializer.validated_data)
    repo_path = validated_data['repo_url']
    revision = validated_data.get('revision')
    analysis_profile = validated_data.get('analysis_profile')
    try:
        analysis = get_repo_analysis(repo_path, revision, analysis_profile=analysis_profile)
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)
    if analysis is None:
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    analysis_run = get_analysis_run_by_revision(repo_path, str(analysis['revision']), analysis_profile=analysis_profile)
    graph = build_graph_response(analysis, analysis_run)
    options = normalize_svg_options(
        width=_query_int(request, 'width') or DEFAULT_SVG_WIDTH,
        height=_query_int(request, 'height') or DEFAULT_SVG_HEIGHT,
        node_limit=_query_int(request, 'limit') or DEFAULT_NODE_LIMIT,
        theme=request.GET.get('theme'),
    )
    svg = render_share_graph_svg(
        {
            'mode': 'repo URL input',
            'title': str(request.GET.get('title') or graph['repo'])[:255],
            'repo': graph['repo'],
            'revision': graph['revision'],
            'analysis_id': graph.get('analysis_id'),
            'graph': graph,
            'warnings': graph.get('warnings', []),
        },
        width=cast(int, options['width']),
        height=cast(int, options['height']),
        node_limit=cast(int, options['node_limit']),
        theme_name=cast(str, options['theme']),
    )
    return _svg_response(svg)


@extend_schema(
    operation_id='share_retrieve',
    description=_SHARE_DETAIL_DESCRIPTION,
    responses=_SHARE_RESPONSE_SCHEMA,
)
@api_view(['GET'])
def share_detail(request, share_id: str):
    if not is_safe_share_id(share_id):
        return Response({'error': 'share를 찾을 수 없습니다'}, status=404)
    try:
        response = get_share_response(share_id)
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)
    except Exception as error:
        return _share_error_response(error)
    if response is None:
        return Response({'error': 'share를 찾을 수 없습니다'}, status=404)
    return Response(_share_payload_with_links(request, response))


@extend_schema(
    operation_id='share_graph_svg',
    description=_SHARE_GRAPH_SVG_DESCRIPTION,
    parameters=[
        OpenApiParameter(name='width', description=f'선택. SVG 너비 px ({MIN_SVG_WIDTH}-{MAX_SVG_WIDTH}).', required=False, type=int),
        OpenApiParameter(name='height', description=f'선택. SVG 높이 px ({MIN_SVG_HEIGHT}-{MAX_SVG_HEIGHT}).', required=False, type=int),
        OpenApiParameter(name='limit', description=f'선택. 모듈 선별 한도 ({MIN_NODE_LIMIT}-{MAX_NODE_LIMIT}). 기본값 권장.', required=False, type=int),
        OpenApiParameter(name='theme', description='선택. light 또는 dark.', required=False, type=str),
    ],
    responses={(200, 'image/svg+xml'): OpenApiTypes.STR},
)
@api_view(['GET', 'HEAD'])
@renderer_classes([JSONRenderer, SvgRenderer])
def share_graph_svg(request, share_id: str):
    if not is_safe_share_id(share_id):
        return Response({'error': 'share를 찾을 수 없습니다'}, status=404)
    try:
        response = get_share_response(share_id)
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)
    except Exception as error:
        return _share_error_response(error)
    if response is None:
        return Response({'error': 'share를 찾을 수 없습니다'}, status=404)

    options = normalize_svg_options(
        width=_query_int(request, 'width') or DEFAULT_SVG_WIDTH,
        height=_query_int(request, 'height') or DEFAULT_SVG_HEIGHT,
        node_limit=_query_int(request, 'limit') or DEFAULT_NODE_LIMIT,
        theme=request.GET.get('theme'),
    )
    svg = render_share_graph_svg(
        response,
        width=cast(int, options['width']),
        height=cast(int, options['height']),
        node_limit=cast(int, options['node_limit']),
        theme_name=cast(str, options['theme']),
    )
    return _svg_response(svg)


@extend_schema(
    operation_id='share_embed',
    description=_EMBED_DESCRIPTION,
    responses={200: OpenApiTypes.STR},
)
@xframe_options_exempt
@api_view(['GET'])
def embed(request, share_id: str):
    if not is_safe_share_id(share_id):
        return Response({'error': 'share를 찾을 수 없습니다'}, status=404)
    try:
        response = get_share_response(share_id)
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)
    except Exception as error:
        return _share_error_response(error)
    if response is None:
        return Response({'error': 'share를 찾을 수 없습니다'}, status=404)

    payload = _share_payload_with_links(request, response)
    graph_json = json.dumps(payload['graph'], ensure_ascii=False)
    escaped_title = html.escape(str(payload.get('title') or payload['repo']))
    escaped_repo = html.escape(str(payload['repo']))
    body = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    body {{ margin: 0; font-family: sans-serif; color: #172026; background: #f7f4ed; }}
    main {{ padding: 16px; }}
    h1 {{ margin: 0 0 8px; font-size: 18px; }}
    p {{ margin: 0 0 12px; }}
    pre {{ overflow: auto; padding: 12px; background: #ffffff; border: 1px solid #ddd6c8; border-radius: 8px; }}
  </style>
</head>
<body>
  <main>
    <h1>{escaped_repo}</h1>
    <p>Revision: {html.escape(str(payload['revision']))}</p>
    <pre id="graph-data">{html.escape(graph_json)}</pre>
  </main>
</body>
</html>
"""
    response_obj = HttpResponse(body, content_type='text/html; charset=utf-8')
    response_obj['Cache-Control'] = 'no-store'
    return response_obj


@extend_schema(
    description=_REPO_FILES_DESCRIPTION,
    parameters=[
        OpenApiParameter(name='url', description='필수. 파일 목록을 조회할 public GitHub 레포 URL.', required=True, type=str),
    ],
    responses=inline_serializer(
        name='RepoFilesResponse',
        fields={
            'repo': serializers.CharField(help_text='정규화된 owner/repo 형식의 레포 이름입니다.'),
            'files': serializers.ListField(child=serializers.CharField(), help_text='레포 안의 파일 경로 목록입니다.'),
        },
    ),
)
@api_view(['GET'])
def get_repo_file(request):
    serializer = RepoUrlSerializer(data={'repo_url': request.GET.get('url')})
    if not serializer.is_valid():
        logger.warning(f"잘못된 URL 요청: {request.GET.get('url')}")
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, str], serializer.validated_data)
    repo_path = validated_data['repo_url']
    logger.info(f"파일트리 요청: {repo_path}")

    try:
        files = get_file_tree_or_raise(repo_path)
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)

    if not files:
        logger.error(f"레포 찾기 실패: {repo_path}")
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    logger.info(f"파일트리 반환 완료: {repo_path}")
    return Response({'repo': repo_path, 'files': files})


@extend_schema(
    description=_TREE_DESCRIPTION,
    parameters=[
        OpenApiParameter(name='url', description='필수. 트리를 조회할 public GitHub 레포 URL.', required=True, type=str),
        OpenApiParameter(name='revision', description='선택. 특정 분석 결과를 고정할 때 `/api/analysis/` 응답의 revision을 넣습니다.', required=False, type=str),
        OpenApiParameter(name='analysis_profile', description='선택. python-v1 또는 multi-lang-js-ts-v1. 생략하면 서버 기본 profile을 사용합니다.', required=False, type=str),
    ],
    responses=inline_serializer(
        name='RepoTreeResponse',
        fields={
            'analysis_id': serializers.IntegerField(allow_null=True, help_text='이 tree가 나온 분석 run ID입니다. summary/qa에 재사용할 수 있습니다.'),
            'repo': serializers.CharField(help_text='정규화된 owner/repo 형식의 레포 이름입니다.'),
            'revision': serializers.CharField(help_text='실제로 분석된 git commit SHA입니다. 같은 tree를 다시 볼 때 query revision으로 재사용합니다.'),
            'analysis_profile': serializers.CharField(help_text='분석 profile ID입니다.'),
            'tree': serializers.JSONField(help_text='프런트엔드 파일/심볼 트리 UI에 사용할 계층 구조입니다.'),
            'warnings': serializers.JSONField(help_text='분석 경고 목록입니다.'),
        },
    ),
)
@api_view(['GET'])
def get_repo_tree(request):
    request_data = {'repo_url': request.GET.get('url')}
    if request.GET.get('revision') is not None:
        request_data['revision'] = request.GET.get('revision')
    if request.GET.get('analysis_profile') is not None:
        request_data['analysis_profile'] = request.GET.get('analysis_profile')
    serializer = AnalysisRequestSerializer(data=request_data)
    if not serializer.is_valid():
        logger.warning(f"잘못된 URL 요청: {request.GET.get('url')}")
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, str], serializer.validated_data)
    repo_path = validated_data['repo_url']
    revision = validated_data.get('revision')
    analysis_profile = validated_data.get('analysis_profile')
    logger.info(f"트리 요청: {repo_path}")

    try:
        analysis = get_repo_analysis(repo_path, revision, analysis_profile=analysis_profile)
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)

    if analysis is None:
        logger.error(f"레포 찾기 실패: {repo_path}")
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    logger.info(f"트리 반환 완료: {repo_path}")

    analysis_run = get_analysis_run_by_revision(repo_path, str(analysis['revision']), analysis_profile=analysis_profile)
    return Response(build_tree_response(analysis, analysis_run))


@extend_schema(
    description=_GRAPH_DESCRIPTION,
    parameters=[
        OpenApiParameter(name='url', description='필수. 그래프를 조회할 public GitHub 레포 URL.', required=True, type=str),
        OpenApiParameter(name='revision', description='선택. 특정 분석 결과를 고정할 때 `/api/analysis/` 응답의 revision을 넣습니다.', required=False, type=str),
        OpenApiParameter(name='analysis_profile', description='선택. python-v1 또는 multi-lang-js-ts-v1. 생략하면 서버 기본 profile을 사용합니다.', required=False, type=str),
    ],
    responses=inline_serializer(
        name='RepoGraphResponse',
        fields={
            'analysis_id': serializers.IntegerField(allow_null=True, help_text='이 graph가 나온 분석 run ID입니다. summary/qa에 재사용할 수 있습니다.'),
            'repo': serializers.CharField(help_text='정규화된 owner/repo 형식의 레포 이름입니다.'),
            'revision': serializers.CharField(help_text='실제로 분석된 git commit SHA입니다. 같은 graph를 다시 볼 때 query revision으로 재사용합니다.'),
            'analysis_profile': serializers.CharField(help_text='분석 profile ID입니다.'),
            'nodes': serializers.JSONField(help_text='그래프 노드 목록입니다. 각 node의 id는 node-summary/qa selected_node_id에 재사용합니다.'),
            'edges': serializers.JSONField(help_text='그래프 엣지 목록입니다. source/target은 nodes[].id를 참조합니다.'),
            'entrypoints': serializers.JSONField(help_text='파서가 추정한 진입점 목록입니다.'),
            'key_modules': serializers.JSONField(help_text='중요도가 높게 계산된 모듈 목록입니다.'),
            'warnings': serializers.JSONField(help_text='분석 경고 목록입니다.'),
        },
    ),
)
@api_view(['GET'])
def get_repo_graph(request):
    request_data = {'repo_url': request.GET.get('url')}
    if request.GET.get('revision') is not None:
        request_data['revision'] = request.GET.get('revision')
    if request.GET.get('analysis_profile') is not None:
        request_data['analysis_profile'] = request.GET.get('analysis_profile')
    serializer = AnalysisRequestSerializer(data=request_data)
    if not serializer.is_valid():
        logger.warning(f"잘못된 URL 요청: {request.GET.get('url')}")
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, str], serializer.validated_data)
    repo_path = validated_data['repo_url']
    revision = validated_data.get('revision')
    analysis_profile = validated_data.get('analysis_profile')
    logger.info(f"그래프 요청: {repo_path}")

    try:
        analysis = get_repo_analysis(repo_path, revision, analysis_profile=analysis_profile)
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)

    if analysis is None:
        logger.error(f"레포 찾기 실패: {repo_path}")
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    logger.info(f"그래프 반환 완료: {repo_path}")

    analysis_run = get_analysis_run_by_revision(repo_path, str(analysis['revision']), analysis_profile=analysis_profile)
    return Response(build_graph_response(analysis, analysis_run))


@extend_schema(
    operation_id='github_issues_list',
    description=_ISSUES_LIST_DESCRIPTION,
    parameters=[
        OpenApiParameter(name='url', description='필수. open issue 목록을 조회할 public GitHub 레포 URL입니다. 예: https://github.com/psf/requests', required=True, type=str),
        OpenApiParameter(name='page', description='선택. 1부터 시작하는 페이지 번호입니다. 기본값 1.', required=False, type=int),
        OpenApiParameter(name='per_page', description='선택. 페이지당 issue 수입니다. 1-100, 기본값 30.', required=False, type=int),
        OpenApiParameter(name='state', description='선택. 현재는 open만 지원합니다. 생략하면 open입니다.', required=False, type=str),
        OpenApiParameter(name='mock', description='선택. true이면 live GitHub 대신 mock issue 목록을 반환합니다.', required=False, type=bool),
    ],
    responses=IssueListResponseSerializer,
    examples=[
        OpenApiExample(
            'Mock open issues response',
            value=_ISSUES_LIST_RESPONSE_EXAMPLE,
            response_only=True,
        ),
    ],
)
@api_view(['GET'])
def issues(request):
    request_data = {'repo_url': request.GET.get('url')}
    for field_name in ('page', 'per_page', 'state', 'mock'):
        if request.GET.get(field_name) is not None:
            request_data[field_name] = request.GET.get(field_name)
    serializer = IssueListRequestSerializer(data=request_data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, Any], serializer.validated_data)
    repo_path = str(validated_data['repo_url'])
    page = int(validated_data['page'])
    per_page = int(validated_data['per_page'])
    if bool(validated_data.get('mock')) or bool(getattr(settings, 'ISSUES_USE_MOCK', False)):
        response = get_mock_issue_list_response(repo_path, page=page, per_page=per_page)
        return Response(response)

    try:
        response = get_live_issue_list_response(
            repo_path,
            page=page,
            per_page=per_page,
            state=str(validated_data['state']),
        )
    except GithubIssueApiError as error:
        return _github_issue_api_error_response(error)
    return Response(response)


@extend_schema(
    operation_id='github_issue_related_nodes',
    description=_ISSUE_RELATED_NODES_DESCRIPTION,
    request=IssueRelatedNodesRequestSerializer,
    responses=IssueRelatedNodesResponseSerializer,
    examples=[
        OpenApiExample(
            'Related nodes request',
            value=_ISSUE_RELATED_NODES_REQUEST_EXAMPLE,
            request_only=True,
        ),
        OpenApiExample(
            'Mock related nodes response',
            value=_ISSUE_RELATED_NODES_RESPONSE_EXAMPLE,
            response_only=True,
        ),
    ],
)
@api_view(['POST'])
def issue_related_nodes(request):
    serializer = IssueRelatedNodesRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, Any], serializer.validated_data)
    analysis_id = int(validated_data['analysis_id'])
    issue_number = int(validated_data['issue_number'])
    max_nodes = int(validated_data['max_nodes'])
    if bool(validated_data.get('mock')) or bool(getattr(settings, 'ISSUES_USE_MOCK', False)):
        response = get_mock_issue_related_nodes_response(analysis_id, issue_number, max_nodes=max_nodes)
        if response is None:
            return Response({'error': '분석 결과 또는 issue를 찾을 수 없습니다'}, status=404)
        return Response(response)

    try:
        response = get_issue_map_response(
            analysis_id,
            issue_number,
            max_nodes=max_nodes,
            include_comments=bool(validated_data['include_comments']),
            max_context_files=int(validated_data['max_context_files']),
        )
    except GithubIssueApiError as error:
        return _github_issue_api_error_response(error)
    except Exception as error:
        return _issue_map_error_response(error)
    return Response(response)


_SUMMARY_RESPONSE_SCHEMA = inline_serializer(
    name='SummaryResponse',
    fields={
        'analysis_id': serializers.IntegerField(help_text='요약 대상 분석 run ID입니다.'),
        'repo': serializers.CharField(help_text='요약 대상 owner/repo입니다.'),
        'revision': serializers.CharField(help_text='요약 대상 git commit SHA입니다.'),
        'summary': serializers.JSONField(help_text='LLM이 생성한 요약 결과입니다.'),
        'cached': serializers.BooleanField(help_text='true면 기존 저장된 요약을 반환한 것입니다. false면 이번 요청에서 생성했습니다.'),
    },
)


@extend_schema(
    description=_SUMMARY_DESCRIPTION,
    parameters=[
        OpenApiParameter(name='analysis_id', description='필수. `/api/analysis/` 응답의 analysis_id.', required=True, type=int),
        OpenApiParameter(name='kind', description='선택. repo_overview 또는 onboarding_guide. 생략하면 repo_overview.', required=False, type=str),
    ],
    responses=_SUMMARY_RESPONSE_SCHEMA,
)
@api_view(['GET'])
def summary(request):
    serializer = SummaryRequestSerializer(data=request.GET)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, Any], serializer.validated_data)
    try:
        response = get_or_create_summary_response(int(validated_data['analysis_id']), str(validated_data['kind']))
    except Exception as error:
        return _summary_error_response(error)
    if response is None:
        return Response({'error': '분석 결과를 찾을 수 없습니다'}, status=404)
    return Response(response)


@extend_schema(
    description=_NODE_SUMMARY_DESCRIPTION,
    parameters=[
        OpenApiParameter(name='analysis_id', description='필수. `/api/analysis/` 응답의 analysis_id.', required=True, type=int),
        OpenApiParameter(name='node_id', description='필수. `/api/graph/` 응답의 nodes[].id.', required=True, type=str),
    ],
    responses=_SUMMARY_RESPONSE_SCHEMA,
)
@api_view(['GET'])
def node_summary(request):
    serializer = NodeSummaryRequestSerializer(data=request.GET)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, Any], serializer.validated_data)
    try:
        response = get_or_create_node_summary_response(int(validated_data['analysis_id']), str(validated_data['node_id']))
    except Exception as error:
        return _summary_error_response(error)
    if response is None:
        return Response({'error': '분석 결과를 찾을 수 없습니다'}, status=404)
    return Response(response)


@extend_schema(
    description=_QA_DESCRIPTION,
    request=inline_serializer(
        name='QARequest',
        fields={
            'repo_url': serializers.CharField(required=False, help_text='analysis_id가 없을 때 필요한 GitHub 레포 URL입니다.'),
            'question': serializers.CharField(help_text='레포에 대해 물어볼 질문입니다.'),
            'revision': serializers.CharField(required=False, help_text='repo_url 방식에서 특정 revision을 고정할 때 사용합니다.'),
            'analysis_profile': serializers.CharField(required=False, help_text='repo_url 방식에서 사용할 분석 profile입니다. python-v1 또는 multi-lang-js-ts-v1.'),
            'analysis_id': serializers.IntegerField(required=False, min_value=1, help_text='/api/analysis/ 응답의 analysis_id입니다. 있으면 repo_url보다 우선 사용합니다.'),
            'selected_node_id': serializers.CharField(required=False, help_text='/api/graph/ 응답의 nodes[].id입니다. 선택 노드 중심으로 답변할 때 사용합니다.'),
            'selected_file_path': serializers.CharField(required=False, help_text='선택 파일 중심으로 답변할 때 사용합니다.'),
            'max_context_files': serializers.IntegerField(required=False, min_value=1, max_value=10, help_text='답변에 사용할 최대 context 파일 수입니다. 기본값 4.'),
            'stream': serializers.BooleanField(required=False, help_text='true이거나 Accept: text/event-stream이면 SSE를 반환합니다. token.data.text, meta, final, error 및 harness_* 진행 이벤트를 사용할 수 있습니다.'),
            'model': serializers.CharField(required=False, help_text='선택할 OpenCode Zen 모델 ID입니다. 생략하면 config.yaml의 opencode.model을 사용합니다.'),
        }
    ),
    responses=inline_serializer(
        name='QAResponse',
        fields={
            'answer': serializers.CharField(),
            'citations': serializers.ListField(child=serializers.CharField()),
            'selected_nodes': serializers.ListField(child=serializers.CharField()),
            'context_files': serializers.ListField(child=serializers.CharField()),
            'context_summary': serializers.JSONField(),
            'tool_trace': serializers.JSONField(),
            'warnings': serializers.JSONField(),
        },
    ),
)
@api_view(['POST'])
@renderer_classes([JSONRenderer, SseRenderer])
def qa(request):
    serializer = QASerializer(data=request.data)
    if not serializer.is_valid():
        logger.warning(f"잘못된 QA 요청: {request.data}")
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, Any], serializer.validated_data)
    question = validated_data['question']
    analysis_id = validated_data.get('analysis_id')
    revision = validated_data.get('revision')
    analysis_profile = validated_data.get('analysis_profile')
    selected_node_id = validated_data.get('selected_node_id')
    selected_file_path = validated_data.get('selected_file_path')
    max_context_files = int(validated_data.get('max_context_files', 4))
    model = validated_data.get('model')
    repo_path = str(validated_data.get('repo_url') or '')

    if analysis_id is not None:
        analysis_response = get_analysis_response_by_id(int(analysis_id))
        if analysis_response is None:
            return Response({'error': '분석 결과를 찾을 수 없습니다'}, status=404)
        artifact = analysis_response.get('artifact')
        if artifact is None:
            return Response(
                {
                    'error': 'QA에 사용할 분석 artifact가 없습니다',
                    'detail': analysis_response.get('error'),
                },
                status=409,
            )
        analysis = cast(dict[str, Any], artifact)
        repo_path = str(analysis_response['repo'])
    else:
        try:
            analysis = get_repo_analysis(repo_path, revision, analysis_profile=analysis_profile)
        except RepoIngestionError as error:
            return _repo_ingestion_error_response(error)

        if analysis is None:
            logger.error(f"레포 찾기 실패: {repo_path}")
            return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    logger.info(f"QA 요청: {repo_path} / 질문: {question}")
    if _qa_stream_requested(request, validated_data):
        logger.info(f"QA 스트리밍 시작: {repo_path}")
        return _qa_streaming_response(
            repo_path,
            cast(dict[str, Any], analysis),
            question,
            selected_node_id=selected_node_id,
            selected_file_path=selected_file_path,
            max_context_files=max_context_files,
            model=model,
        )

    answer = answer_question(
        repo_path,
        cast(dict[str, Any], analysis),
        question,
        selected_node_id=selected_node_id,
        selected_file_path=selected_file_path,
        max_context_files=max_context_files,
        model=model,
    )
    logger.info(f"QA 완료: {repo_path}")

    return Response(answer)
