import importlib
import logging
from typing import Any, cast
from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter, inline_serializer
from rest_framework import serializers

from github_repo.services import RepoIngestionError, get_file_tree_or_raise
from llm.services import answer_question
from .serializers import (
    AnalysisDiffRequestSerializer,
    AnalysisRequestSerializer,
    DiffRequestSerializer,
    NodeSummaryRequestSerializer,
    RepoUrlSerializer,
    QASerializer,
    SummaryRequestSerializer,
    is_safe_revision,
)

logger = logging.getLogger(__name__)
api_services = importlib.import_module('api.services')


def get_repo_analysis(repo_path: str, revision: str | None = None):
    return api_services.get_repo_analysis(repo_path, revision)


def get_analysis_response(repo_path: str, revision: str | None = None):
    return api_services.get_analysis_response(repo_path, revision)


def get_analysis_response_by_id(analysis_id: int):
    return api_services.get_analysis_response_by_id(analysis_id)


def get_analysis_run_by_revision(repo_path: str, revision: str):
    return api_services.get_analysis_run_by_revision(repo_path, revision)


def build_tree_response(payload, analysis_run=None):
    return api_services.build_tree_response(payload, analysis_run)


def build_graph_response(payload, analysis_run=None):
    return api_services.build_graph_response(payload, analysis_run)


def get_or_create_summary_response(analysis_id: int, kind: str):
    return api_services.get_or_create_summary_response(analysis_id, kind)


def get_or_create_node_summary_response(analysis_id: int, node_id: str):
    return api_services.get_or_create_node_summary_response(analysis_id, node_id)


def get_diff_response(repo_path: str, base_revision: str, head_revision: str | None = None):
    return api_services.get_diff_response(repo_path, base_revision, head_revision)


def get_diff_response_by_analysis_id(head_analysis_id: int, base_analysis_id: int):
    return api_services.get_diff_response_by_analysis_id(head_analysis_id, base_analysis_id)


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


_ANALYSIS_REQUEST_SCHEMA = inline_serializer(
    name='AnalysisRequest',
    fields={
        'repo_url': serializers.CharField(),
        'revision': serializers.CharField(required=False),
    },
)
_ANALYSIS_RESPONSE_SCHEMA = inline_serializer(
    name='AnalysisResponse',
    fields={
        'analysis_id': serializers.IntegerField(allow_null=True),
        'repo': serializers.CharField(),
        'revision': serializers.CharField(),
        'status': serializers.CharField(),
        'artifact': serializers.JSONField(allow_null=True),
        'warnings': serializers.JSONField(),
    },
)
_ANALYSIS_DETAIL_RESPONSE_SCHEMA = inline_serializer(
    name='AnalysisByIdResponse',
    fields={
        'analysis_id': serializers.IntegerField(),
        'repo': serializers.CharField(),
        'revision': serializers.CharField(allow_blank=True),
        'status': serializers.CharField(),
        'artifact': serializers.JSONField(allow_null=True),
        'warnings': serializers.JSONField(),
        'error': serializers.JSONField(required=False, allow_null=True),
    },
)
_DIFF_RESPONSE_SCHEMA = inline_serializer(
    name='GraphDiffResponse',
    fields={
        'repo': serializers.CharField(),
        'base': serializers.JSONField(),
        'head': serializers.JSONField(),
        'diff': serializers.JSONField(),
        'warnings': serializers.JSONField(),
    },
)


@extend_schema(
    methods=['GET'],
    operation_id='analysis_retrieve_by_url',
    parameters=[
        OpenApiParameter(name='url', description='GitHub 레포 URL', required=True, type=str),
        OpenApiParameter(name='revision', description='캐시된 분석 revision', required=False, type=str),
    ],
    responses=_ANALYSIS_RESPONSE_SCHEMA,
)
@extend_schema(
    methods=['POST'],
    operation_id='analysis_create',
    request=_ANALYSIS_REQUEST_SCHEMA,
    responses=_ANALYSIS_RESPONSE_SCHEMA,
)
@api_view(['GET', 'POST'])
def analysis(request):
    if request.method == 'GET':
        request_data = {'repo_url': request.GET.get('url')}
        if request.GET.get('revision') is not None:
            request_data['revision'] = request.GET.get('revision')
        serializer = AnalysisRequestSerializer(data=request_data)
    else:
        serializer = AnalysisRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, str], serializer.validated_data)
    repo_path = validated_data['repo_url']
    revision = validated_data.get('revision')
    try:
        response = get_analysis_response(repo_path, revision)
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)

    if response is None:
        return Response({'error': '분석 결과를 찾을 수 없습니다'}, status=404)
    return Response(response)


@extend_schema(
    operation_id='analysis_retrieve_by_id',
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
    parameters=[
        OpenApiParameter(name='base', description='비교 기준 분석 run ID', required=True, type=int),
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
    parameters=[
        OpenApiParameter(name='url', description='GitHub 레포 URL', required=True, type=str),
        OpenApiParameter(name='base', description='기준 revision', required=True, type=str),
        OpenApiParameter(name='head', description='대상 revision. 생략하면 latest HEAD', required=False, type=str),
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
    serializer = DiffRequestSerializer(data=request_data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, str], serializer.validated_data)
    try:
        response = get_diff_response(
            validated_data['repo_url'],
            validated_data['base'],
            validated_data.get('head'),
        )
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)
    except Exception as error:
        return _diff_error_response(error)
    if response is None:
        return Response({'error': '비교할 분석 결과를 찾을 수 없습니다'}, status=404)
    return Response(response)


@extend_schema(
    parameters=[
        OpenApiParameter(name='url', description='GitHub 레포 URL', required=True, type=str),
    ],
    responses=inline_serializer(
        name='RepoFilesResponse',
        fields={
            'repo': serializers.CharField(),
            'files': serializers.ListField(child=serializers.CharField()),
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
    parameters=[
        OpenApiParameter(name='url', description='GitHub 레포 URL', required=True, type=str),
        OpenApiParameter(name='revision', description='캐시된 분석 revision', required=False, type=str),
    ],
    responses=inline_serializer(
        name='RepoTreeResponse',
        fields={
            'analysis_id': serializers.IntegerField(allow_null=True),
            'repo': serializers.CharField(),
            'revision': serializers.CharField(),
            'tree': serializers.JSONField(),
            'warnings': serializers.JSONField(),
        },
    ),
)
@api_view(['GET'])
def get_repo_tree(request):
    serializer = RepoUrlSerializer(data={'repo_url': request.GET.get('url')})
    if not serializer.is_valid():
        logger.warning(f"잘못된 URL 요청: {request.GET.get('url')}")
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, str], serializer.validated_data)
    repo_path = validated_data['repo_url']
    revision = request.GET.get('revision')
    if revision is not None and not is_safe_revision(revision):
        return Response({'revision': ['올바른 revision이 아닙니다']}, status=400)
    logger.info(f"트리 요청: {repo_path}")

    try:
        analysis = get_repo_analysis(repo_path, revision)
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)

    if analysis is None:
        logger.error(f"레포 찾기 실패: {repo_path}")
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    logger.info(f"트리 반환 완료: {repo_path}")

    analysis_run = get_analysis_run_by_revision(repo_path, str(analysis['revision']))
    return Response(build_tree_response(analysis, analysis_run))


@extend_schema(
    parameters=[
        OpenApiParameter(name='url', description='GitHub 레포 URL', required=True, type=str),
        OpenApiParameter(name='revision', description='캐시된 분석 revision', required=False, type=str),
    ],
    responses=inline_serializer(
        name='RepoGraphResponse',
        fields={
            'analysis_id': serializers.IntegerField(allow_null=True),
            'repo': serializers.CharField(),
            'revision': serializers.CharField(),
            'nodes': serializers.JSONField(),
            'edges': serializers.JSONField(),
            'entrypoints': serializers.JSONField(),
            'key_modules': serializers.JSONField(),
            'warnings': serializers.JSONField(),
        },
    ),
)
@api_view(['GET'])
def get_repo_graph(request):
    serializer = RepoUrlSerializer(data={'repo_url': request.GET.get('url')})
    if not serializer.is_valid():
        logger.warning(f"잘못된 URL 요청: {request.GET.get('url')}")
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, str], serializer.validated_data)
    repo_path = validated_data['repo_url']
    revision = request.GET.get('revision')
    if revision is not None and not is_safe_revision(revision):
        return Response({'revision': ['올바른 revision이 아닙니다']}, status=400)
    logger.info(f"그래프 요청: {repo_path}")

    try:
        analysis = get_repo_analysis(repo_path, revision)
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)

    if analysis is None:
        logger.error(f"레포 찾기 실패: {repo_path}")
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    logger.info(f"그래프 반환 완료: {repo_path}")

    analysis_run = get_analysis_run_by_revision(repo_path, str(analysis['revision']))
    return Response(build_graph_response(analysis, analysis_run))


_SUMMARY_RESPONSE_SCHEMA = inline_serializer(
    name='SummaryResponse',
    fields={
        'analysis_id': serializers.IntegerField(),
        'repo': serializers.CharField(),
        'revision': serializers.CharField(),
        'summary': serializers.JSONField(),
        'cached': serializers.BooleanField(),
    },
)


@extend_schema(
    parameters=[
        OpenApiParameter(name='analysis_id', description='분석 run ID', required=True, type=int),
        OpenApiParameter(name='kind', description='repo_overview 또는 onboarding_guide', required=False, type=str),
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
    parameters=[
        OpenApiParameter(name='analysis_id', description='분석 run ID', required=True, type=int),
        OpenApiParameter(name='node_id', description='요약할 graph node ID', required=True, type=str),
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
    request=inline_serializer(
        name='QARequest',
        fields={
            'repo_url': serializers.CharField(),
            'question': serializers.CharField(),
            'revision': serializers.CharField(required=False),
            'analysis_id': serializers.IntegerField(required=False, min_value=1),
            'selected_node_id': serializers.CharField(required=False),
            'selected_file_path': serializers.CharField(required=False),
            'max_context_files': serializers.IntegerField(required=False, min_value=1, max_value=10),
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
def qa(request):
    serializer = QASerializer(data=request.data)
    if not serializer.is_valid():
        logger.warning(f"잘못된 QA 요청: {request.data}")
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, Any], serializer.validated_data)
    question = validated_data['question']
    analysis_id = validated_data.get('analysis_id')
    revision = validated_data.get('revision')
    selected_node_id = validated_data.get('selected_node_id')
    selected_file_path = validated_data.get('selected_file_path')
    max_context_files = int(validated_data.get('max_context_files', 4))
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
            analysis = get_repo_analysis(repo_path, revision)
        except RepoIngestionError as error:
            return _repo_ingestion_error_response(error)

        if analysis is None:
            logger.error(f"레포 찾기 실패: {repo_path}")
            return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    logger.info(f"QA 요청: {repo_path} / 질문: {question}")
    answer = answer_question(
        repo_path,
        cast(dict[str, Any], analysis),
        question,
        selected_node_id=selected_node_id,
        selected_file_path=selected_file_path,
        max_context_files=max_context_files,
    )
    logger.info(f"QA 완료: {repo_path}")

    return Response(answer)
