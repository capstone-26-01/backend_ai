import importlib
import logging
from typing import cast
from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter, inline_serializer
from rest_framework import serializers

from github_repo.services import RepoIngestionError, get_file_tree_or_raise
from llm.services import answer_question
from .serializers import AnalysisRequestSerializer, RepoUrlSerializer, QASerializer, is_safe_revision

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


def _repo_ingestion_error_response(error: RepoIngestionError) -> Response:
    status_by_code = {
        'invalid_repo_path': 400,
        'unsafe_path': 400,
        'repo_not_found': 404,
        'private_repo': 404,
        'timeout': 504,
        'too_large': 413,
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


@extend_schema(
    request=inline_serializer(
        name='QARequest',
        fields={
            'repo_url': serializers.CharField(),
            'question': serializers.CharField(),
            'revision': serializers.CharField(required=False),
        }
    ),
    responses=inline_serializer(
        name='QAResponse',
        fields={
            'answer': serializers.CharField(),
            'citations': serializers.ListField(child=serializers.CharField()),
        },
    ),
)
@api_view(['POST'])
def qa(request):
    serializer = QASerializer(data=request.data)
    if not serializer.is_valid():
        logger.warning(f"잘못된 QA 요청: {request.data}")
        return Response(serializer.errors, status=400)

    validated_data = cast(dict[str, str], serializer.validated_data)
    repo_path = validated_data['repo_url']
    question = validated_data['question']
    revision = validated_data.get('revision')
    logger.info(f"QA 요청: {repo_path} / 질문: {question}")

    try:
        analysis = get_repo_analysis(repo_path, revision)
    except RepoIngestionError as error:
        return _repo_ingestion_error_response(error)

    if analysis is None:
        logger.error(f"레포 찾기 실패: {repo_path}")
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    answer = answer_question(repo_path, analysis, question)
    logger.info(f"QA 완료: {repo_path}")

    return Response(answer)
