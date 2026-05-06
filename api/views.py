import importlib
import logging
from typing import cast
from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter, inline_serializer
from rest_framework import serializers

from github_repo.services import get_file_tree
from llm.services import answer_question
from .serializers import RepoUrlSerializer, QASerializer, is_safe_revision

logger = logging.getLogger(__name__)
api_services = importlib.import_module('api.services')


def get_repo_analysis(repo_path: str, revision: str | None = None):
    return api_services.get_repo_analysis(repo_path, revision)


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

    files = get_file_tree(repo_path)
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
            'repo': serializers.CharField(),
            'revision': serializers.CharField(),
            'tree': serializers.JSONField(),
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

    analysis = get_repo_analysis(repo_path, revision)
    if analysis is None:
        logger.error(f"레포 찾기 실패: {repo_path}")
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    logger.info(f"트리 반환 완료: {repo_path}")

    return Response({'repo': repo_path, 'tree': analysis['tree'], 'revision': analysis['revision']})


@extend_schema(
    parameters=[
        OpenApiParameter(name='url', description='GitHub 레포 URL', required=True, type=str),
        OpenApiParameter(name='revision', description='캐시된 분석 revision', required=False, type=str),
    ],
    responses=inline_serializer(
        name='RepoGraphResponse',
        fields={
            'repo': serializers.CharField(),
            'revision': serializers.CharField(),
            'nodes': serializers.JSONField(),
            'edges': serializers.JSONField(),
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

    analysis = get_repo_analysis(repo_path, revision)
    if analysis is None:
        logger.error(f"레포 찾기 실패: {repo_path}")
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    logger.info(f"그래프 반환 완료: {repo_path}")

    return Response({
        'repo': repo_path,
        'nodes': analysis['nodes'],
        'edges': analysis['edges'],
        'revision': analysis['revision'],
    })


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

    analysis = get_repo_analysis(repo_path, revision)
    if analysis is None:
        logger.error(f"레포 찾기 실패: {repo_path}")
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    answer = answer_question(repo_path, analysis, question)
    logger.info(f"QA 완료: {repo_path}")

    return Response(answer)
