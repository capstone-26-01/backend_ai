import logging
from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter, inline_serializer
from rest_framework import serializers

from parser.services import parse_repo
from github_repo.services import get_file_tree, get_file_content
from llm.services import answer_question
from .serializers import RepoUrlSerializer, QASerializer

logger = logging.getLogger(__name__)


@extend_schema(
    parameters=[
        OpenApiParameter(name='url', description='GitHub 레포 URL', required=True, type=str)
    ]
)
@api_view(['GET'])
def get_repo_file(request):
    serializer = RepoUrlSerializer(data={'repo_url': request.GET.get('url')})
    if not serializer.is_valid():
        logger.warning(f"잘못된 URL 요청: {request.GET.get('url')}")
        return Response(serializer.errors, status=400)

    repo_path = serializer.validated_data['repo_url']
    logger.info(f"파일트리 요청: {repo_path}")

    files = get_file_tree(repo_path)
    if not files:
        logger.error(f"레포 찾기 실패: {repo_path}")
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    logger.info(f"파일트리 반환 완료: {repo_path}")
    return Response({'repo': repo_path, 'files': files})


@extend_schema(
    parameters=[
        OpenApiParameter(name='url', description='GitHub 레포 URL', required=True, type=str)
    ]
)
@api_view(['GET'])
def get_repo_tree(request):
    serializer = RepoUrlSerializer(data={'repo_url': request.GET.get('url')})
    if not serializer.is_valid():
        logger.warning(f"잘못된 URL 요청: {request.GET.get('url')}")
        return Response(serializer.errors, status=400)

    repo_path = serializer.validated_data['repo_url']
    logger.info(f"트리 요청: {repo_path}")

    files = get_file_tree(repo_path)
    if not files:
        logger.error(f"레포 찾기 실패: {repo_path}")
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    graph = parse_repo(repo_path, files, get_file_content)
    logger.info(f"트리 반환 완료: {repo_path}")

    return Response({'repo': repo_path, 'tree': graph['tree']})


@extend_schema(
    parameters=[
        OpenApiParameter(name='url', description='GitHub 레포 URL', required=True, type=str)
    ]
)
@api_view(['GET'])
def get_repo_graph(request):
    serializer = RepoUrlSerializer(data={'repo_url': request.GET.get('url')})
    if not serializer.is_valid():
        logger.warning(f"잘못된 URL 요청: {request.GET.get('url')}")
        return Response(serializer.errors, status=400)

    repo_path = serializer.validated_data['repo_url']
    logger.info(f"그래프 요청: {repo_path}")

    files = get_file_tree(repo_path)
    if not files:
        logger.error(f"레포 찾기 실패: {repo_path}")
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    graph = parse_repo(repo_path, files, get_file_content)
    logger.info(f"그래프 반환 완료: {repo_path}")

    return Response({'repo': repo_path, 'nodes': graph['nodes'], 'edges': graph['edges']})


@extend_schema(
    request=inline_serializer(
        name='QARequest',
        fields={
            'repo_url': serializers.CharField(),
            'question': serializers.CharField(),
        }
    )
)
@api_view(['POST'])
def qa(request):
    serializer = QASerializer(data=request.data)
    if not serializer.is_valid():
        logger.warning(f"잘못된 QA 요청: {request.data}")
        return Response(serializer.errors, status=400)

    repo_path = serializer.validated_data['repo_url']
    question = serializer.validated_data['question']
    logger.info(f"QA 요청: {repo_path} / 질문: {question}")

    files = get_file_tree(repo_path)
    if not files:
        logger.error(f"레포 찾기 실패: {repo_path}")
        return Response({'error': '레포를 찾을 수 없습니다'}, status=404)

    answer = answer_question(repo_path, files, question, get_file_content)
    logger.info(f"QA 완료: {repo_path}")

    return Response({'answer': answer})