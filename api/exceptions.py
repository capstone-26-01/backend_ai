import logging
from rest_framework.views import exception_handler
from rest_framework.response import Response

logger = logging.getLogger(__name__)


def custom_exception_handler(exc, context):
    #DRF 기본 예외 처리 먼저 시도
    response = exception_handler(exc, context)

    if response is not None:
        # DRF가 처리한 예외 (400, 401, 403, 404 등)
        logger.warning(f"클라이언트 에러: {exc}")
        return response

    # DRF가 처리 못한 예외 (GitHub API 오류, 파싱 오류 등 예상치 못한 에러)
    logger.error(f"서버 에러: {exc}", exc_info=True)#exc_info=True : 어디서 터졌는지 상세하게 출력
    return Response(
        {'error': '서버 오류가 발생했습니다', 'detail': str(exc)},
        status=500
    )