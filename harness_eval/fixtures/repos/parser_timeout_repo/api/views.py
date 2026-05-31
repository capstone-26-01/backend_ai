from api.services import get_repo_analysis


def analysis(request):
    return get_repo_analysis(request.repo_path)
