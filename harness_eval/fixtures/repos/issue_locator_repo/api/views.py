from api.services import get_repo_analysis, summarize_repo
from auth.permissions import RepoPermission


def analysis(request):
    return get_repo_analysis(request.repo_id, request.revision)


class SummaryView:
    permission = RepoPermission()

    def get(self, request, repo_id):
        repo = {"id": repo_id, "private": request.private, "owner_id": request.owner_id}
        if not self.permission.can_view(request.user, repo):
            return {"status": 403}
        return {"status": 200, "summary": summarize_repo(repo_id)}


def readme_graph(request):
    return {"format": "svg", "repo": request.repo_id}
