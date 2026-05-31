class BasePermission:
    def can_view(self, user, repo):
        return False


class RepoPermission(BasePermission):
    def can_view(self, user, repo):
        if not repo.get("private"):
            return True
        return bool(getattr(user, "is_authenticated", False))


def ensure_public_repo(repo):
    if repo.get("private"):
        raise PermissionError("private repository")
    return True


def require_repo_access(user, repo):
    return RepoPermission().can_view(user, repo)
