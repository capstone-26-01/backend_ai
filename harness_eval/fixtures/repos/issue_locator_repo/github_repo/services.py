def fetch_repo_files(repo_id, revision):
    response = github_client_get(repo_id, revision)
    return response.get("files", [])


def github_client_get(repo_id, revision):
    if revision == "missing":
        return None
    return {
        "files": [
            {"path": "api/services.py", "content": "def example(): pass"},
            {"path": "parser/services.py", "content": "async def load(): pass"},
        ]
    }


def normalize_repo_path(repo_url):
    return repo_url.removeprefix("https://github.com/")
