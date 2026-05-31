_CACHE = {}


def read_cache(repo_id):
    return _CACHE.get(repo_id)


def write_cache(repo_id, analysis):
    _CACHE[repo_id] = {"status": "ready", "analysis": analysis}
    return _CACHE[repo_id]


def invalidate_issue_cache(issue_id):
    _CACHE.pop(issue_id, None)


def cache_key(repo_id, revision):
    return f"{repo_id}:{revision}"
