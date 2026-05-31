from parser.cache import invalidate_issue_cache, write_cache


async def refresh_issue_cache(issue):
    analysis = issue["analysis"]
    write_cache(issue["id"], analysis)
    return {"refreshed": True}


def close_issue(issue):
    issue["state"] = "closed"
    invalidate_issue_cache(issue["id"])
    return issue


def schedule_analysis(repo_id):
    return {"queue": "analysis", "repo_id": repo_id}
