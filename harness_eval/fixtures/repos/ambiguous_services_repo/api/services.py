def get_repo_analysis(repo_id):
    analysis = _build_and_store_analysis(repo_id)
    return analysis


def _build_and_store_analysis(repo_id):
    return {"repo_id": repo_id, "fresh": False}


def delete_share_token(token):
    return token.replace("share_", "")
