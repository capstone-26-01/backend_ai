from parser.services import parse_repo


def _build_and_store_analysis(repo_path):
    tree = parse_repo(repo_path)
    return {"tree": tree, "status": "stored"}


def get_repo_analysis(repo_path):
    return _build_and_store_analysis(repo_path)
