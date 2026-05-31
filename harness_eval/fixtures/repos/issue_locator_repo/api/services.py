from graph.builders import attach_labels, build_dependency_graph
from github_repo.services import fetch_repo_files
from parser.cache import read_cache, write_cache
from parser.services import parse_repo


def get_repo_analysis(repo_id, revision="main"):
    cached = read_cache(repo_id)
    if cached and cached.get("status") == "ready":
        return cached["analysis"]
    return _build_and_store_analysis(repo_id, revision)


def _build_and_store_analysis(repo_id, revision):
    files = fetch_repo_files(repo_id, revision)
    symbols = parse_repo(files)
    graph = build_dependency_graph(symbols)
    labeled = attach_labels(graph, symbols)
    write_cache(repo_id, labeled)
    return labeled


def summarize_repo(repo_id):
    graph = get_repo_analysis(repo_id)
    return {"node_count": len(graph.get("nodes", []))}


def delete_share_token(token):
    return token.removeprefix("share_")
