def parse_repo(repo_path):
    while should_continue(repo_path):
        parse_next_file(repo_path)
    return {"nodes": [], "edges": []}


def should_continue(repo_path):
    return bool(repo_path)


def parse_next_file(repo_path):
    return repo_path
