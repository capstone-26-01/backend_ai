def build_dependency_graph(symbols):
    nodes = [
        {"id": symbol["id"], "path": symbol["path"]}
        for symbol in symbols
        if not symbol["name"].startswith("_")
    ]
    edges = connect_edges(symbols)
    return {"nodes": nodes, "edges": edges}


def attach_labels(graph, symbols):
    for node in graph.get("nodes", []):
        node["label"] = node["path"].split("/")[-1]
    return graph


def connect_edges(symbols):
    return [
        {"source": symbol["id"], "target": target}
        for symbol in symbols
        for target in symbol.get("calls", [])
    ]


def filter_internal_nodes(graph):
    graph["nodes"] = [node for node in graph.get("nodes", []) if "internal" not in node["id"]]
    return graph
