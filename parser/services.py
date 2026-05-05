from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())


def parse_python_file(code, file_path):
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(bytes(code, 'utf-8'))
    root = tree.root_node

    nodes = []
    edges = []
    tree_children = []

    for node in root.children:
        if node.type == 'class_definition':
            class_name = node.child_by_field_name('name').text.decode('utf-8')
            class_id = f"{file_path}::{class_name}"
            
            # 메서드 파싱
            method_children = []
            body = node.child_by_field_name('body')
            if body:
                for child in body.children:
                    if child.type == 'function_definition':
                        func_name = child.child_by_field_name('name').text.decode('utf-8')
                        func_id = f"{class_id}::{func_name}"

                        # nodes에 추가
                        nodes.append({
                            "id": func_id,
                            "type": "function",
                            "label": func_name,
                            "file": file_path
                        })

                        # tree children에 추가
                        method_children.append({
                            "id": func_id,
                            "type": "function",
                            "label": func_name
                        })

            # nodes에 클래스 추가
            nodes.append({
                "id": class_id,
                "type": "class",
                "label": class_name,
                "file": file_path
            })

            # tree children에 클래스 추가
            tree_children.append({
                "id": class_id,
                "type": "class",
                "label": class_name,
                "children": method_children
            })

            # 상속관계 edges 추가
            superclasses = node.child_by_field_name('superclasses')
            if superclasses:
                for child in superclasses.children:
                    if child.type == 'identifier':
                        parent_name = child.text.decode('utf-8')
                        # 부모 클래스 id는 나중에 resolve (일단 이름만)
                        edges.append({
                            "source": class_id,
                            "target": parent_name,  # resolve_edges()에서 처리
                            "type": "inherits"
                        })

        elif node.type == 'function_definition':
            func_name = node.child_by_field_name('name').text.decode('utf-8')
            func_id = f"{file_path}::{func_name}"

            nodes.append({
                "id": func_id,
                "type": "function",
                "label": func_name,
                "file": file_path
            })

            tree_children.append({
                "id": func_id,
                "type": "function",
                "label": func_name,
                "children": []
            })

    # file 트리 노드
    file_tree_node = {
        "id": file_path,
        "type": "file",
        "label": file_path.split("/")[-1],
        "children": tree_children
    }

    # file 노드 (flat)
    file_node = {
        "id": file_path,
        "type": "file",
        "label": file_path.split("/")[-1]
    }

    return file_tree_node, file_node, nodes, edges


def resolve_edges(edges, nodes):
    """
    상속관계 edges의 target이 현재 클래스명만 있는 경우,
    nodes에서 실제 id(file_path::ClassName)로 매핑
    """
    # 클래스명 -> id 매핑 테이블
    class_name_to_id = {
        node["label"]: node["id"]
        for node in nodes
        if node["type"] == "class"
    }

    resolved = []
    for i, edge in enumerate(edges):
        target = edge["target"]
        # target이 이미 :: 포함한 full id면 그대로
        if "::" not in target:
            target = class_name_to_id.get(target, target)
        resolved.append({
            "id": f"e{i+1}",
            "source": edge["source"],
            "target": target,
            "type": edge["type"]
        })
    return resolved


def parse_repo(repo_path, files, get_content_func):
    all_tree = []
    all_nodes = []
    all_edges = []

    for file_path in files:
        if not file_path.endswith('.py'):
            continue

        code = get_content_func(repo_path, file_path)
        if not code:
            continue

        file_tree_node, file_node, nodes, edges = parse_python_file(code, file_path)

        all_tree.append(file_tree_node)
        all_nodes.append(file_node)
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    # 상속관계 target id resolve
    all_edges = resolve_edges(all_edges, all_nodes)

    return {
        "tree": all_tree,
        "nodes": all_nodes,
        "edges": all_edges
    }