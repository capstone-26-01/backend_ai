from __future__ import annotations

from collections import defaultdict
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())


def _read_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode('utf-8')


def _get_name(node, source_bytes: bytes) -> str | None:
    name_node = node.child_by_field_name('name')
    if name_node is None:
        return None
    return _read_text(name_node, source_bytes)


def _walk(node):
    yield node
    for child in node.named_children:
        yield from _walk(child)


def _walk_function_body(node, *, is_root: bool = True):
    yield node
    for child in node.named_children:
        child = _unwrap_definition(child)
        if not is_root and child.type in {'function_definition', 'class_definition'}:
            continue
        yield from _walk_function_body(child, is_root=False)


def _unwrap_definition(node):
    if node.type == 'decorated_definition' and node.named_children:
        return node.named_children[-1]
    return node


def _make_symbol_node(symbol_id: str, symbol_type: str, label: str, file_path: str, parent: str | None) -> dict[str, str]:
    node = {
        'id': symbol_id,
        'type': symbol_type,
        'label': label,
        'file': file_path,
    }
    if parent is not None:
        node['parent'] = parent
    return node


def _make_module_node(module_id: str, label: str) -> dict[str, str]:
    return {
        'id': module_id,
        'type': 'module',
        'label': label,
    }


def _make_edge(source: str, target: str, edge_type: str, file_path: str) -> dict[str, str]:
    return {
        'source': source,
        'target': target,
        'type': edge_type,
        'file': file_path,
    }


def _extract_import_targets(import_text: str) -> list[str]:
    if import_text.startswith('import '):
        names = []
        for part in import_text.removeprefix('import ').split(','):
            module_name = part.split(' as ')[0].strip()
            if module_name:
                names.append(module_name)
        return names

    if import_text.startswith('from ') and ' import ' in import_text:
        module_name = import_text.removeprefix('from ').split(' import ', maxsplit=1)[0].strip()
        return [module_name] if module_name else []

    return []


def _extract_call_edges(function_node, source_bytes: bytes, caller_id: str, file_path: str) -> list[dict[str, str]]:
    call_edges: list[dict[str, str]] = []

    caller_class_id: str | None = None
    if caller_id.count('::') >= 2:
        caller_class_id = '::'.join(caller_id.split('::')[:-1])

    for node in _walk_function_body(function_node):
        node = _unwrap_definition(node)
        if node.type != 'call':
            continue

        target_node = node.child_by_field_name('function')
        if target_node is None:
            continue

        target_name: str | None = None
        if target_node.type == 'identifier':
            target_name = _read_text(target_node, source_bytes)
        elif target_node.type == 'attribute':
            attribute_node = target_node.child_by_field_name('attribute')
            object_node = target_node.child_by_field_name('object')
            if attribute_node is None:
                continue
            attribute_name = _read_text(attribute_node, source_bytes)
            if object_node is not None and _read_text(object_node, source_bytes) == 'self' and caller_class_id is not None:
                target_name = f'{caller_class_id}::{attribute_name}'
            else:
                target_name = f'attribute::{attribute_name}'

        if target_name is None:
            continue

        call_edges.append(_make_edge(caller_id, target_name, 'calls', file_path))

    return call_edges


def _sort_nodes(nodes: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(nodes, key=lambda node: (node.get('file', ''), node['type'], node['id']))


def _sort_tree_children(children: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(children, key=lambda child: (str(child['type']), str(child['label']), str(child['id'])))


def parse_python_file(code: str, file_path: str):
    parser = Parser(PY_LANGUAGE)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)
    root = tree.root_node

    nodes: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []
    tree_children: list[dict[str, object]] = []
    import_nodes: dict[str, dict[str, str]] = {}

    for node in root.children:
        node = _unwrap_definition(node)
        if node.type in {'import_statement', 'import_from_statement'}:
            import_text = _read_text(node, source_bytes)
            for module_name in _extract_import_targets(import_text):
                module_id = f'module::{module_name}'
                import_nodes.setdefault(
                    module_id,
                    _make_module_node(module_id, module_name),
                )
                edges.append(_make_edge(file_path, module_id, 'imports', file_path))
            continue

        if node.type == 'class_definition':
            class_name = _get_name(node, source_bytes)
            if class_name is None:
                continue

            class_id = f'{file_path}::{class_name}'
            method_children: list[dict[str, object]] = []

            nodes.append(_make_symbol_node(class_id, 'class', class_name, file_path, file_path))
            edges.append(_make_edge(file_path, class_id, 'contains', file_path))

            body = node.child_by_field_name('body')
            if body is not None:
                for child in body.children:
                    child = _unwrap_definition(child)
                    if child.type != 'function_definition':
                        continue

                    func_name = _get_name(child, source_bytes)
                    if func_name is None:
                        continue

                    func_id = f'{class_id}::{func_name}'
                    nodes.append(_make_symbol_node(func_id, 'function', func_name, file_path, class_id))
                    edges.append(_make_edge(class_id, func_id, 'contains', file_path))
                    edges.extend(_extract_call_edges(child, source_bytes, func_id, file_path))
                    method_children.append({
                        'id': func_id,
                        'type': 'function',
                        'label': func_name,
                        'children': [],
                    })

            tree_children.append({
                'id': class_id,
                'type': 'class',
                'label': class_name,
                'children': _sort_tree_children(method_children),
            })

            superclasses = node.child_by_field_name('superclasses')
            if superclasses is not None:
                for child in superclasses.named_children:
                    parent_name = _read_text(child, source_bytes)
                    edges.append(_make_edge(class_id, parent_name, 'inherits', file_path))
            continue

        if node.type == 'function_definition':
            func_name = _get_name(node, source_bytes)
            if func_name is None:
                continue

            func_id = f'{file_path}::{func_name}'
            nodes.append(_make_symbol_node(func_id, 'function', func_name, file_path, file_path))
            edges.append(_make_edge(file_path, func_id, 'contains', file_path))
            edges.extend(_extract_call_edges(node, source_bytes, func_id, file_path))

            tree_children.append({
                'id': func_id,
                'type': 'function',
                'label': func_name,
                'children': [],
            })

    file_tree_node = {
        'id': file_path,
        'type': 'file',
        'label': file_path.split('/')[-1],
        'children': _sort_tree_children(tree_children),
    }
    file_node = {
        'id': file_path,
        'type': 'file',
        'label': file_path.split('/')[-1],
        'file': file_path,
    }

    sorted_nodes = _sort_nodes([*import_nodes.values(), *nodes])
    return file_tree_node, file_node, sorted_nodes, edges


def resolve_edges(edges, nodes):
    class_ids_by_name = defaultdict(list)
    symbol_ids_by_name = defaultdict(list)
    symbol_ids_by_file_and_name = defaultdict(list)

    for node in nodes:
        label = node['label']
        node_id = node['id']
        file_path = node.get('file', '')
        if node['type'] == 'class':
            class_ids_by_name[label].append(node_id)
        if node['type'] in {'class', 'function'}:
            symbol_ids_by_name[label].append(node_id)
            symbol_ids_by_file_and_name[(file_path, label)].append(node_id)

    resolved: list[dict[str, str]] = []
    for edge in edges:
        target = edge['target']
        if '::' not in target:
            same_file_candidates = symbol_ids_by_file_and_name[(edge['file'], target)]
            global_candidates = symbol_ids_by_name[target]

            if edge['type'] == 'inherits':
                same_file_class_candidates = [candidate for candidate in same_file_candidates if candidate in class_ids_by_name[target]]
                if len(same_file_class_candidates) == 1:
                    target = same_file_class_candidates[0]
                elif len(class_ids_by_name[target]) == 1:
                    target = class_ids_by_name[target][0]
            elif len(same_file_candidates) == 1:
                target = same_file_candidates[0]
            elif len(global_candidates) == 1:
                target = global_candidates[0]

        resolved.append({
            'source': edge['source'],
            'target': target,
            'type': edge['type'],
            'file': edge['file'],
        })

    sorted_edges = sorted(
        resolved,
        key=lambda edge: (edge['file'], edge['type'], edge['source'], edge['target']),
    )

    return [
        {
            'id': f'e{index}',
            **edge,
        }
        for index, edge in enumerate(sorted_edges, start=1)
    ]


def parse_repo(repo_path, files, get_content_func):
    all_tree = []
    all_nodes = []
    all_edges = []

    for file_path in sorted(files):
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

    deduplicated_nodes = list({node['id']: node for node in all_nodes}.values())
    sorted_nodes = _sort_nodes(deduplicated_nodes)
    resolved_edges = resolve_edges(all_edges, sorted_nodes)

    return {
        'tree': sorted(all_tree, key=lambda node: node['id']),
        'nodes': sorted_nodes,
        'edges': resolved_edges,
    }
