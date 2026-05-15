from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath
import ast

from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())

TREE_TYPE_ORDER = {
    'directory': 0,
    'file': 1,
    'module': 2,
    'class': 3,
    'function': 4,
    'method': 5,
}


def _read_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode('utf-8')


def _point_row(point) -> int:
    return point.row if hasattr(point, 'row') else point[0]


def _line_range(node) -> tuple[int, int]:
    return _point_row(node.start_point) + 1, _point_row(node.end_point) + 1


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


def _unwrap_definition_with_decorators(node, source_bytes: bytes):
    if node.type != 'decorated_definition':
        return node, []

    decorators = []
    for child in node.children:
        if child.type != 'decorator':
            continue
        decorator = _read_text(child, source_bytes).strip()
        decorators.append(decorator.removeprefix('@').strip())
    return _unwrap_definition(node), decorators


def _first_sentence(value: str) -> str:
    normalized = ' '.join(value.strip().split())
    if not normalized:
        return ''
    for separator in ('. ', '? ', '! '):
        if separator in normalized:
            return normalized.split(separator, maxsplit=1)[0] + separator.strip()
    return normalized[:160]


def _extract_docstring(node, source_bytes: bytes) -> str | None:
    body = node.child_by_field_name('body')
    if body is None or not body.named_children:
        return None

    first_statement = body.named_children[0]
    if first_statement.type != 'expression_statement' or not first_statement.named_children:
        return None

    literal_node = first_statement.named_children[0]
    if literal_node.type not in {'string', 'concatenated_string'}:
        return None

    try:
        value = ast.literal_eval(_read_text(literal_node, source_bytes))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, str):
        return None
    return _first_sentence(value)


def _module_name_from_path(file_path: str) -> str:
    parts = file_path.removesuffix('.py').split('/')
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts) if parts else '__init__'


def _make_tree_node(node_id: str, node_type: str, label: str, children: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        'id': node_id,
        'type': node_type,
        'label': label,
        'children': children or [],
    }


def _make_graph_node(
    node_id: str,
    node_type: str,
    label: str,
    *,
    file_path: str | None = None,
    path: str | None = None,
    parent: str | None = None,
    symbol: str | None = None,
    language: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    node: dict[str, object] = {
        'id': node_id,
        'type': node_type,
        'label': label,
        'metadata': metadata or {},
    }
    if file_path is not None:
        node['file'] = file_path
    if path is not None:
        node['path'] = path
    if parent is not None:
        node['parent'] = parent
    if symbol is not None:
        node['symbol'] = symbol
    if language is not None:
        node['language'] = language
    if start_line is not None:
        node['start_line'] = start_line
    if end_line is not None:
        node['end_line'] = end_line
    return node


def _make_module_node(module_id: str, label: str, file_path: str | None = None, parent: str | None = None) -> dict[str, object]:
    return _make_graph_node(
        module_id,
        'module',
        label,
        file_path=file_path,
        path=file_path,
        parent=parent,
        symbol=label,
        language='python' if file_path else None,
        metadata={'external': file_path is None},
    )


def _make_edge(source: str, target: str, edge_type: str, file_path: str) -> dict[str, object]:
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


def _extract_call_edges(function_node, source_bytes: bytes, caller_id: str, file_path: str) -> list[dict[str, object]]:
    call_edges: list[dict[str, object]] = []

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


def _sort_nodes(nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(nodes, key=lambda node: (str(node.get('file') or node.get('path') or ''), str(node['type']), str(node['id'])))


def _sort_tree_children(children: list[dict[str, object]]) -> list[dict[str, object]]:
    sorted_children = sorted(
        children,
        key=lambda child: (TREE_TYPE_ORDER.get(str(child['type']), 99), str(child['label']), str(child['id'])),
    )
    for child in sorted_children:
        child['children'] = _sort_tree_children(list(child.get('children', [])))
    return sorted_children


def _symbol_metadata(decorators: list[str], docstring: str | None) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if decorators:
        metadata['decorators'] = decorators
    if docstring:
        metadata['docstring'] = docstring
    return metadata


def _parse_class_body(class_node, source_bytes: bytes, class_id: str, file_path: str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    tree_children: list[dict[str, object]] = []
    body = class_node.child_by_field_name('body')
    if body is None:
        return nodes, edges, tree_children

    for child in body.children:
        child, decorators = _unwrap_definition_with_decorators(child, source_bytes)
        if child.type != 'function_definition':
            continue

        func_name = _get_name(child, source_bytes)
        if func_name is None:
            continue

        func_id = f'{class_id}::{func_name}'
        start_line, end_line = _line_range(child)
        nodes.append(_make_graph_node(
            func_id,
            'method',
            func_name,
            file_path=file_path,
            path=file_path,
            parent=class_id,
            symbol=func_name,
            language='python',
            start_line=start_line,
            end_line=end_line,
            metadata=_symbol_metadata(decorators, _extract_docstring(child, source_bytes)),
        ))
        edges.append(_make_edge(class_id, func_id, 'contains', file_path))
        edges.extend(_extract_call_edges(child, source_bytes, func_id, file_path))
        tree_children.append(_make_tree_node(func_id, 'method', func_name))

    return nodes, edges, tree_children


def parse_python_file(code: str, file_path: str):
    parser = Parser(PY_LANGUAGE)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)
    root = tree.root_node

    module_name = _module_name_from_path(file_path)
    module_id = f'module::{module_name}'
    nodes: list[dict[str, object]] = [_make_module_node(module_id, module_name, file_path, file_path)]
    edges: list[dict[str, object]] = [_make_edge(file_path, module_id, 'contains', file_path)]
    module_children: list[dict[str, object]] = []
    import_nodes: dict[str, dict[str, object]] = {}
    warnings: list[dict[str, object]] = []

    if root.has_error:
        warnings.append({
            'code': 'syntax_error',
            'message': 'Python syntax error; symbol extraction skipped.',
            'path': file_path,
        })
        return _make_tree_node(module_id, 'module', module_name), nodes, edges, warnings

    for raw_node in root.children:
        node, decorators = _unwrap_definition_with_decorators(raw_node, source_bytes)
        if node.type in {'import_statement', 'import_from_statement'}:
            import_text = _read_text(node, source_bytes)
            for imported_module in _extract_import_targets(import_text):
                imported_module_id = f'module::{imported_module}'
                import_nodes.setdefault(
                    imported_module_id,
                    _make_module_node(imported_module_id, imported_module),
                )
                edges.append(_make_edge(file_path, imported_module_id, 'imports', file_path))
            continue

        if node.type == 'class_definition':
            class_name = _get_name(node, source_bytes)
            if class_name is None:
                continue

            class_id = f'{file_path}::{class_name}'
            start_line, end_line = _line_range(node)
            method_nodes, method_edges, method_tree_children = _parse_class_body(node, source_bytes, class_id, file_path)

            nodes.append(_make_graph_node(
                class_id,
                'class',
                class_name,
                file_path=file_path,
                path=file_path,
                parent=module_id,
                symbol=class_name,
                language='python',
                start_line=start_line,
                end_line=end_line,
                metadata=_symbol_metadata(decorators, _extract_docstring(node, source_bytes)),
            ))
            nodes.extend(method_nodes)
            edges.append(_make_edge(module_id, class_id, 'contains', file_path))
            edges.extend(method_edges)
            module_children.append(_make_tree_node(class_id, 'class', class_name, _sort_tree_children(method_tree_children)))

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
            start_line, end_line = _line_range(node)
            nodes.append(_make_graph_node(
                func_id,
                'function',
                func_name,
                file_path=file_path,
                path=file_path,
                parent=module_id,
                symbol=func_name,
                language='python',
                start_line=start_line,
                end_line=end_line,
                metadata=_symbol_metadata(decorators, _extract_docstring(node, source_bytes)),
            ))
            edges.append(_make_edge(module_id, func_id, 'contains', file_path))
            edges.extend(_extract_call_edges(node, source_bytes, func_id, file_path))
            module_children.append(_make_tree_node(func_id, 'function', func_name))

    module_tree_node = _make_tree_node(module_id, 'module', module_name, _sort_tree_children(module_children))
    return module_tree_node, [*import_nodes.values(), *nodes], edges, warnings


def resolve_edges(edges, nodes):
    class_ids_by_name = defaultdict(list)
    symbol_ids_by_name = defaultdict(list)
    symbol_ids_by_file_and_name = defaultdict(list)

    for node in nodes:
        label = str(node['label'])
        node_id = str(node['id'])
        file_path = str(node.get('file', ''))
        if node['type'] == 'class':
            class_ids_by_name[label].append(node_id)
        if node['type'] in {'class', 'function', 'method'}:
            symbol_ids_by_name[label].append(node_id)
            symbol_ids_by_file_and_name[(file_path, label)].append(node_id)

    resolved: list[dict[str, object]] = []
    for edge in edges:
        target = str(edge['target'])
        if '::' not in target:
            same_file_candidates = symbol_ids_by_file_and_name[(str(edge['file']), target)]
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
        key=lambda edge: (str(edge['file']), str(edge['type']), str(edge['source']), str(edge['target'])),
    )

    return [
        {
            'id': f'e{index}',
            **edge,
        }
        for index, edge in enumerate(sorted_edges, start=1)
    ]


def _normalize_repo_path(file_path: str) -> str | None:
    path = PurePosixPath(file_path)
    if path.is_absolute() or '..' in path.parts:
        return None
    return path.as_posix()


def _make_file_node(file_path: str, parent: str | None, *, analyzed: bool, language: str | None) -> dict[str, object]:
    return _make_graph_node(
        file_path,
        'file',
        file_path.split('/')[-1],
        file_path=file_path,
        path=file_path,
        parent=parent,
        language=language,
        metadata={
            'analyzed': analyzed,
            'language': language,
            'unsupported': language is None,
        },
    )


def _deduplicate_nodes(nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    deduplicated: dict[str, dict[str, object]] = {}
    for node in nodes:
        node_id = str(node['id'])
        existing = deduplicated.get(node_id)
        if existing is None or (not existing.get('file') and node.get('file')):
            deduplicated[node_id] = node
    return list(deduplicated.values())


def parse_repo(repo_path, files, get_content_func):
    root_tree_children: list[dict[str, object]] = []
    directory_tree_nodes: dict[str, dict[str, object]] = {}
    all_nodes: list[dict[str, object]] = []
    all_edges: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    def ensure_directory(dir_path: str, label: str, parent_path: str | None, siblings: list[dict[str, object]]) -> dict[str, object]:
        existing = directory_tree_nodes.get(dir_path)
        if existing is not None:
            return existing

        tree_node = _make_tree_node(dir_path, 'directory', label)
        directory_tree_nodes[dir_path] = tree_node
        siblings.append(tree_node)
        all_nodes.append(_make_graph_node(dir_path, 'directory', label, path=dir_path, parent=parent_path))
        if parent_path is not None:
            all_edges.append(_make_edge(parent_path, dir_path, 'contains', dir_path))
        return tree_node

    for raw_file_path in sorted(files):
        file_path = _normalize_repo_path(raw_file_path)
        if file_path is None:
            warnings.append({
                'code': 'unsafe_path',
                'message': 'Unsafe repository path skipped.',
                'path': raw_file_path,
            })
            continue

        parts = file_path.split('/')
        parent_path: str | None = None
        siblings = root_tree_children
        for index, directory_name in enumerate(parts[:-1]):
            dir_path = '/'.join(parts[:index + 1])
            directory_node = ensure_directory(dir_path, directory_name, parent_path, siblings)
            parent_path = dir_path
            siblings = directory_node['children']

        is_python = file_path.endswith('.py')
        language = 'python' if is_python else None
        file_tree_node = _make_tree_node(file_path, 'file', parts[-1])
        analyzed = False

        if is_python:
            code = get_content_func(repo_path, file_path)
            if code is None:
                warnings.append({
                    'code': 'missing_content',
                    'message': 'Python file content was not available for analysis.',
                    'path': file_path,
                })
            else:
                module_tree_node, nodes, edges, file_warnings = parse_python_file(code, file_path)
                file_tree_node['children'].append(module_tree_node)
                all_nodes.extend(nodes)
                all_edges.extend(edges)
                warnings.extend(file_warnings)
                analyzed = not any(warning.get('code') == 'syntax_error' for warning in file_warnings)

        siblings.append(file_tree_node)
        all_nodes.append(_make_file_node(file_path, parent_path, analyzed=analyzed, language=language))
        if parent_path is not None:
            all_edges.append(_make_edge(parent_path, file_path, 'contains', file_path))

    deduplicated_nodes = _deduplicate_nodes(all_nodes)
    sorted_nodes = _sort_nodes(deduplicated_nodes)
    resolved_edges = resolve_edges(all_edges, sorted_nodes)

    return {
        'tree': _sort_tree_children(root_tree_children),
        'nodes': sorted_nodes,
        'edges': resolved_edges,
        'warnings': warnings,
    }
