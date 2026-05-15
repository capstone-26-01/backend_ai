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


def _make_edge(
    source: str,
    target: str,
    edge_type: str,
    file_path: str,
    *,
    confidence: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        'source': source,
        'target': target,
        'type': edge_type,
        'file': file_path,
        'confidence': confidence,
        'metadata': metadata or {},
    }


def _split_alias(value: str) -> tuple[str, str | None]:
    if ' as ' not in value:
        return value.strip(), None
    name, alias = value.split(' as ', maxsplit=1)
    return name.strip(), alias.strip()


def _normalize_import_module(module_name: str, current_module: str) -> str:
    if not module_name.startswith('.'):
        return module_name

    level = len(module_name) - len(module_name.lstrip('.'))
    tail = module_name[level:]
    current_package = current_module.split('.')[:-1]
    base_length = max(len(current_package) - level + 1, 0)
    parts = current_package[:base_length]
    if tail:
        parts.extend(part for part in tail.split('.') if part)
    return '.'.join(parts)


def _extract_import_details(import_text: str, current_module: str) -> tuple[list[str], dict[str, str], dict[str, str]]:
    module_targets: list[str] = []
    call_aliases: dict[str, str] = {}
    module_aliases: dict[str, str] = {}

    if import_text.startswith('import '):
        for part in import_text.removeprefix('import ').split(','):
            module_name, alias = _split_alias(part)
            if module_name:
                module_targets.append(module_name)
                alias_name = alias or module_name.split('.')[0]
                module_aliases[alias_name] = module_name
        return module_targets, call_aliases, module_aliases

    if import_text.startswith('from ') and ' import ' in import_text:
        module_name, imported_names = import_text.removeprefix('from ').split(' import ', maxsplit=1)
        normalized_module = _normalize_import_module(module_name.strip(), current_module)
        if normalized_module:
            module_targets.append(normalized_module)
        for part in imported_names.split(','):
            imported_name, alias = _split_alias(part)
            if not imported_name or imported_name == '*':
                continue
            call_aliases[alias or imported_name] = imported_name
        return module_targets, call_aliases, module_aliases

    return module_targets, call_aliases, module_aliases


def _extract_call_edges(
    function_node,
    source_bytes: bytes,
    caller_id: str,
    file_path: str,
    *,
    call_aliases: dict[str, str] | None = None,
    module_aliases: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    call_edges: list[dict[str, object]] = []
    call_aliases = call_aliases or {}
    module_aliases = module_aliases or {}

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
        confidence = 0.8
        metadata: dict[str, object] = {}
        if target_node.type == 'identifier':
            identifier = _read_text(target_node, source_bytes)
            target_name = call_aliases.get(identifier, identifier)
            if identifier in call_aliases:
                confidence = 0.9
                metadata['import_alias'] = identifier
        elif target_node.type == 'attribute':
            attribute_node = target_node.child_by_field_name('attribute')
            object_node = target_node.child_by_field_name('object')
            if attribute_node is None:
                continue
            attribute_name = _read_text(attribute_node, source_bytes)
            if object_node is not None and _read_text(object_node, source_bytes) == 'self' and caller_class_id is not None:
                target_name = f'{caller_class_id}::{attribute_name}'
                confidence = 1.0
            else:
                object_name = _read_text(object_node, source_bytes) if object_node is not None else ''
                if object_name in module_aliases:
                    target_name = attribute_name
                    confidence = 0.75
                    metadata['module_alias'] = object_name
                    metadata['module'] = module_aliases[object_name]
                else:
                    target_name = f'attribute::{attribute_name}'
                    confidence = 0.4
                    metadata['unresolved_attribute'] = object_name

        if target_name is None:
            continue

        call_edges.append(_make_edge(caller_id, target_name, 'calls', file_path, confidence=confidence, metadata=metadata))

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


def _parse_class_body(
    class_node,
    source_bytes: bytes,
    class_id: str,
    file_path: str,
    *,
    call_aliases: dict[str, str],
    module_aliases: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
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
        edges.extend(_extract_call_edges(child, source_bytes, func_id, file_path, call_aliases=call_aliases, module_aliases=module_aliases))
        tree_children.append(_make_tree_node(func_id, 'method', func_name))

    return nodes, edges, tree_children


def parse_python_file(code: str, file_path: str):
    parser = Parser(PY_LANGUAGE)
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)
    root = tree.root_node

    module_name = _module_name_from_path(file_path)
    module_id = f'module::{module_name}'
    module_node = _make_module_node(module_id, module_name, file_path, file_path)
    if '__name__' in code and '__main__' in code:
        cast_metadata = module_node['metadata']
        if isinstance(cast_metadata, dict):
            cast_metadata['has_main_guard'] = True
    nodes: list[dict[str, object]] = [module_node]
    edges: list[dict[str, object]] = [_make_edge(file_path, module_id, 'contains', file_path)]
    module_children: list[dict[str, object]] = []
    import_nodes: dict[str, dict[str, object]] = {}
    warnings: list[dict[str, object]] = []
    call_aliases: dict[str, str] = {}
    module_aliases: dict[str, str] = {}

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
            imported_modules, imported_call_aliases, imported_module_aliases = _extract_import_details(import_text, module_name)
            call_aliases.update(imported_call_aliases)
            module_aliases.update(imported_module_aliases)
            for imported_module in imported_modules:
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
            method_nodes, method_edges, method_tree_children = _parse_class_body(
                node,
                source_bytes,
                class_id,
                file_path,
                call_aliases=call_aliases,
                module_aliases=module_aliases,
            )

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
            edges.extend(_extract_call_edges(node, source_bytes, func_id, file_path, call_aliases=call_aliases, module_aliases=module_aliases))
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
            'confidence': edge.get('confidence', 1.0),
            'metadata': edge.get('metadata', {}),
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


def _external_label(external_id: str) -> str:
    return external_id.split('::')[-1]


def _attach_external_call_nodes(nodes: list[dict[str, object]], edges: list[dict[str, object]], warnings: list[dict[str, object]]) -> list[dict[str, object]]:
    node_ids = {str(node['id']) for node in nodes}
    external_nodes: dict[str, dict[str, object]] = {}

    for edge in edges:
        if edge['type'] != 'calls':
            continue
        target = str(edge['target'])
        if target in node_ids:
            continue

        external_id = target if '::' in target else f'external::{target}'
        edge['target'] = external_id
        edge['confidence'] = min(float(edge.get('confidence', 1.0)), 0.4)
        metadata = dict(edge.get('metadata', {}))
        metadata['unresolved_target'] = target
        edge['metadata'] = metadata

        if external_id not in external_nodes:
            external_nodes[external_id] = _make_graph_node(
                external_id,
                'external',
                _external_label(external_id),
                metadata={'original_target': target},
            )
            warnings.append({
                'code': 'unresolved_call',
                'message': 'Call target could not be resolved to a local symbol.',
                'path': edge['file'],
                'target': target,
            })

    return [*nodes, *external_nodes.values()]


def _detect_entrypoints(nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    entrypoints: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(node: dict[str, object], kind: str, confidence: float, reason: str) -> None:
        node_id = str(node['id'])
        key = f'{node_id}:{kind}'
        if key in seen:
            return
        seen.add(key)
        entrypoints.append({
            'id': node_id,
            'kind': kind,
            'label': node['label'],
            'path': node.get('file') or node.get('path'),
            'confidence': confidence,
            'reason': reason,
        })

    for node in nodes:
        node_type = node.get('type')
        label = str(node.get('label', ''))
        path = str(node.get('file') or node.get('path') or '')
        metadata = node.get('metadata') if isinstance(node.get('metadata'), dict) else {}

        if node_type == 'module' and metadata.get('has_main_guard'):
            add(node, 'python_main_guard', 1.0, '__name__ == "__main__" guard')
        if node_type in {'function', 'method'} and label == 'main':
            add(node, 'main_function', 0.9, 'function named main')
        if path.endswith('manage.py'):
            add(node, 'django_manage', 0.95, 'manage.py convention')
        if path.endswith(('asgi.py', 'wsgi.py')):
            add(node, 'django_server_entry', 0.8, 'ASGI/WSGI convention')
        decorators = metadata.get('decorators') if isinstance(metadata, dict) else None
        if isinstance(decorators, list) and any(str(decorator).startswith(('app.get', 'app.post', 'app.put', 'app.delete', 'router.', 'app.route')) for decorator in decorators):
            add(node, 'web_route', 0.85, 'route decorator')

    return sorted(entrypoints, key=lambda entrypoint: (-float(entrypoint['confidence']), str(entrypoint['path']), str(entrypoint['id'])))


def _score_key_modules(nodes: list[dict[str, object]], edges: list[dict[str, object]], entrypoints: list[dict[str, object]]) -> list[dict[str, object]]:
    module_nodes = {str(node['id']): node for node in nodes if node.get('type') == 'module' and node.get('file')}
    if not module_nodes:
        return []

    fan_in = defaultdict(int)
    fan_out = defaultdict(int)
    parent_by_id = {str(node['id']): str(node.get('parent', '')) for node in nodes}
    entrypoint_paths = {str(entrypoint.get('path')) for entrypoint in entrypoints if entrypoint.get('path')}

    for edge in edges:
        source_module = str(edge['source']) if str(edge['source']) in module_nodes else parent_by_id.get(str(edge['source']), '')
        target_module = str(edge['target']) if str(edge['target']) in module_nodes else parent_by_id.get(str(edge['target']), '')
        if source_module in module_nodes:
            fan_out[source_module] += 1
        if target_module in module_nodes:
            fan_in[target_module] += 1

    hints = {'service', 'api', 'view', 'views', 'model', 'models', 'router', 'parser', 'config', 'settings', 'main', 'manage'}
    scored: list[dict[str, object]] = []
    for module_id, node in module_nodes.items():
        path = str(node.get('file') or '')
        path_tokens = set(path.removesuffix('.py').replace('/', '.').replace('_', '.').split('.'))
        score = fan_in[module_id] * 2 + fan_out[module_id]
        reasons = []
        if fan_in[module_id]:
            reasons.append('fan_in')
        if fan_out[module_id]:
            reasons.append('fan_out')
        if path in entrypoint_paths:
            score += 5
            reasons.append('entrypoint')
        if path_tokens & hints:
            score += 2
            reasons.append('name_hint')
        if score <= 0:
            continue
        scored.append({
            'id': module_id,
            'label': node['label'],
            'path': path,
            'score': score,
            'reasons': reasons,
        })

    return sorted(scored, key=lambda item: (-int(item['score']), str(item['path']), str(item['id'])))[:10]


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
    nodes_with_external = _sort_nodes(_attach_external_call_nodes(sorted_nodes, resolved_edges, warnings))
    entrypoints = _detect_entrypoints(nodes_with_external)
    key_modules = _score_key_modules(nodes_with_external, resolved_edges, entrypoints)

    for entrypoint in entrypoints:
        path = str(entrypoint.get('path') or '')
        target = str(entrypoint['id'])
        if path and path != target:
            resolved_edges.append(_make_edge(path, target, 'entrypoint', path, confidence=float(entrypoint['confidence'])))
    resolved_edges = resolve_edges(resolved_edges, nodes_with_external)

    return {
        'tree': _sort_tree_children(root_tree_children),
        'nodes': nodes_with_external,
        'edges': resolved_edges,
        'entrypoints': entrypoints,
        'key_modules': key_modules,
        'warnings': warnings,
    }
