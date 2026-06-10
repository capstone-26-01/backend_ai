from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath
import ast

from tree_sitter import Language, Parser
import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript

from parser.language_registry import JAVASCRIPT_BUILTINS, ignored_reason, language_for_path, normalize_enabled_languages
from parser.languages.javascript_resolver import resolve_js_module_path

PY_LANGUAGE = Language(tspython.language())
JS_LANGUAGE = Language(tsjavascript.language())
TS_LANGUAGE = Language(tstypescript.language_typescript())
TSX_LANGUAGE = Language(tstypescript.language_tsx())

JS_TS_DECLARATION_TYPES = {
    'arrow_function',
    'class',
    'class_declaration',
    'abstract_class_declaration',
    'enum_declaration',
    'function_expression',
    'function_declaration',
    'interface_declaration',
    'lexical_declaration',
    'type_alias_declaration',
}
JS_TS_FUNCTION_VALUE_TYPES = {'arrow_function', 'function', 'function_expression'}
JS_TS_NESTED_CALL_BOUNDARIES = {
    'arrow_function',
    'class_declaration',
    'function',
    'function_declaration',
    'function_expression',
    'method_definition',
}
JS_TS_SOURCE_EXTENSIONS = ('.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.mts', '.cts')
JS_TS_MANIFEST_NAMES = {'tsconfig.json', 'jsconfig.json', 'pnpm-workspace.yaml', 'package.json'}
JS_NAMESPACE_TARGET_PREFIX = 'namespace::'
JS_TS_TYPE_REFERENCE_BUILTINS = set(JAVASCRIPT_BUILTINS) | {
    'Array',
    'Awaited',
    'NonNullable',
    'Omit',
    'Partial',
    'Pick',
    'Readonly',
    'Record',
    'Required',
    'ReturnType',
    'boolean',
    'bigint',
    'never',
    'null',
    'number',
    'object',
    'string',
    'symbol',
    'undefined',
    'unknown',
    'void',
}

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


def _js_language_for_path(file_path: str):
    if file_path.endswith('.tsx'):
        return TSX_LANGUAGE
    if file_path.endswith(('.ts', '.mts', '.cts')):
        return TS_LANGUAGE
    return JS_LANGUAGE


def _js_language_id_for_path(file_path: str) -> str:
    return 'typescript' if file_path.endswith(('.ts', '.tsx', '.mts', '.cts')) else 'javascript'


def _module_id_for_source_path(file_path: str) -> str:
    return f'module::{file_path}'


def _clean_js_string_literal(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in {'"', "'", '`'} and value[-1] == value[0]:
        return value[1:-1]
    return value


def _js_namespace_target(object_name: str, member_name: str) -> str:
    return f'{JS_NAMESPACE_TARGET_PREFIX}{object_name}::{member_name}'


def _split_js_namespace_target(target: str) -> tuple[str, str] | None:
    if not target.startswith(JS_NAMESPACE_TARGET_PREFIX):
        return None
    parts = target.removeprefix(JS_NAMESPACE_TARGET_PREFIX).split('::', 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _js_node_name(node, source_bytes: bytes) -> str | None:
    name_node = node.child_by_field_name('name')
    if name_node is not None:
        return _read_text(name_node, source_bytes)
    for child in node.named_children:
        if child.type in {'identifier', 'property_identifier', 'type_identifier'}:
            return _read_text(child, source_bytes)
    return None


def _unwrap_js_export(node):
    if node.type != 'export_statement':
        return node, False
    for child in node.named_children:
        if child.type in JS_TS_DECLARATION_TYPES:
            return child, True
    return node, True


def _js_is_default_export(node, source_bytes: bytes) -> bool:
    return node.type == 'export_statement' and _read_text(node, source_bytes).lstrip().startswith('export default')


def _js_is_export_all(node) -> bool:
    return node.type == 'export_statement' and not any(child.type in {'export_clause', 'namespace_export'} for child in node.named_children)


def _js_import_source(node, source_bytes: bytes) -> str | None:
    source_node = node.child_by_field_name('source')
    if source_node is not None:
        return _clean_js_string_literal(_read_text(source_node, source_bytes))
    for child in reversed(node.named_children):
        if child.type == 'string':
            return _clean_js_string_literal(_read_text(child, source_bytes))
    return None


def _js_import_aliases(node, source_bytes: bytes) -> dict[str, str]:
    aliases: dict[str, str] = {}
    import_clause = next((child for child in node.named_children if child.type == 'import_clause'), None)
    if import_clause is not None:
        for child in import_clause.named_children:
            if child.type == 'identifier':
                aliases[_read_text(child, source_bytes)] = 'default'
                continue
            if child.type == 'namespace_import':
                name = _js_node_name(child, source_bytes)
                if name:
                    aliases[name] = '*'
                continue
            if child.type != 'named_imports':
                continue
            for specifier in child.named_children:
                if specifier.type != 'import_specifier':
                    continue
                name_node = specifier.child_by_field_name('name')
                alias_node = specifier.child_by_field_name('alias')
                if name_node is None:
                    continue
                imported = _read_text(name_node, source_bytes)
                local = _read_text(alias_node, source_bytes) if alias_node is not None else imported
                aliases[local] = imported

    export_clause = next((child for child in node.named_children if child.type == 'export_clause'), None)
    if export_clause is not None:
        for specifier in export_clause.named_children:
            if specifier.type != 'export_specifier':
                continue
            name_node = specifier.child_by_field_name('name')
            alias_node = specifier.child_by_field_name('alias')
            if name_node is None:
                continue
            imported = _read_text(name_node, source_bytes)
            local = _read_text(alias_node, source_bytes) if alias_node is not None else imported
            aliases[local] = imported
    return aliases


def _js_variable_function_name(node, source_bytes: bytes):
    if node.type != 'lexical_declaration':
        return None
    for child in node.named_children:
        if child.type != 'variable_declarator':
            continue
        name_node = child.child_by_field_name('name')
        value_node = child.child_by_field_name('value')
        if name_node is None or value_node is None or value_node.type not in JS_TS_FUNCTION_VALUE_TYPES:
            continue
        return _read_text(name_node, source_bytes), value_node.type, value_node
    return None


def _first_js_expression_identifier(node, source_bytes: bytes) -> str | None:
    if node.type in {'identifier', 'property_identifier', 'type_identifier'}:
        return _read_text(node, source_bytes)
    if node.type == 'member_expression':
        object_node = node.child_by_field_name('object')
        property_node = node.child_by_field_name('property')
        if property_node is None:
            return None
        property_name = _read_text(property_node, source_bytes)
        if object_node is not None and object_node.type == 'identifier':
            return _js_namespace_target(_read_text(object_node, source_bytes), property_name)
        return property_name
    for child in node.named_children:
        value = _first_js_expression_identifier(child, source_bytes)
        if value:
            return value
    return None


def _js_heritage_edges(node, source_bytes: bytes, source_id: str, file_path: str) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for child in _walk(node):
        if child.type not in {'extends_clause', 'extends_type_clause', 'implements_clause'}:
            continue
        relation = 'implements' if child.type == 'implements_clause' else 'extends'
        for candidate in child.named_children:
            name = _first_js_expression_identifier(candidate, source_bytes)
            if not name:
                continue
            edges.append(_make_edge(
                source_id,
                name,
                'inherits',
                file_path,
                confidence=0.75 if relation == 'implements' else 0.85,
                metadata={'ts_relation': relation},
            ))
    return edges


def _walk_js_callable_body(node, *, is_root: bool = True):
    yield node
    for child in node.named_children:
        if not is_root and child.type in JS_TS_NESTED_CALL_BOUNDARIES:
            continue
        yield from _walk_js_callable_body(child, is_root=False)


def _js_call_target(node, source_bytes: bytes, caller_class_id: str | None) -> tuple[str | None, float, dict[str, object]]:
    metadata: dict[str, object] = {}
    if node.type == 'call_expression':
        function_node = node.child_by_field_name('function')
    elif node.type == 'new_expression':
        function_node = node.child_by_field_name('constructor') or (node.named_children[0] if node.named_children else None)
        metadata['constructs'] = True
    else:
        return None, 0.0, metadata
    if function_node is None:
        return None, 0.0, metadata

    if function_node.type == 'identifier':
        identifier = _read_text(function_node, source_bytes)
        if identifier in JAVASCRIPT_BUILTINS:
            return None, 0.0, metadata
        return identifier, 0.65, metadata

    if function_node.type == 'member_expression':
        object_node = function_node.child_by_field_name('object')
        property_node = function_node.child_by_field_name('property')
        if property_node is None:
            return None, 0.0, metadata
        property_name = _read_text(property_node, source_bytes)
        object_name = _read_text(object_node, source_bytes) if object_node is not None else ''
        if object_name in JAVASCRIPT_BUILTINS:
            return None, 0.0, metadata
        if object_name in {'this', 'super'} and caller_class_id is not None:
            return f'{caller_class_id}::{property_name}', 0.9, {'object': object_name}
        if property_name in JAVASCRIPT_BUILTINS:
            return None, 0.0, metadata
        if object_node is not None and object_node.type == 'identifier':
            return _js_namespace_target(object_name, property_name), 0.45, {'member_object': object_name, 'member_name': property_name}
        return f'attribute::{property_name}', 0.35, {'unresolved_attribute': object_name}

    return None, 0.0, metadata


def _extract_js_call_edges(node, source_bytes: bytes, caller_id: str, file_path: str) -> list[dict[str, object]]:
    call_edges: list[dict[str, object]] = []
    caller_class_id = '::'.join(caller_id.split('::')[:-1]) if caller_id.count('::') >= 2 else None
    for child in _walk_js_callable_body(node):
        if child.type not in {'call_expression', 'new_expression'}:
            continue
        target, confidence, metadata = _js_call_target(child, source_bytes, caller_class_id)
        if target is None:
            continue
        call_edges.append(_make_edge(caller_id, target, 'calls', file_path, confidence=confidence, metadata=metadata))
    return call_edges


def _extract_js_reference_edges(
    node,
    source_bytes: bytes,
    source_id: str,
    file_path: str,
    *,
    own_symbol: str | None = None,
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    seen: set[str] = set()
    for child in _walk(node):
        if child.type != 'type_identifier':
            continue
        target = _read_text(child, source_bytes)
        if target == own_symbol or target in JS_TS_TYPE_REFERENCE_BUILTINS or target in seen:
            continue
        seen.add(target)
        edges.append(_make_edge(
            source_id,
            target,
            'references',
            file_path,
            confidence=0.75,
            metadata={'reference_kind': 'type'},
        ))
    return edges


def _parse_js_class_body(class_node, source_bytes: bytes, class_id: str, file_path: str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    tree_children: list[dict[str, object]] = []
    body = class_node.child_by_field_name('body')
    if body is None:
        return nodes, edges, tree_children

    language = _js_language_id_for_path(file_path)
    for child in body.named_children:
        if child.type != 'method_definition':
            continue
        method_name = _js_node_name(child, source_bytes)
        if method_name is None:
            continue
        method_id = f'{class_id}::{method_name}'
        start_line, end_line = _line_range(child)
        nodes.append(_make_graph_node(
            method_id,
            'method',
            method_name,
            file_path=file_path,
            path=file_path,
            parent=class_id,
            symbol=method_name,
            language=language,
            start_line=start_line,
            end_line=end_line,
            metadata={'symbol_kind': 'method'},
        ))
        edges.append(_make_edge(class_id, method_id, 'contains', file_path))
        edges.extend(_extract_js_call_edges(child, source_bytes, method_id, file_path))
        edges.extend(_extract_js_reference_edges(child, source_bytes, method_id, file_path, own_symbol=method_name))
        tree_children.append(_make_tree_node(method_id, 'method', method_name))
    return nodes, edges, tree_children


def parse_javascript_file(code: str, file_path: str):
    parser = Parser(_js_language_for_path(file_path))
    source_bytes = code.encode('utf-8')
    tree = parser.parse(source_bytes)
    root = tree.root_node

    language = _js_language_id_for_path(file_path)
    module_id = _module_id_for_source_path(file_path)
    module_node = _make_graph_node(
        module_id,
        'module',
        file_path,
        file_path=file_path,
        path=file_path,
        parent=file_path,
        symbol=file_path,
        language=language,
        metadata={'external': False, 'support_level': 'relationships'},
    )
    nodes: list[dict[str, object]] = [module_node]
    edges: list[dict[str, object]] = [_make_edge(file_path, module_id, 'contains', file_path)]
    module_children: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    if root.has_error:
        warnings.append({
            'code': 'syntax_error',
            'message': 'JavaScript/TypeScript syntax error; symbol extraction may be incomplete.',
            'path': file_path,
        })

    for raw_node in root.children:
        default_export = _js_is_default_export(raw_node, source_bytes)
        node, exported = _unwrap_js_export(raw_node)
        if node.type in {'import_statement', 'export_statement'}:
            source = _js_import_source(node, source_bytes)
            if source:
                import_aliases = _js_import_aliases(node, source_bytes)
                target_module_id = f'module::{source}'
                edges.append(_make_edge(
                    file_path,
                    target_module_id,
                    'imports',
                    file_path,
                    confidence=0.55 if node.type == 'export_statement' else 0.5,
                    metadata={
                        'import_specifier': source,
                        'import_aliases': import_aliases,
                        'language_family': 'javascript',
                        're_export': node.type == 'export_statement',
                        're_export_all': _js_is_export_all(node),
                    },
                ))
            if node.type == 'export_statement':
                node, exported = _unwrap_js_export(raw_node)
                if node.type == 'export_statement':
                    continue
            else:
                continue

        if node.type in {'class', 'class_declaration', 'abstract_class_declaration', 'interface_declaration', 'type_alias_declaration', 'enum_declaration'}:
            symbol_name = _js_node_name(node, source_bytes)
            if symbol_name is None and default_export:
                symbol_name = 'default'
            if symbol_name is None:
                continue
            symbol_id = f'{file_path}::{symbol_name}'
            start_line, end_line = _line_range(node)
            node_kind = 'class'
            metadata = {
                'symbol_kind': 'class' if node.type in {'class', 'class_declaration', 'abstract_class_declaration'} else node.type.removesuffix('_declaration'),
                'js_symbol_kind': node.type,
                'exported': exported,
                'default_export': default_export,
            }
            method_nodes, method_edges, method_tree_children = _parse_js_class_body(node, source_bytes, symbol_id, file_path)
            nodes.append(_make_graph_node(
                symbol_id,
                node_kind,
                symbol_name,
                file_path=file_path,
                path=file_path,
                parent=module_id,
                symbol=symbol_name,
                language=language,
                start_line=start_line,
                end_line=end_line,
                metadata=metadata,
            ))
            nodes.extend(method_nodes)
            edges.append(_make_edge(module_id, symbol_id, 'contains', file_path))
            edges.extend(method_edges)
            edges.extend(_js_heritage_edges(node, source_bytes, symbol_id, file_path))
            edges.extend(_extract_js_reference_edges(node, source_bytes, symbol_id, file_path, own_symbol=symbol_name))
            module_children.append(_make_tree_node(symbol_id, node_kind, symbol_name, _sort_tree_children(method_tree_children)))
            continue

        if node.type in {'arrow_function', 'function_declaration', 'function_expression'}:
            symbol_name = _js_node_name(node, source_bytes)
            if symbol_name is None:
                symbol_name = 'default' if default_export else None
            if symbol_name is None:
                continue
            symbol_id = f'{file_path}::{symbol_name}'
            start_line, end_line = _line_range(node)
            nodes.append(_make_graph_node(
                symbol_id,
                'function',
                symbol_name,
                file_path=file_path,
                path=file_path,
                parent=module_id,
                symbol=symbol_name,
                language=language,
                start_line=start_line,
                end_line=end_line,
                metadata={'symbol_kind': 'function', 'exported': exported, 'default_export': default_export},
            ))
            edges.append(_make_edge(module_id, symbol_id, 'contains', file_path))
            edges.extend(_extract_js_call_edges(node, source_bytes, symbol_id, file_path))
            edges.extend(_extract_js_reference_edges(node, source_bytes, symbol_id, file_path, own_symbol=symbol_name))
            module_children.append(_make_tree_node(symbol_id, 'function', symbol_name))
            continue

        variable_function = _js_variable_function_name(node, source_bytes)
        if variable_function is not None:
            symbol_name, value_type, value_node = variable_function
            symbol_id = f'{file_path}::{symbol_name}'
            start_line, end_line = _line_range(node)
            nodes.append(_make_graph_node(
                symbol_id,
                'function',
                symbol_name,
                file_path=file_path,
                path=file_path,
                parent=module_id,
                symbol=symbol_name,
                language=language,
                start_line=start_line,
                end_line=end_line,
                metadata={'symbol_kind': value_type, 'exported': exported},
            ))
            edges.append(_make_edge(module_id, symbol_id, 'contains', file_path))
            edges.extend(_extract_js_call_edges(value_node, source_bytes, symbol_id, file_path))
            edges.extend(_extract_js_reference_edges(node, source_bytes, symbol_id, file_path, own_symbol=symbol_name))
            module_children.append(_make_tree_node(symbol_id, 'function', symbol_name))

    module_tree_node = _make_tree_node(module_id, 'module', file_path, _sort_tree_children(module_children))
    return module_tree_node, nodes, edges, warnings


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


def _resolve_javascript_import_edges(
    edges: list[dict[str, object]],
    all_paths: set[str],
    resolver_file_contents: dict[str, str],
    warnings: list[dict[str, object]],
) -> list[dict[str, object]]:
    resolved_edges: list[dict[str, object]] = []
    for edge in edges:
        if edge.get('type') != 'imports':
            resolved_edges.append(edge)
            continue
        metadata = dict(edge.get('metadata') or {})
        if metadata.get('language_family') != 'javascript':
            resolved_edges.append(edge)
            continue
        specifier = str(metadata.get('import_specifier') or '')
        source_path = str(edge.get('file') or '')
        resolved_path, resolver_name = resolve_js_module_path(source_path, specifier, all_paths, resolver_file_contents)
        if resolved_path is None:
            warnings.append({
                'code': 'unresolved_import',
                'message': 'JavaScript/TypeScript import could not be resolved to a local source file.',
                'path': source_path,
                'target': specifier,
            })
            resolved_edges.append(edge)
            continue
        metadata['resolver'] = resolver_name
        metadata['resolved_path'] = resolved_path
        resolved_edge = dict(edge)
        resolved_edge['target'] = _module_id_for_source_path(resolved_path)
        resolved_edge['confidence'] = 0.9 if resolver_name == 'relative' else 0.85
        resolved_edge['metadata'] = metadata
        resolved_edges.append(resolved_edge)
    return resolved_edges


def _resolve_javascript_symbol_alias_edges(edges: list[dict[str, object]], nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    node_ids = {str(node['id']) for node in nodes}
    default_exports_by_file: dict[str, str] = {}
    for node in nodes:
        file_path = str(node.get('file') or '')
        metadata = node.get('metadata') if isinstance(node.get('metadata'), dict) else {}
        if file_path and metadata.get('default_export') and file_path not in default_exports_by_file:
            default_exports_by_file[file_path] = str(node['id'])

    aliases_by_file: dict[tuple[str, str], tuple[str, str]] = {}
    namespace_aliases_by_file: dict[tuple[str, str], str] = {}
    reexport_aliases_by_file: dict[tuple[str, str], tuple[str, str]] = {}
    reexport_all_by_file: list[tuple[str, str]] = []
    alias_records: list[tuple[str, str, str, str, bool]] = []

    def resolve_imported_symbol(resolved_path: str, imported: str, seen: set[tuple[str, str]] | None = None) -> tuple[str, str] | None:
        seen = seen or set()
        marker = (resolved_path, imported)
        if marker in seen:
            return None
        seen.add(marker)
        if imported == 'default':
            target_id = default_exports_by_file.get(resolved_path)
            if target_id is None and f'{resolved_path}::default' in node_ids:
                target_id = f'{resolved_path}::default'
            return (target_id, imported) if target_id is not None else None
        target_id = f'{resolved_path}::{imported}'
        if target_id in node_ids:
            return target_id, imported
        reexported = reexport_aliases_by_file.get((resolved_path, imported))
        if reexported is not None:
            return reexported
        for barrel_path, target_path in reexport_all_by_file:
            if barrel_path != resolved_path:
                continue
            reexported_all_target = resolve_imported_symbol(target_path, imported, seen)
            if reexported_all_target is not None:
                return reexported_all_target
        return None

    for edge in edges:
        if edge.get('type') != 'imports':
            continue
        metadata = dict(edge.get('metadata') or {})
        if metadata.get('language_family') != 'javascript':
            continue
        resolved_path = str(metadata.get('resolved_path') or '')
        if not resolved_path:
            continue
        raw_aliases = metadata.get('import_aliases')
        source_path = str(edge.get('file') or '')
        if not isinstance(raw_aliases, dict):
            if metadata.get('re_export_all'):
                reexport_all_by_file.append((source_path, resolved_path))
            continue
        if metadata.get('re_export_all'):
            reexport_all_by_file.append((source_path, resolved_path))
        for local_name, imported_name in raw_aliases.items():
            local = str(local_name)
            imported = str(imported_name)
            if not local or not imported:
                continue
            if imported == '*':
                namespace_aliases_by_file[(source_path, local)] = resolved_path
                continue
            alias_records.append((source_path, resolved_path, local, imported, bool(metadata.get('re_export'))))

    changed = True
    while changed:
        changed = False
        for source_path, resolved_path, local, imported, is_re_export in alias_records:
            alias = resolve_imported_symbol(resolved_path, imported)
            if alias is None:
                continue
            target_map = reexport_aliases_by_file if is_re_export else aliases_by_file
            key = (source_path, local)
            if target_map.get(key) != alias:
                target_map[key] = alias
                changed = True

    if not aliases_by_file and not namespace_aliases_by_file:
        return edges

    resolved_edges: list[dict[str, object]] = []
    for edge in edges:
        if edge.get('type') not in {'calls', 'inherits', 'references'}:
            resolved_edges.append(edge)
            continue
        source_path = str(edge.get('file') or '')
        local_target = str(edge.get('target') or '')
        namespace_target = _split_js_namespace_target(local_target)
        if namespace_target is not None:
            local_namespace, imported_name = namespace_target
            resolved_path = namespace_aliases_by_file.get((source_path, local_namespace))
            if resolved_path is None:
                resolved_edges.append(edge)
                continue
            alias = resolve_imported_symbol(resolved_path, imported_name)
            if alias is None:
                resolved_edges.append(edge)
                continue
            target_id, resolved_imported_name = alias
            metadata = dict(edge.get('metadata') or {})
            metadata.update({
                'import_alias': local_namespace,
                'imported_name': resolved_imported_name,
                'namespace_alias_resolved': True,
                'alias_resolved': True,
            })
            resolved_edge = dict(edge)
            resolved_edge['target'] = target_id
            resolved_edge['metadata'] = metadata
            resolved_edge['confidence'] = max(float(edge.get('confidence', 0.0)), 0.85)
            resolved_edges.append(resolved_edge)
            continue
        if '::' in local_target:
            resolved_edges.append(edge)
            continue
        alias = aliases_by_file.get((source_path, local_target))
        if alias is None:
            resolved_edges.append(edge)
            continue
        target_id, imported_name = alias
        metadata = dict(edge.get('metadata') or {})
        metadata.update({
            'import_alias': local_target,
            'imported_name': imported_name,
            'alias_resolved': True,
        })
        resolved_edge = dict(edge)
        resolved_edge['target'] = target_id
        resolved_edge['metadata'] = metadata
        resolved_edge['confidence'] = max(float(edge.get('confidence', 0.0)), 0.85)
        resolved_edges.append(resolved_edge)
    return resolved_edges


def _attach_external_import_nodes(nodes: list[dict[str, object]], edges: list[dict[str, object]]) -> list[dict[str, object]]:
    node_ids = {str(node['id']) for node in nodes}
    external_nodes: dict[str, dict[str, object]] = {}
    for edge in edges:
        if edge.get('type') != 'imports':
            continue
        target = str(edge.get('target') or '')
        if not target or target in node_ids:
            continue
        if target.startswith('module::'):
            external_nodes[target] = _make_module_node(target, target.removeprefix('module::'))
        else:
            external_nodes[target] = _make_graph_node(target, 'external', _external_label(target), metadata={'external': True})
    return [*nodes, *external_nodes.values()]


def _normalize_repo_path(file_path: str) -> str | None:
    path = PurePosixPath(file_path)
    if path.is_absolute() or '..' in path.parts:
        return None
    return path.as_posix()


def _make_file_node(file_path: str, parent: str | None, *, analyzed: bool, language: str | None, skip_reason: str | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        'analyzed': analyzed,
        'language': language,
        'unsupported': language is None,
    }
    if skip_reason is not None:
        metadata['ignored'] = True
        metadata['skip_reason'] = skip_reason
    return _make_graph_node(
        file_path,
        'file',
        file_path.split('/')[-1],
        file_path=file_path,
        path=file_path,
        parent=parent,
        language=language,
        metadata=metadata,
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


def _should_warn_unresolved_call(edge: dict[str, object], target: str) -> bool:
    file_path = str(edge.get('file') or '')
    metadata = dict(edge.get('metadata') or {})
    if file_path.endswith(JS_TS_SOURCE_EXTENSIONS) and (
        target.startswith('attribute::')
        or target.startswith(JS_NAMESPACE_TARGET_PREFIX)
        or 'unresolved_attribute' in metadata
        or 'member_object' in metadata
    ):
        return False
    return True


def _attach_external_call_nodes(nodes: list[dict[str, object]], edges: list[dict[str, object]], warnings: list[dict[str, object]]) -> list[dict[str, object]]:
    node_ids = {str(node['id']) for node in nodes}
    external_nodes: dict[str, dict[str, object]] = {}

    for edge in edges:
        if edge['type'] not in {'calls', 'references'}:
            continue
        target = str(edge['target'])
        if target in node_ids:
            continue

        external_id = target if '::' in target else f'external::{target}'
        edge['target'] = external_id
        if edge['type'] == 'calls':
            edge['confidence'] = min(float(edge.get('confidence', 1.0)), 0.4)
        else:
            edge['confidence'] = min(float(edge.get('confidence', 1.0)), 0.5)
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
            if edge['type'] == 'calls' and _should_warn_unresolved_call(edge, target):
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
        if path.endswith(('src/main.ts', 'src/main.js', 'src/index.ts', 'src/index.js', 'server.ts', 'server.js', 'app.ts', 'app.js')):
            add(node, 'js_ts_entry_file', 0.75, 'JavaScript/TypeScript entrypoint filename convention')
        if path.endswith(('/route.ts', '/route.js', '/route.tsx', '/route.jsx')):
            add(node, 'js_ts_route_file', 0.8, 'JavaScript/TypeScript route file convention')
        if path.endswith(('vite.config.ts', 'vite.config.js', 'next.config.js', 'next.config.ts')):
            add(node, 'js_ts_config_entry', 0.65, 'JavaScript/TypeScript framework config convention')
        if node_type in {'function', 'method'} and label in {'GET', 'POST', 'PUT', 'PATCH', 'DELETE'} and '/api/' in path:
            add(node, 'js_ts_route_handler', 0.85, 'HTTP method export in route file')
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
        path_stem = path
        for suffix in JS_TS_SOURCE_EXTENSIONS + ('.py',):
            if path_stem.endswith(suffix):
                path_stem = path_stem.removesuffix(suffix)
                break
        path_tokens = set(path_stem.replace('/', '.').replace('_', '.').replace('-', '.').split('.'))
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


def parse_repo(repo_path, files, get_content_func, *, enabled_languages=None, resolver_file_contents: dict[str, str] | None = None):
    enabled_languages = normalize_enabled_languages(enabled_languages)
    resolver_file_contents = resolver_file_contents or {}
    root_tree_children: list[dict[str, object]] = []
    directory_tree_nodes: dict[str, dict[str, object]] = {}
    all_nodes: list[dict[str, object]] = []
    all_edges: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    normalized_file_paths: set[str] = set()

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
        normalized_file_paths.add(file_path)

        parts = file_path.split('/')
        parent_path: str | None = None
        siblings = root_tree_children
        for index, directory_name in enumerate(parts[:-1]):
            dir_path = '/'.join(parts[:index + 1])
            directory_node = ensure_directory(dir_path, directory_name, parent_path, siblings)
            parent_path = dir_path
            siblings = directory_node['children']

        language_spec = language_for_path(file_path, enabled_languages=enabled_languages)
        language = language_spec.id if language_spec is not None else None
        skip_reason = ignored_reason(file_path, language_spec=language_spec) if language_spec is not None else None
        file_tree_node = _make_tree_node(file_path, 'file', parts[-1])
        analyzed = False

        if skip_reason is not None:
            pass
        elif language == 'python':
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
        elif language in {'javascript', 'typescript'}:
            code = get_content_func(repo_path, file_path)
            if code is None:
                warnings.append({
                    'code': 'missing_content',
                    'message': 'JavaScript/TypeScript file content was not available for analysis.',
                    'path': file_path,
                })
            else:
                module_tree_node, nodes, edges, file_warnings = parse_javascript_file(code, file_path)
                file_tree_node['children'].append(module_tree_node)
                all_nodes.extend(nodes)
                all_edges.extend(edges)
                warnings.extend(file_warnings)
                analyzed = not any(warning.get('code') == 'syntax_error' for warning in file_warnings)

        siblings.append(file_tree_node)
        all_nodes.append(_make_file_node(file_path, parent_path, analyzed=analyzed, language=language, skip_reason=skip_reason))
        if parent_path is not None:
            all_edges.append(_make_edge(parent_path, file_path, 'contains', file_path))

    deduplicated_nodes = _deduplicate_nodes(all_nodes)
    all_edges = _resolve_javascript_import_edges(all_edges, normalized_file_paths, resolver_file_contents, warnings)
    all_edges = _resolve_javascript_symbol_alias_edges(all_edges, deduplicated_nodes)
    sorted_nodes = _sort_nodes(deduplicated_nodes)
    resolved_edges = resolve_edges(all_edges, sorted_nodes)
    nodes_with_imports = _sort_nodes(_attach_external_import_nodes(sorted_nodes, resolved_edges))
    nodes_with_external = _sort_nodes(_attach_external_call_nodes(nodes_with_imports, resolved_edges, warnings))
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
