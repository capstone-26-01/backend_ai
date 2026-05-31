from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
from math import ceil
from typing import Any, Mapping, cast


DEFAULT_SVG_WIDTH = 1400
DEFAULT_SVG_HEIGHT = 860
DEFAULT_NODE_LIMIT = 96

MIN_SVG_WIDTH = 900
MAX_SVG_WIDTH = 2200
MIN_SVG_HEIGHT = 560
MAX_SVG_HEIGHT = 1400
MIN_NODE_LIMIT = 20
MAX_NODE_LIMIT = 180

RELATIONSHIP_KINDS = {'imports', 'inherits', 'calls', 'references', 'entrypoint'}
PUBLIC_MODULE_BASENAMES = {'__init__.py', 'api.py', 'app.py', 'main.py', 'cli.py', 'manage.py', 'urls.py', 'asgi.py', 'wsgi.py'}
FLOW_MODULE_TOKENS = {'client', 'clients', 'handler', 'handlers', 'pipeline', 'route', 'routes', 'service', 'services', 'session', 'sessions', 'view', 'views', 'workflow'}
DOMAIN_MODULE_TOKENS = {'adapter', 'adapters', 'auth', 'cookie', 'cookies', 'github', 'llm', 'model', 'models', 'parser', 'parsers', 'repo', 'repository', 'repositories', 'schema', 'schemas', 'serializer', 'serializers', 'transport'}
SUPPORT_MODULE_TOKENS = {'compat', 'config', 'constant', 'constants', 'exception', 'exceptions', 'helper', 'helpers', 'hook', 'hooks', 'internal', 'middleware', 'setting', 'settings', 'signal', 'signals', 'status', 'structure', 'structures', 'throttle', 'throttles', 'type', 'types', 'utility', 'utils', 'validator', 'validators'}
NODE_KIND_PRIORITY = {
    'directory': 0,
    'module': 1,
    'file': 2,
    'class': 3,
    'function': 4,
    'method': 5,
    'external': 6,
}

THEMES = {
    'light': {
        'bg': '#f8f3e8',
        'bg2': '#e9f0dc',
        'panel': '#fffdf7',
        'panel2': '#f2ead9',
        'text': '#19231d',
        'muted': '#667161',
        'line': '#aaad9d',
        'relationship': '#2c7a6f',
        'accent': '#d05a2f',
        'shadow': '#d8cdb9',
    },
    'dark': {
        'bg': '#101713',
        'bg2': '#1e2d26',
        'panel': '#17211c',
        'panel2': '#21352c',
        'text': '#f2f0e7',
        'muted': '#aab7a6',
        'line': '#526157',
        'relationship': '#77d6c5',
        'accent': '#f39c65',
        'shadow': '#07100c',
    },
}

NODE_COLORS = {
    'directory': ('#4e8b58', '#f5fff0'),
    'file': ('#8b6f47', '#fff8e8'),
    'module': ('#287c91', '#ecfbff'),
    'class': ('#b27228', '#fff1da'),
    'function': ('#b24f38', '#fff0ec'),
    'method': ('#8756a3', '#f8edff'),
    'external': ('#6c7370', '#f3f5f4'),
}


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def normalize_svg_options(
    *,
    width: int | None = None,
    height: int | None = None,
    node_limit: int | None = None,
    theme: str | None = None,
) -> dict[str, int | str]:
    return {
        'width': clamp_int(width or DEFAULT_SVG_WIDTH, MIN_SVG_WIDTH, MAX_SVG_WIDTH),
        'height': clamp_int(height or DEFAULT_SVG_HEIGHT, MIN_SVG_HEIGHT, MAX_SVG_HEIGHT),
        'node_limit': clamp_int(node_limit or DEFAULT_NODE_LIMIT, MIN_NODE_LIMIT, MAX_NODE_LIMIT),
        'theme': theme if theme in THEMES else 'light',
    }


def _node_id(node: Mapping[str, Any]) -> str:
    return str(node.get('id') or '')


def _node_kind(node: Mapping[str, Any]) -> str:
    return str(node.get('kind') or node.get('type') or 'file')


def _node_path(node: Mapping[str, Any]) -> str:
    value = node.get('path') or node.get('file') or ''
    return str(value) if value is not None else ''


def _node_parent(node: Mapping[str, Any]) -> str:
    value = node.get('parent_id') or node.get('parent') or ''
    return str(value) if value is not None else ''


def _node_label(node: Mapping[str, Any]) -> str:
    label = str(node.get('symbol') or node.get('label') or _node_path(node) or _node_id(node))
    if label.startswith('module::'):
        label = label.removeprefix('module::')
    return label


def _shorten(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return f'{value[:max(1, max_length - 3)]}...'


def _top_group(node: Mapping[str, Any]) -> str:
    path = _node_path(node)
    if path:
        return path.split('/')[0]
    label = _node_label(node)
    return label.split('.')[0] if label else 'root'


def _edge_kind(edge: Mapping[str, Any]) -> str:
    return str(edge.get('kind') or edge.get('type') or '')


def _build_scores(
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    edges: list[Mapping[str, Any]],
    entrypoint_ids: set[str],
    key_module_ids: set[str],
) -> dict[str, int]:
    scores: dict[str, int] = {}
    degree = defaultdict(int)
    relationship_degree = defaultdict(int)

    for edge in edges:
        source = str(edge.get('source') or '')
        target = str(edge.get('target') or '')
        if source:
            degree[source] += 1
        if target:
            degree[target] += 1
        if _edge_kind(edge) in RELATIONSHIP_KINDS:
            relationship_degree[source] += 2
            relationship_degree[target] += 2

    for node_id, node in nodes_by_id.items():
        kind = _node_kind(node)
        path = _node_path(node)
        score = {
            'directory': 10,
            'file': 12,
            'module': 28,
            'class': 22,
            'function': 18,
            'method': 16,
            'external': 4,
        }.get(kind, 8)
        score += min(degree[node_id], 12) * 3
        score += min(relationship_degree[node_id], 18) * 4
        if node_id in entrypoint_ids:
            score += 140
        if node_id in key_module_ids:
            score += 110
        if kind == 'directory' and path and '/' not in path:
            score += 50
        if kind == 'external' and relationship_degree[node_id] == 0:
            score -= 20
        scores[node_id] = score
    return scores


def _ancestor_ids(node_id: str, nodes_by_id: Mapping[str, Mapping[str, Any]]) -> list[str]:
    ancestors = []
    current = node_id
    seen: set[str] = set()
    while current and current not in seen and current in nodes_by_id:
        seen.add(current)
        ancestors.append(current)
        current = _node_parent(nodes_by_id[current])
    return ancestors


def _node_title(node: Mapping[str, Any]) -> str:
    path = _node_path(node)
    if path:
        return path.split('/')[-1]
    return _node_label(node)


def _node_subtitle(node: Mapping[str, Any]) -> str:
    path = _node_path(node)
    if path:
        return path
    return _node_id(node)


def _node_card(
    *,
    card_id: str,
    category: str,
    node: Mapping[str, Any],
    source_ids: set[str] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    score: int = 0,
) -> dict[str, Any]:
    node_id = _node_id(node)
    return {
        'id': card_id,
        'category': category,
        'node_id': node_id,
        'title': title or _node_title(node),
        'subtitle': subtitle or _node_subtitle(node),
        'kind': _node_kind(node),
        'score': score,
        'source_ids': set(source_ids or {node_id}),
    }


def _nearest_module_id(node_id: str, nodes_by_id: Mapping[str, Mapping[str, Any]]) -> str | None:
    for ancestor_id in _ancestor_ids(node_id, nodes_by_id):
        node = nodes_by_id.get(ancestor_id)
        if node is not None and _node_kind(node) == 'module' and _node_path(node):
            return ancestor_id
    return None


def _module_basename(node: Mapping[str, Any]) -> str:
    path = _node_path(node)
    return path.rsplit('/', 1)[-1] if path else _node_label(node)


def _is_test_path(path: str) -> bool:
    return path.startswith(('tests/', 'test/')) or '/tests/' in path or '/test/' in path


def _module_relation_counts(
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    edges: list[Mapping[str, Any]],
) -> tuple[Counter[tuple[str, str]], Counter[str], Counter[str]]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    fan_in: Counter[str] = Counter()
    fan_out: Counter[str] = Counter()
    for edge in edges:
        if _edge_kind(edge) not in RELATIONSHIP_KINDS:
            continue
        source_module = _nearest_module_id(str(edge.get('source') or ''), nodes_by_id)
        target_module = _nearest_module_id(str(edge.get('target') or ''), nodes_by_id)
        if not source_module or not target_module or source_module == target_module:
            continue
        pair_counts[(source_module, target_module)] += 1
        fan_out[source_module] += 1
        fan_in[target_module] += 1
    return pair_counts, fan_in, fan_out


def _module_score(
    module_id: str,
    module: Mapping[str, Any],
    fan_in: Mapping[str, int],
    fan_out: Mapping[str, int],
    key_module_ids: set[str],
    entrypoint_module_ids: set[str],
) -> int:
    path = _node_path(module)
    basename = _module_basename(module)
    tokens = _module_tokens(module)
    hints = PUBLIC_MODULE_BASENAMES | FLOW_MODULE_TOKENS | DOMAIN_MODULE_TOKENS | SUPPORT_MODULE_TOKENS
    score = fan_in.get(module_id, 0) * 3 + fan_out.get(module_id, 0) * 2
    if module_id in key_module_ids:
        score += 70
    if module_id in entrypoint_module_ids:
        score += 55
    if basename in {'__init__.py', 'api.py', 'app.py', 'main.py', 'cli.py', 'manage.py'}:
        score += 65
    if tokens & hints:
        score += 28
    if basename.startswith('_') and score > 0:
        score -= 10
    return score


def _module_tokens(module: Mapping[str, Any]) -> set[str]:
    path = _node_path(module).removesuffix('.py')
    return {token for token in path.replace('/', '.').replace('-', '_').replace('_', '.').split('.') if token}


def _module_name_tokens(module: Mapping[str, Any]) -> set[str]:
    basename = _module_basename(module).removesuffix('.py')
    return {token for token in basename.replace('-', '_').replace('_', '.').split('.') if token}


def _public_module_score(module_id: str, module: Mapping[str, Any], entrypoint_module_ids: set[str]) -> int:
    path = _node_path(module)
    basename = _module_basename(module)
    score = 0
    if basename in {'api.py', 'app.py', 'main.py', 'cli.py', 'manage.py', 'urls.py'}:
        score += 110
    if basename == '__init__.py':
        score += 75
    if basename in {'asgi.py', 'wsgi.py'}:
        score += 45
    if module_id in entrypoint_module_ids and basename not in {'certs.py', 'help.py'}:
        score += 25
    if basename in {'help.py', 'certs.py'}:
        score += 8
    if path.count('/') <= 1:
        score += 20
    return score


def _module_role(
    module_id: str,
    module: Mapping[str, Any],
    entrypoint_module_ids: set[str],
) -> str:
    basename = _module_basename(module)
    tokens = _module_name_tokens(module)
    if basename in PUBLIC_MODULE_BASENAMES:
        return 'entry'
    if basename.startswith('_') or tokens & SUPPORT_MODULE_TOKENS:
        return 'support'
    if tokens & FLOW_MODULE_TOKENS:
        return 'flow'
    if tokens & DOMAIN_MODULE_TOKENS:
        return 'domain'
    return 'domain'


def _entrypoint_card(
    entrypoint: Mapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, int],
) -> dict[str, Any] | None:
    entrypoint_id = str(entrypoint.get('id') or '')
    node = nodes_by_id.get(entrypoint_id)
    if node is None:
        return None
    title = str(entrypoint.get('label') or _node_label(node))
    subtitle = str(entrypoint.get('path') or _node_subtitle(node))
    return _node_card(
        card_id=f'entry:{entrypoint_id}',
        category='entry',
        node=node,
        title=title,
        subtitle=subtitle,
        score=scores.get(entrypoint_id, 0),
    )


def _dedupe_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    for card in cards:
        node_id = str(card.get('node_id') or '')
        category = str(card.get('category') or '')
        dedupe_key = f'{category}:{node_id}'
        if dedupe_key in seen_nodes:
            continue
        seen_nodes.add(dedupe_key)
        result.append(card)
    return result


def _build_overview_cards(
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    edges: list[Mapping[str, Any]],
    entrypoints: list[Mapping[str, Any]],
    key_modules: list[Mapping[str, Any]],
    *,
    node_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    key_module_ids = {str(module.get('id')) for module in key_modules if module.get('id')}
    entrypoint_module_ids = {
        module_id
        for entrypoint in entrypoints
        if (module_id := _nearest_module_id(str(entrypoint.get('id') or ''), nodes_by_id)) is not None
    }
    module_nodes = {
        node_id: node
        for node_id, node in nodes_by_id.items()
        if _node_kind(node) == 'module' and _node_path(node) and not _is_test_path(_node_path(node))
    }
    _pair_counts, fan_in, fan_out = _module_relation_counts(nodes_by_id, edges)
    scores = {
        module_id: _module_score(module_id, module, fan_in, fan_out, key_module_ids, entrypoint_module_ids)
        for module_id, module in module_nodes.items()
    }
    max_cards = clamp_int(max(10, node_limit // 7), 10, 16)

    ranked_modules = sorted(
        module_nodes,
        key=lambda module_id: (-scores.get(module_id, 0), _node_path(module_nodes[module_id]), module_id),
    )
    public_modules = sorted(
        module_nodes,
        key=lambda module_id: (-_public_module_score(module_id, module_nodes[module_id], entrypoint_module_ids), -scores.get(module_id, 0), _node_path(module_nodes[module_id])),
    )
    public_ids = [
        module_id
        for module_id in public_modules
        if _public_module_score(module_id, module_nodes[module_id], entrypoint_module_ids) >= 75
    ][:2]
    if not public_ids:
        public_ids = ranked_modules[:2]

    roles = {
        module_id: _module_role(module_id, module_nodes[module_id], entrypoint_module_ids)
        for module_id in module_nodes
    }
    selected = set(public_ids)

    def take(role: str, limit: int) -> list[str]:
        result = [
            module_id
            for module_id in ranked_modules
            if module_id not in selected and roles.get(module_id) == role
        ][: max(0, limit)]
        selected.update(result)
        return result

    flow_ids = take('flow', min(3, max_cards - len(selected)))
    if not flow_ids and len(selected) < max_cards:
        fallback = [module_id for module_id in ranked_modules if module_id not in selected][:1]
        selected.update(fallback)
        flow_ids = fallback
    domain_ids = take('domain', min(4, max_cards - len(selected)))
    support_ids = take('support', min(5, max_cards - len(selected)))

    entry_cards = [
        _node_card(card_id=f'entry:{module_id}', category='entry', node=module_nodes[module_id], score=scores.get(module_id, 0))
        for module_id in public_ids
    ]
    flow_cards = [
        _node_card(card_id=f'flow:{module_id}', category='flow', node=module_nodes[module_id], score=scores.get(module_id, 0))
        for module_id in flow_ids
    ]
    domain_cards = [
        _node_card(card_id=f'domain:{module_id}', category='domain', node=module_nodes[module_id], score=scores.get(module_id, 0))
        for module_id in domain_ids
    ]
    support_cards = [
        _node_card(card_id=f'support:{module_id}', category='support', node=module_nodes[module_id], score=scores.get(module_id, 0))
        for module_id in support_ids
    ]
    return [*entry_cards, *flow_cards, *domain_cards, *support_cards], scores


def select_overview_cards(
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    edges: list[Mapping[str, Any]],
    entrypoints: list[Mapping[str, Any]],
    key_modules: list[Mapping[str, Any]],
    *,
    node_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    return _build_overview_cards(
        nodes_by_id,
        edges,
        entrypoints,
        key_modules,
        node_limit=node_limit,
    )


def _layout_cards(
    cards: list[dict[str, Any]],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[dict[str, tuple[float, float, float, float]], list[dict[str, Any]]]:
    categories = [
        ('entry', 'Public API'),
        ('flow', 'Execution Flow'),
        ('domain', 'Core Modules'),
        ('support', 'Shared Support'),
    ]
    column_gap = 20
    column_width = (width - column_gap * (len(categories) - 1)) / len(categories)
    card_gap = 14
    category_cards = {category: [card for card in cards if card['category'] == category] for category, _label in categories}
    positions: dict[str, tuple[float, float, float, float]] = {}
    columns = []

    for index, (category, label) in enumerate(categories):
        column_x = x + index * (column_width + column_gap)
        column_y = y
        available_height = height
        cards_for_column = category_cards[category]
        card_count = max(1, len(cards_for_column))
        card_height = clamp_int(int((available_height - card_gap * (card_count - 1) - 58) / card_count), 58, 82)
        columns.append({
            'category': category,
            'label': label,
            'x': column_x,
            'y': column_y,
            'width': column_width,
            'height': available_height,
            'count': len(cards_for_column),
        })
        current_y = column_y + 48
        for card in cards_for_column:
            if current_y + card_height > column_y + available_height - 8:
                break
            positions[str(card['id'])] = (column_x, current_y, column_width, card_height)
            current_y += card_height + card_gap

    return positions, columns


def _card_owner_map(cards: list[dict[str, Any]], nodes_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    source_owner = {}
    for card in cards:
        for source_id in cast(set[str], card['source_ids']):
            source_owner[source_id] = str(card['id'])

    owner = {}
    for node_id in nodes_by_id:
        for ancestor_id in _ancestor_ids(node_id, nodes_by_id):
            if ancestor_id in source_owner:
                owner[node_id] = source_owner[ancestor_id]
                break
    return owner


def _overview_edges(
    cards: list[dict[str, Any]],
    card_positions: Mapping[str, tuple[float, float, float, float]],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    edges: list[Mapping[str, Any]],
    *,
    limit: int = 24,
) -> list[dict[str, Any]]:
    owner = _card_owner_map(cards, nodes_by_id)
    category_order = {'entry': 0, 'flow': 1, 'domain': 2, 'support': 3}
    card_by_id = {str(card['id']): card for card in cards}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    kinds_by_pair: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for edge in edges:
        kind = _edge_kind(edge)
        if kind not in RELATIONSHIP_KINDS:
            continue
        source_card = owner.get(str(edge.get('source') or ''))
        target_card = owner.get(str(edge.get('target') or ''))
        if not source_card or not target_card or source_card == target_card:
            continue
        if source_card not in card_positions or target_card not in card_positions:
            continue
        source_category = str(card_by_id[source_card]['category'])
        target_category = str(card_by_id[target_card]['category'])
        if source_category == target_category:
            continue
        if category_order.get(source_category, 99) > category_order.get(target_category, 99):
            source_card, target_card = target_card, source_card
        pair = (source_card, target_card)
        counts[pair] += 1
        kinds_by_pair[pair][kind] += 1

    actual_edges = [
        {
            'source': source,
            'target': target,
            'kind': kinds_by_pair[(source, target)].most_common(1)[0][0],
            'count': count,
            'inferred': False,
        }
        for (source, target), count in counts.items()
    ]
    inferred_edges: list[dict[str, Any]] = []
    has_entry_edge = any(str(card_by_id.get(edge['source'], {}).get('category')) == 'entry' for edge in actual_edges)
    if not has_entry_edge:
        entry_cards = [card for card in cards if str(card.get('category')) == 'entry' and str(card.get('id')) in card_positions]
        flow_cards = [
            card for card in cards
            if str(card.get('category')) == 'flow' and str(card.get('id')) in card_positions
        ]
        domain_cards = [
            card for card in cards
            if str(card.get('category')) == 'domain' and str(card.get('id')) in card_positions
        ]
        flow_cards.sort(key=lambda card: (-int(card.get('score') or 0), str(card.get('title') or '')))
        domain_cards.sort(key=lambda card: (-int(card.get('score') or 0), str(card.get('title') or '')))
        downstream_cards = flow_cards or domain_cards
        if downstream_cards:
            target = str(downstream_cards[0]['id'])
            for card in entry_cards[:2]:
                inferred_edges.append({
                    'source': str(card['id']),
                    'target': target,
                    'kind': 'facade',
                    'count': 1,
                    'inferred': True,
                })

    def edge_sort_key(item: Mapping[str, Any]) -> tuple[int, int, str, str]:
        source_category = str(card_by_id.get(str(item['source']), {}).get('category') or '')
        target_category = str(card_by_id.get(str(item['target']), {}).get('category') or '')
        pair_priority = {
            ('flow', 'domain'): 0,
            ('domain', 'support'): 1,
            ('flow', 'support'): 2,
            ('entry', 'flow'): 3,
            ('entry', 'domain'): 4,
        }.get((source_category, target_category), 5)
        return (pair_priority, -int(item['count']), str(item['source']), str(item['target']))

    actual_limit = max(0, limit - len(inferred_edges))
    sorted_actual = sorted(actual_edges, key=edge_sort_key)[:actual_limit]
    return [*inferred_edges, *sorted_actual]


def _render_stat(label: str, value: object, x: int, y: int, theme: Mapping[str, str]) -> str:
    return (
        f'<g>'
        f'<rect x="{x}" y="{y}" width="76" height="48" rx="14" fill="{theme["panel2"]}" opacity="0.94"/>'
        f'<text x="{x + 12}" y="{y + 19}" font-size="11" fill="{theme["muted"]}">{escape(label)}</text>'
        f'<text x="{x + 12}" y="{y + 38}" font-size="18" font-weight="800" fill="{theme["text"]}">{escape(str(value))}</text>'
        f'</g>'
    )


def _safe_edge_id(edge: Mapping[str, Any], index: int) -> str:
    return str(edge.get('id') or f'edge-{index}')


def _category_style(category: str) -> tuple[str, str]:
    return {
        'entry': ('#d05a2f', '#fff4ec'),
        'flow': ('#287c91', '#ecfbff'),
        'domain': ('#5f6fb3', '#f0f3ff'),
        'support': ('#4e8b58', '#f4fff1'),
    }.get(category, ('#6c7370', '#f3f5f4'))


def _category_title(category: str) -> str:
    return {
        'entry': 'Public API',
        'flow': 'Execution Flow',
        'domain': 'Core Modules',
        'support': 'Shared Support',
    }.get(category, 'Modules')


def _category_description(category: str) -> str:
    return {
        'entry': 'Where callers enter',
        'flow': 'Coordinates work',
        'domain': 'Owns repo behavior',
        'support': 'Shared helpers',
    }.get(category, 'Selected modules')


def _render_module_chip(
    card: Mapping[str, Any],
    *,
    x_pos: float,
    y_pos: float,
    width: float,
    theme: Mapping[str, str],
) -> str:
    stroke, fill = _category_style(str(card.get('category') or 'support'))
    title = _shorten(str(card.get('title') or ''), 19)
    subtitle = _shorten(str(card.get('subtitle') or ''), 28)
    return (
        f'<g>'
        f'<rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{width:.1f}" height="54" rx="15" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        f'<circle cx="{x_pos + 19:.1f}" cy="{y_pos + 20:.1f}" r="6.5" fill="{stroke}" opacity=".9"/>'
        f'<text x="{x_pos + 34:.1f}" y="{y_pos + 23:.1f}" class="card-title">{escape(title)}</text>'
        f'<text x="{x_pos + 15:.1f}" y="{y_pos + 42:.1f}" class="card-subtitle">{escape(subtitle)}</text>'
        f'</g>'
    )


def _render_structure_overview(
    cards: list[dict[str, Any]],
    *,
    x_pos: float,
    y_pos: float,
    width: float,
    height: float,
    theme: Mapping[str, str],
) -> str:
    categories = ['entry', 'flow', 'domain', 'support']
    card_by_id = {str(card['id']): card for card in cards}
    cards_by_category = {
        category: sorted(
            [card for card in cards if str(card.get('category')) == category],
            key=lambda card: (-int(card.get('score') or 0), str(card.get('title') or '')),
        )
        for category in categories
    }
    column_gap = 18
    column_width = (width - column_gap * (len(categories) - 1)) / len(categories)
    header_y = y_pos + 30
    rail_y = y_pos + 58
    capsule_y = y_pos + 82
    capsule_height = 52
    panel_y = y_pos + 150
    panel_height = height - 168
    chip_gap = 11
    chip_height = 54
    max_chips = max(1, int((panel_height - 66) / (chip_height + chip_gap)))
    chunks = [
        f'<rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{width:.1f}" height="{height:.1f}" rx="28" fill="{theme["panel"]}" opacity=".70" filter="url(#soft-shadow)"/>',
        f'<text x="{x_pos + 24:.1f}" y="{header_y:.1f}" class="column-label">CODEBASE STRUCTURE OVERVIEW</text>',
        f'<text x="{x_pos + width - 24:.1f}" y="{header_y:.1f}" text-anchor="end" class="small">role groups first, raw edges only summarized</text>',
    ]

    for index in range(len(categories) - 1):
        source_center = x_pos + index * (column_width + column_gap) + column_width / 2
        target_center = x_pos + (index + 1) * (column_width + column_gap) + column_width / 2
        start_x = source_center + 46
        end_x = target_center - 46
        chunks.append(
            f'<path d="M {start_x:.1f} {rail_y:.1f} L {end_x:.1f} {rail_y:.1f}" stroke="{theme["relationship"]}" stroke-width="3" stroke-linecap="round" opacity=".52" marker-end="url(#arrow)"/>'
        )

    for index, category in enumerate(categories):
        column_x = x_pos + index * (column_width + column_gap)
        stroke, fill = _category_style(category)
        title = _category_title(category)
        description = _category_description(category)
        category_cards = cards_by_category[category]
        chunks.append(
            f'<rect x="{column_x:.1f}" y="{capsule_y:.1f}" width="{column_width:.1f}" height="{capsule_height:.1f}" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>'
            f'<text x="{column_x + column_width / 2:.1f}" y="{capsule_y + 23:.1f}" text-anchor="middle" class="capsule-title">{escape(title)}</text>'
            f'<text x="{column_x + column_width / 2:.1f}" y="{capsule_y + 40:.1f}" text-anchor="middle" class="capsule-subtitle">{escape(description)}</text>'
            f'<rect x="{column_x:.1f}" y="{panel_y:.1f}" width="{column_width:.1f}" height="{panel_height:.1f}" rx="22" fill="{fill}" opacity=".58"/>'
            f'<text x="{column_x + 16:.1f}" y="{panel_y + 30:.1f}" class="group-label">{len(category_cards)} MODULES</text>'
        )
        chip_y = panel_y + 48
        for card in category_cards[:max_chips]:
            chunks.append(
                _render_module_chip(
                    card,
                    x_pos=column_x + 12,
                    y_pos=chip_y,
                    width=column_width - 24,
                    theme=theme,
                )
            )
            chip_y += chip_height + chip_gap
        hidden_count = max(0, len(category_cards) - max_chips)
        if hidden_count:
            chunks.append(
                f'<text x="{column_x + column_width / 2:.1f}" y="{panel_y + panel_height - 18:.1f}" text-anchor="middle" class="small">+ {hidden_count} more in app</text>'
            )
        if not category_cards:
            chunks.append(
                f'<text x="{column_x + column_width / 2:.1f}" y="{panel_y + 82:.1f}" text-anchor="middle" class="small">No module selected</text>'
            )

    return f'<g>{"".join(chunks)}</g>'


def render_share_graph_svg(
    payload: Mapping[str, Any],
    *,
    width: int = DEFAULT_SVG_WIDTH,
    height: int = DEFAULT_SVG_HEIGHT,
    node_limit: int = DEFAULT_NODE_LIMIT,
    theme_name: str = 'light',
) -> str:
    options = normalize_svg_options(width=width, height=height, node_limit=node_limit, theme=theme_name)
    width = cast(int, options['width'])
    height = cast(int, options['height'])
    node_limit = cast(int, options['node_limit'])
    theme = THEMES[cast(str, options['theme'])]

    graph = cast(Mapping[str, Any], payload.get('graph') or {})
    nodes = [cast(Mapping[str, Any], node) for node in graph.get('nodes', []) if isinstance(node, Mapping)]
    edges = [cast(Mapping[str, Any], edge) for edge in graph.get('edges', []) if isinstance(edge, Mapping)]
    nodes_by_id = {_node_id(node): node for node in nodes if _node_id(node)}
    entrypoints = [cast(Mapping[str, Any], item) for item in graph.get('entrypoints', []) if isinstance(item, Mapping)]
    key_modules = [cast(Mapping[str, Any], item) for item in graph.get('key_modules', []) if isinstance(item, Mapping)]

    graph_x = 344
    graph_y = 154
    graph_width = width - graph_x - 36
    graph_height = height - graph_y - 50
    cards, _scores = select_overview_cards(nodes_by_id, edges, entrypoints, key_modules, node_limit=node_limit)
    card_positions, columns = _layout_cards(cards, x=graph_x, y=graph_y, width=graph_width, height=graph_height)
    visible_cards = [card for card in cards if str(card['id']) in card_positions]
    visible_edges = _overview_edges(visible_cards, card_positions, nodes_by_id, edges, limit=7)

    hidden_nodes = max(0, len(nodes_by_id) - len(visible_cards))
    represented_edges = sum(int(edge['count']) for edge in visible_edges)
    hidden_edges = max(0, len(edges) - represented_edges)
    repo = str(payload.get('repo') or graph.get('repo') or 'unknown/repo')
    revision = _shorten(str(payload.get('revision') or graph.get('revision') or ''), 12)
    title = str(payload.get('title') or repo)

    defs = f'''
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="{theme['bg']}"/>
      <stop offset="100%" stop-color="{theme['bg2']}"/>
    </linearGradient>
    <filter id="soft-shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="10" flood-color="{theme['shadow']}" flood-opacity="0.22"/>
    </filter>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L8,3 z" fill="{theme['relationship']}"/>
    </marker>
  </defs>'''

    styles = f'''
  <style>
    .title {{ font: 800 34px Georgia, 'Times New Roman', serif; fill: {theme['text']}; }}
    .subtitle {{ font: 500 14px ui-sans-serif, system-ui, sans-serif; fill: {theme['muted']}; }}
    .small {{ font: 600 10px ui-sans-serif, system-ui, sans-serif; fill: {theme['muted']}; }}
    .group-label {{ font: 800 12px ui-sans-serif, system-ui, sans-serif; fill: {theme['muted']}; letter-spacing: .02em; }}
    .column-label {{ font: 900 13px ui-sans-serif, system-ui, sans-serif; fill: {theme['text']}; letter-spacing: .08em; }}
    .card-title {{ font: 800 14px ui-sans-serif, system-ui, sans-serif; fill: #16221d; }}
    .card-subtitle {{ font: 600 11px ui-sans-serif, system-ui, sans-serif; fill: #5f6d64; }}
    .capsule-title {{ font: 900 12px ui-sans-serif, system-ui, sans-serif; fill: #16221d; letter-spacing: .04em; }}
    .capsule-subtitle {{ font: 700 9px ui-sans-serif, system-ui, sans-serif; fill: #5f6d64; }}
    .tiny {{ font: 700 9px ui-sans-serif, system-ui, sans-serif; fill: #69766e; }}
    .edge-label {{ font: 900 10px ui-sans-serif, system-ui, sans-serif; fill: {theme['muted']}; }}
  </style>'''

    stat_y = 44
    header = f'''
  <rect width="{width}" height="{height}" rx="0" fill="url(#bg)"/>
  <circle cx="{width - 120}" cy="92" r="178" fill="{theme['panel']}" opacity=".32"/>
  <circle cx="{width - 38}" cy="{height - 60}" r="132" fill="{theme['accent']}" opacity=".10"/>
  <text x="34" y="48" class="title">{escape(_shorten(title, 46))}</text>
  <text x="36" y="76" class="subtitle">GitStarter dynamic codebase graph - {escape(repo)} - {escape(revision)}</text>
  {_render_stat('nodes', len(nodes_by_id), width - 358, stat_y, theme)}
  {_render_stat('edges', len(edges), width - 274, stat_y, theme)}
  {_render_stat('cards', len(visible_cards), width - 190, stat_y, theme)}
  {_render_stat('hidden', hidden_nodes, width - 106, stat_y, theme)}
'''

    sidebar_lines = [
        f'<text x="58" y="154" class="group-label">HOW TO READ</text>',
    ]
    cursor_y = 178
    guide_lines = [
        'Start at Public API',
        'Read columns as module roles',
        'Arrows summarize group dependencies',
        'Open app for raw symbol graph',
    ]
    for item in guide_lines:
        sidebar_lines.append(f'<text x="58" y="{cursor_y}" class="small">- {escape(item)}</text>')
        cursor_y += 19
    cursor_y += 18
    sidebar_lines.append(f'<text x="58" y="{cursor_y}" class="group-label">RENDERED SCOPE</text>')
    cursor_y += 24
    scope_lines = [
        f'{len(visible_cards)} modules represented',
        f'{len(visible_edges)} relationships summarized',
        f'{hidden_nodes} low-signal symbols collapsed',
        'Tests and externals omitted',
    ]
    for item in scope_lines:
        sidebar_lines.append(f'<text x="58" y="{cursor_y}" class="small">- {escape(item)}</text>')
        cursor_y += 19
    sidebar_lines.append(f'<text x="58" y="{height - 78}" class="small">{escape(str(payload.get("mode") or "fixed"))} - README-safe SVG</text>')
    sidebar_lines.append(f'<text x="58" y="{height - 56}" class="small">Collapsed symbols open in the app</text>')

    sidebar = f'''
  <rect x="34" y="116" width="276" height="{height - 148}" rx="26" fill="{theme['panel']}" opacity=".92" filter="url(#soft-shadow)"/>
  <circle cx="72" cy="104" r="18" fill="{theme['accent']}"/>
  <text x="66" y="111" font-size="18" font-weight="900" fill="#fff">G</text>
  {''.join(sidebar_lines)}
'''

    structure_overview = _render_structure_overview(
        visible_cards,
        x_pos=graph_x,
        y_pos=graph_y,
        width=graph_width,
        height=graph_height,
        theme=theme,
    )

    legend_y = height - 34
    legend = f'''
  <text x="{graph_x}" y="{legend_y}" class="small">Structure overview: modules are grouped by role and arrows summarize dependency direction. Builtins, external calls, tests, and {hidden_nodes} low-signal nodes are collapsed.</text>
'''

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(repo)} GitStarter codebase graph">
{defs}
{styles}
{header}
{sidebar}
  {structure_overview}
{legend}
</svg>
'''
