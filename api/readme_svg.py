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


def _entrypoint_card(
    entrypoint: Mapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, int],
) -> dict[str, Any] | None:
    entrypoint_id = str(entrypoint.get('id') or '')
    node = nodes_by_id.get(entrypoint_id)
    if node is None:
        return None
    source_ids = set(_ancestor_ids(entrypoint_id, nodes_by_id))
    title = str(entrypoint.get('label') or _node_label(node))
    subtitle = str(entrypoint.get('path') or _node_subtitle(node))
    return _node_card(
        card_id=f'entry:{entrypoint_id}',
        category='entry',
        node=node,
        source_ids=source_ids,
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
    entrypoint_ids = {str(entrypoint.get('id')) for entrypoint in entrypoints if entrypoint.get('id')}
    scores = _build_scores(nodes_by_id, edges, entrypoint_ids, key_module_ids)
    max_cards = clamp_int(max(12, node_limit // 5), 12, 24)

    entry_cards = [
        card
        for entrypoint in entrypoints[:4]
        if (card := _entrypoint_card(entrypoint, nodes_by_id, scores)) is not None
    ]

    core_cards = []
    for module in key_modules:
        node_id = str(module.get('id') or '')
        node = nodes_by_id.get(node_id)
        if node is None or _node_kind(node) == 'external':
            continue
        if _node_path(node).startswith('tests/'):
            continue
        core_cards.append(
            _node_card(
                card_id=f'core:{node_id}',
                category='core',
                node=node,
                source_ids=set(_ancestor_ids(node_id, nodes_by_id)),
                score=scores.get(node_id, 0),
            )
        )
    core_cards = _dedupe_cards(core_cards)[: min(6, max_cards - len(entry_cards))]

    used_node_ids = {str(card['node_id']) for card in [*entry_cards, *core_cards]}
    support_candidates = []
    for node_id, node in nodes_by_id.items():
        kind = _node_kind(node)
        path = _node_path(node)
        if node_id in used_node_ids or kind == 'external':
            continue
        if kind == 'directory' and path and '/' not in path:
            support_candidates.append(node_id)
        elif kind == 'module' and path and not path.startswith('tests/'):
            support_candidates.append(node_id)

    support_candidates = sorted(
        support_candidates,
        key=lambda node_id: (-scores.get(node_id, 0), NODE_KIND_PRIORITY.get(_node_kind(nodes_by_id[node_id]), 9), _node_path(nodes_by_id[node_id]), node_id),
    )
    support_budget = max(0, max_cards - len(entry_cards) - len(core_cards))
    support_cards = [
        _node_card(
            card_id=f'support:{node_id}',
            category='support',
            node=nodes_by_id[node_id],
            source_ids=set(_ancestor_ids(node_id, nodes_by_id)),
            score=scores.get(node_id, 0),
        )
        for node_id in support_candidates[: min(6, support_budget)]
    ]

    return [*entry_cards, *core_cards, *support_cards], scores


def _layout_cards(
    cards: list[dict[str, Any]],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[dict[str, tuple[float, float, float, float]], list[dict[str, Any]]]:
    categories = [
        ('entry', 'Start here'),
        ('core', 'Core modules'),
        ('support', 'Structure'),
    ]
    column_gap = 26
    column_width = (width - column_gap * 2) / 3
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
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
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
        counts[(source_card, target_card, kind)] += 1

    result = [
        {'source': source, 'target': target, 'kind': kind, 'count': count}
        for (source, target, kind), count in counts.items()
    ]
    return sorted(result, key=lambda item: (-int(item['count']), str(item['source']), str(item['target'])))[:limit]


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
        'core': ('#287c91', '#ecfbff'),
        'support': ('#4e8b58', '#f4fff1'),
    }.get(category, ('#6c7370', '#f3f5f4'))


def _render_card(card: Mapping[str, Any], box: tuple[float, float, float, float], theme: Mapping[str, str]) -> str:
    x_pos, y_pos, width, height = box
    stroke, fill = _category_style(str(card.get('category') or 'support'))
    title = _shorten(str(card.get('title') or ''), 26)
    subtitle = _shorten(str(card.get('subtitle') or ''), 40)
    kind = _shorten(str(card.get('kind') or ''), 12)
    return (
        f'<g>'
        f'<rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{width:.1f}" height="{height:.1f}" rx="18" fill="{fill}" stroke="{stroke}" stroke-width="1.8" filter="url(#soft-shadow)"/>'
        f'<circle cx="{x_pos + 24:.1f}" cy="{y_pos + 25:.1f}" r="8" fill="{stroke}" opacity=".9"/>'
        f'<text x="{x_pos + 42:.1f}" y="{y_pos + 27:.1f}" class="card-title">{escape(title)}</text>'
        f'<text x="{x_pos + 18:.1f}" y="{y_pos + 50:.1f}" class="card-subtitle">{escape(subtitle)}</text>'
        f'<rect x="{x_pos + width - 76:.1f}" y="{y_pos + height - 27:.1f}" width="58" height="18" rx="9" fill="{theme["panel"]}" opacity=".82"/>'
        f'<text x="{x_pos + width - 47:.1f}" y="{y_pos + height - 14:.1f}" text-anchor="middle" class="tiny">{escape(kind)}</text>'
        f'</g>'
    )


def _render_relationship_pipeline(
    columns: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    visible_edges: list[dict[str, Any]],
    theme: Mapping[str, str],
    *,
    x: float,
    width: float,
    y: float,
) -> str:
    columns_by_category = {str(column['category']): column for column in columns}
    cards_by_id = {str(card['id']): card for card in cards}
    pair_counts: Counter[tuple[str, str]] = Counter()
    kind_counts: Counter[str] = Counter()
    for edge in visible_edges:
        source_card = cards_by_id.get(str(edge['source']))
        target_card = cards_by_id.get(str(edge['target']))
        if source_card is None or target_card is None:
            continue
        count = int(edge['count'])
        source_category = str(source_card['category'])
        target_category = str(target_card['category'])
        pair_counts[(source_category, target_category)] += count
        kind_counts[str(edge['kind'])] += count

    total_count = sum(kind_counts.values())

    def center(category: str) -> tuple[float, float]:
        column = columns_by_category[category]
        return float(column['x']) + float(column['width']) / 2, y

    def segment(source: str, target: str) -> str:
        source_x, source_y = center(source)
        target_x, target_y = center(target)
        count = pair_counts[(source, target)] + pair_counts[(target, source)]
        stroke = theme['relationship'] if count else theme['line']
        opacity = '.82' if count else '.26'
        return (
            f'<path d="M {source_x + 78:.1f} {source_y:.1f} L {target_x - 78:.1f} {target_y:.1f}" '
            f'stroke="{stroke}" stroke-width="4.5" stroke-linecap="round" opacity="{opacity}" marker-end="url(#arrow)"/>'
        )

    return f'''
  <g>
    <rect x="{x:.1f}" y="{y - 28:.1f}" width="{width:.1f}" height="58" rx="25" fill="{theme['panel']}" opacity=".55"/>
    <text x="{x + width / 2:.1f}" y="{y - 11:.1f}" text-anchor="middle" class="pipeline-small">{total_count} cross-card relationships summarized</text>
    {segment('entry', 'core')}
    {segment('core', 'support')}
    <circle cx="{center('entry')[0]:.1f}" cy="{y:.1f}" r="15" fill="#d05a2f"/>
    <circle cx="{center('core')[0]:.1f}" cy="{y:.1f}" r="15" fill="#287c91"/>
    <circle cx="{center('support')[0]:.1f}" cy="{y:.1f}" r="15" fill="#4e8b58"/>
    <text x="{center('entry')[0]:.1f}" y="{y + 4:.1f}" text-anchor="middle" class="pipeline-number">1</text>
    <text x="{center('core')[0]:.1f}" y="{y + 4:.1f}" text-anchor="middle" class="pipeline-number">2</text>
    <text x="{center('support')[0]:.1f}" y="{y + 4:.1f}" text-anchor="middle" class="pipeline-number">3</text>
    <text x="{center('entry')[0]:.1f}" y="{y + 29:.1f}" text-anchor="middle" class="pipeline-label">entrypoints</text>
    <text x="{center('core')[0]:.1f}" y="{y + 29:.1f}" text-anchor="middle" class="pipeline-label">core modules</text>
    <text x="{center('support')[0]:.1f}" y="{y + 29:.1f}" text-anchor="middle" class="pipeline-label">repo structure</text>
  </g>
'''


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
    cards, _scores = _build_overview_cards(nodes_by_id, edges, entrypoints, key_modules, node_limit=node_limit)
    card_positions, columns = _layout_cards(cards, x=graph_x, y=graph_y, width=graph_width, height=graph_height)
    visible_cards = [card for card in cards if str(card['id']) in card_positions]
    visible_edges = _overview_edges(visible_cards, card_positions, nodes_by_id, edges, limit=16)

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
    .tiny {{ font: 700 9px ui-sans-serif, system-ui, sans-serif; fill: #69766e; }}
    .pipeline-label {{ font: 800 10px ui-sans-serif, system-ui, sans-serif; fill: {theme['muted']}; letter-spacing: .04em; }}
    .pipeline-small {{ font: 700 9px ui-sans-serif, system-ui, sans-serif; fill: {theme['muted']}; }}
    .pipeline-number {{ font: 900 11px ui-sans-serif, system-ui, sans-serif; fill: #ffffff; }}
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
        f'<text x="58" y="154" class="group-label">ENTRYPOINTS</text>',
    ]
    cursor_y = 178
    for item in [str(entry.get('path') or entry.get('label') or entry.get('id')) for entry in entrypoints[:6]]:
        sidebar_lines.append(f'<text x="58" y="{cursor_y}" class="small">- {escape(_shorten(item, 34))}</text>')
        cursor_y += 19
    if not entrypoints:
        sidebar_lines.append(f'<text x="58" y="{cursor_y}" class="small">No deterministic entrypoint found</text>')
        cursor_y += 19
    cursor_y += 18
    sidebar_lines.append(f'<text x="58" y="{cursor_y}" class="group-label">KEY MODULES</text>')
    cursor_y += 24
    for item in [str(module.get('path') or module.get('label') or module.get('id')) for module in key_modules[:7]]:
        sidebar_lines.append(f'<text x="58" y="{cursor_y}" class="small">- {escape(_shorten(item, 34))}</text>')
        cursor_y += 19
    if not key_modules:
        sidebar_lines.append(f'<text x="58" y="{cursor_y}" class="small">No key module score yet</text>')
        cursor_y += 19
    sidebar_lines.append(f'<text x="58" y="{height - 78}" class="small">{escape(str(payload.get("mode") or "fixed"))} share - README-safe SVG</text>')
    sidebar_lines.append(f'<text x="58" y="{height - 56}" class="small">Collapsed symbols open in the app</text>')

    sidebar = f'''
  <rect x="34" y="116" width="276" height="{height - 148}" rx="26" fill="{theme['panel']}" opacity=".92" filter="url(#soft-shadow)"/>
  <circle cx="72" cy="104" r="18" fill="{theme['accent']}"/>
  <text x="66" y="111" font-size="18" font-weight="900" fill="#fff">G</text>
  {''.join(sidebar_lines)}
'''

    column_svg = []
    for column in columns:
        column_svg.append(
            f'<rect x="{column["x"]:.1f}" y="{column["y"]:.1f}" width="{column["width"]:.1f}" height="{column["height"]:.1f}" rx="26" fill="{theme["panel"]}" opacity=".46"/>'
            f'<text x="{column["x"] + 18:.1f}" y="{column["y"] + 29:.1f}" class="column-label">{escape(str(column["label"]).upper())}</text>'
            f'<text x="{column["x"] + column["width"] - 22:.1f}" y="{column["y"] + 29:.1f}" text-anchor="end" class="small">{column["count"]} cards</text>'
        )

    relationship_pipeline = _render_relationship_pipeline(
        columns,
        visible_cards,
        visible_edges,
        theme,
        x=graph_x,
        width=graph_width,
        y=118,
    )

    card_svg = []
    for card in visible_cards:
        card_svg.append(_render_card(card, card_positions[str(card['id'])], theme))

    legend_y = height - 34
    legend = f'''
  <text x="{graph_x}" y="{legend_y}" class="small">Overview graph: {len(visible_cards)} cards, {len(visible_edges)} aggregated relationships. Builtins, external calls, and {hidden_nodes} low-signal nodes are collapsed.</text>
'''

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(repo)} GitStarter codebase graph">
{defs}
{styles}
{header}
{sidebar}
{relationship_pipeline}
  <g>{''.join(column_svg)}</g>
  <g>{''.join(card_svg)}</g>
{legend}
</svg>
'''
