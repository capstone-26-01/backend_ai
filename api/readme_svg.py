from __future__ import annotations

from collections import defaultdict
from html import escape
from math import ceil, cos, pi, sin, sqrt
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


def _select_nodes(
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    edges: list[Mapping[str, Any]],
    entrypoint_ids: set[str],
    key_module_ids: set[str],
    node_limit: int,
) -> tuple[list[str], dict[str, int]]:
    scores = _build_scores(nodes_by_id, edges, entrypoint_ids, key_module_ids)
    selected: set[str] = set()
    seed_ids = [node_id for node_id in [*entrypoint_ids, *key_module_ids] if node_id in nodes_by_id]
    selected.update(seed_ids)

    for edge in edges:
        source = str(edge.get('source') or '')
        target = str(edge.get('target') or '')
        if _edge_kind(edge) in RELATIONSHIP_KINDS and (source in selected or target in selected):
            if source in nodes_by_id:
                selected.add(source)
            if target in nodes_by_id:
                selected.add(target)

    for node_id, node in nodes_by_id.items():
        if _node_kind(node) == 'directory' and _node_path(node) and '/' not in _node_path(node):
            selected.add(node_id)

    ranked = sorted(nodes_by_id, key=lambda node_id: (-scores.get(node_id, 0), _node_path(nodes_by_id[node_id]), node_id))
    for node_id in ranked:
        if len(selected) >= node_limit:
            break
        selected.add(node_id)

    changed = True
    while changed and len(selected) < node_limit:
        changed = False
        for node_id in list(selected):
            parent_id = _node_parent(nodes_by_id[node_id])
            if parent_id and parent_id in nodes_by_id and parent_id not in selected:
                selected.add(parent_id)
                changed = True
                if len(selected) >= node_limit:
                    break

    ordered = sorted(selected, key=lambda node_id: (-scores.get(node_id, 0), NODE_KIND_PRIORITY.get(_node_kind(nodes_by_id[node_id]), 9), _node_path(nodes_by_id[node_id]), node_id))
    return ordered[:node_limit], scores


def _layout_nodes(
    selected_ids: list[str],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, int],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[dict[str, tuple[float, float]], list[dict[str, Any]]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for node_id in selected_ids:
        groups[_top_group(nodes_by_id[node_id])].append(node_id)

    group_items = sorted(
        groups.items(),
        key=lambda item: (-max(scores.get(node_id, 0) for node_id in item[1]), item[0]),
    )
    group_count = max(1, len(group_items))
    columns = max(1, min(4, ceil(sqrt(group_count * width / max(height, 1)))))
    rows = ceil(group_count / columns)
    cell_width = width / columns
    cell_height = height / rows

    positions: dict[str, tuple[float, float]] = {}
    group_boxes: list[dict[str, Any]] = []
    for index, (group_name, node_ids) in enumerate(group_items):
        column = index % columns
        row = index // columns
        cell_x = x + column * cell_width
        cell_y = y + row * cell_height
        center_x = cell_x + cell_width / 2
        center_y = cell_y + cell_height / 2 + 8
        radius = max(34, min(cell_width, cell_height) * 0.34)
        ordered_nodes = sorted(node_ids, key=lambda node_id: (-scores.get(node_id, 0), NODE_KIND_PRIORITY.get(_node_kind(nodes_by_id[node_id]), 9), node_id))

        group_boxes.append({
            'name': group_name,
            'x': cell_x + 8,
            'y': cell_y + 8,
            'width': max(10, cell_width - 16),
            'height': max(10, cell_height - 16),
            'count': len(node_ids),
        })

        for node_index, node_id in enumerate(ordered_nodes):
            if node_index == 0:
                positions[node_id] = (center_x, center_y)
                continue
            ring = 1 + int(sqrt(node_index - 1) / 2)
            ring_position = node_index - 1
            angle = (ring_position * 137.508 / 180) * pi
            ring_radius = min(radius, 30 + ring * 27)
            node_x = center_x + cos(angle) * ring_radius
            node_y = center_y + sin(angle) * ring_radius
            positions[node_id] = (
                max(cell_x + 30, min(cell_x + cell_width - 30, node_x)),
                max(cell_y + 42, min(cell_y + cell_height - 28, node_y)),
            )

    return positions, group_boxes


def _node_radius(kind: str, score: int, is_seed: bool) -> int:
    base = {
        'directory': 13,
        'file': 11,
        'module': 16,
        'class': 14,
        'function': 12,
        'method': 11,
        'external': 9,
    }.get(kind, 10)
    if is_seed:
        base += 4
    elif score > 70:
        base += 2
    return base


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
    entrypoint_ids = {str(item.get('id')) for item in entrypoints if item.get('id')}
    key_module_ids = {str(item.get('id')) for item in key_modules if item.get('id')}

    selected_ids, scores = _select_nodes(nodes_by_id, edges, entrypoint_ids, key_module_ids, node_limit)
    selected = set(selected_ids)
    graph_x = 344
    graph_y = 126
    graph_width = width - graph_x - 36
    graph_height = height - graph_y - 50
    positions, group_boxes = _layout_nodes(selected_ids, nodes_by_id, scores, x=graph_x, y=graph_y, width=graph_width, height=graph_height)

    visible_edges = [
        edge
        for edge in edges
        if str(edge.get('source') or '') in selected and str(edge.get('target') or '') in selected
    ]
    visible_edges.sort(key=lambda edge: (_edge_kind(edge) not in RELATIONSHIP_KINDS, _safe_edge_id(edge, 0)))
    visible_edges = visible_edges[: min(260, max(60, node_limit * 3))]

    hidden_nodes = max(0, len(nodes_by_id) - len(selected_ids))
    hidden_edges = max(0, len(edges) - len(visible_edges))
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
  </defs>'''

    styles = f'''
  <style>
    .title {{ font: 800 34px Georgia, 'Times New Roman', serif; fill: {theme['text']}; }}
    .subtitle {{ font: 500 14px ui-sans-serif, system-ui, sans-serif; fill: {theme['muted']}; }}
    .label {{ font: 700 11px ui-sans-serif, system-ui, sans-serif; fill: {theme['text']}; paint-order: stroke; stroke: {theme['panel']}; stroke-width: 4px; stroke-linejoin: round; }}
    .small {{ font: 600 10px ui-sans-serif, system-ui, sans-serif; fill: {theme['muted']}; }}
    .group-label {{ font: 800 12px ui-sans-serif, system-ui, sans-serif; fill: {theme['muted']}; letter-spacing: .02em; }}
    .edge {{ fill: none; stroke: {theme['line']}; stroke-width: 1.3; opacity: .38; }}
    .edge.rel {{ stroke: {theme['relationship']}; stroke-width: 1.8; opacity: .68; stroke-dasharray: 7 6; animation: flow 20s linear infinite; }}
    .seed {{ animation: pulse 3.2s ease-in-out infinite; }}
    @keyframes flow {{ to {{ stroke-dashoffset: -180; }} }}
    @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .72; }} }}
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
  {_render_stat('shown', len(selected_ids), width - 190, stat_y, theme)}
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
    sidebar_lines.append(f'<text x="58" y="{height - 56}" class="small">Click image to open interactive map</text>')

    sidebar = f'''
  <rect x="34" y="116" width="276" height="{height - 148}" rx="26" fill="{theme['panel']}" opacity=".92" filter="url(#soft-shadow)"/>
  <circle cx="72" cy="104" r="18" fill="{theme['accent']}"/>
  <text x="66" y="111" font-size="18" font-weight="900" fill="#fff">G</text>
  {''.join(sidebar_lines)}
'''

    group_svg = []
    for box in group_boxes:
        group_svg.append(
            f'<rect x="{box["x"]:.1f}" y="{box["y"]:.1f}" width="{box["width"]:.1f}" height="{box["height"]:.1f}" rx="24" fill="{theme["panel"]}" opacity=".42"/>'
            f'<text x="{box["x"] + 18:.1f}" y="{box["y"] + 25:.1f}" class="group-label">{escape(_shorten(str(box["name"]), 20))} - {box["count"]}</text>'
        )

    edge_svg = []
    for index, edge in enumerate(visible_edges):
        source = str(edge.get('source') or '')
        target = str(edge.get('target') or '')
        if source not in positions or target not in positions:
            continue
        sx, sy = positions[source]
        tx, ty = positions[target]
        curve = max(24, abs(tx - sx) * 0.22)
        class_name = 'edge rel' if _edge_kind(edge) in RELATIONSHIP_KINDS else 'edge'
        edge_svg.append(
            f'<path class="{class_name}" d="M {sx:.1f} {sy:.1f} C {sx + curve:.1f} {sy:.1f}, {tx - curve:.1f} {ty:.1f}, {tx:.1f} {ty:.1f}"/>'
        )

    node_svg = []
    labeled = 0
    for node_id in selected_ids:
        node = nodes_by_id[node_id]
        x_pos, y_pos = positions[node_id]
        kind = _node_kind(node)
        stroke, fill = NODE_COLORS.get(kind, ('#777', '#fff'))
        is_seed = node_id in entrypoint_ids or node_id in key_module_ids
        radius = _node_radius(kind, scores.get(node_id, 0), is_seed)
        label = _shorten(_node_label(node), 22)
        node_class = 'seed' if is_seed else ''
        node_svg.append(
            f'<g class="{node_class}">'
            f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{3 if is_seed else 2}"/>'
            f'<circle cx="{x_pos - radius / 3:.1f}" cy="{y_pos - radius / 3:.1f}" r="{max(2, radius / 5):.1f}" fill="{stroke}" opacity=".35"/>'
            f'</g>'
        )
        should_label = is_seed or kind in {'directory', 'module', 'class'} or labeled < 54
        if should_label:
            node_svg.append(f'<text x="{x_pos + radius + 5:.1f}" y="{y_pos + 4:.1f}" class="label">{escape(label)}</text>')
            labeled += 1

    legend_x = width - 582
    legend_y = height - 34
    legend = f'''
  <g opacity=".94">
    <text x="{legend_x}" y="{legend_y}" class="small">directory</text>
    <text x="{legend_x + 92}" y="{legend_y}" class="small">module</text>
    <text x="{legend_x + 168}" y="{legend_y}" class="small">class</text>
    <text x="{legend_x + 232}" y="{legend_y}" class="small">function</text>
    <text x="{legend_x + 328}" y="{legend_y}" class="small">animated dashed lines = code relationships</text>
  </g>
  <text x="{graph_x}" y="{height - 34}" class="small">Overview graph: {len(selected_ids)} of {len(nodes_by_id)} nodes, {len(visible_edges)} of {len(edges)} edges rendered. {hidden_edges} edges hidden for README readability.</text>
'''

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(repo)} GitStarter codebase graph">
{defs}
{styles}
{header}
{sidebar}
  <g>{''.join(group_svg)}</g>
  <g>{''.join(edge_svg)}</g>
  <g>{''.join(node_svg)}</g>
{legend}
</svg>
'''
