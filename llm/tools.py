from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

from llm.context_selection import identifier_tokens, question_tokens


class ToolLimitExceeded(RuntimeError):
    pass


def _node_id(node: Mapping[str, Any]) -> str:
    return str(node.get('id', ''))


def _node_file(node: Mapping[str, Any]) -> str | None:
    file_path = node.get('path') or node.get('file')
    if isinstance(file_path, str) and file_path:
        return file_path
    return None


def _short_result(result: Any) -> Any:
    if isinstance(result, dict):
        return {key: result[key] for key in list(result)[:6]}
    if isinstance(result, list):
        return result[:5]
    return result


@dataclass
class ArtifactToolbox:
    analysis: Mapping[str, Any]
    max_tool_calls: int = 8
    max_excerpt_chars: int = 3000
    tool_trace: list[dict[str, Any]] = field(default_factory=list)

    def _record(self, tool: str, arguments: dict[str, Any], result: Any) -> Any:
        if len(self.tool_trace) >= self.max_tool_calls:
            raise ToolLimitExceeded('smolagents tool call limit exceeded')
        self.tool_trace.append(
            {
                'tool': tool,
                'arguments': arguments,
                'result': _short_result(result),
            }
        )
        return result

    @property
    def nodes_by_id(self) -> dict[str, dict[str, Any]]:
        return {
            _node_id(cast(Mapping[str, Any], node)): dict(cast(Mapping[str, Any], node))
            for node in self.analysis.get('nodes', [])
            if _node_id(cast(Mapping[str, Any], node))
        }

    def search_symbols(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        tokens = set(question_tokens(query)) | identifier_tokens(query)
        matches = []
        for node in self.nodes_by_id.values():
            node_text = ' '.join(str(node.get(key, '')) for key in ('id', 'label', 'symbol', 'path', 'file')).lower()
            score = sum(1 for token in tokens if token and token in node_text)
            if score:
                matches.append(
                    {
                        'id': _node_id(node),
                        'kind': node.get('kind') or node.get('type'),
                        'label': node.get('label'),
                        'path': _node_file(node),
                        'score': score,
                    }
                )
        matches.sort(key=lambda item: (-int(item['score']), str(item['path']), str(item['id'])))
        return self._record('search_symbols', {'query': query, 'limit': limit}, matches[:limit])

    def get_node(self, node_id: str) -> dict[str, Any]:
        node = self.nodes_by_id.get(node_id)
        if node is None:
            return self._record('get_node', {'node_id': node_id}, {'error': 'node_not_found', 'node_id': node_id})
        return self._record('get_node', {'node_id': node_id}, node)

    def get_neighbors(self, node_id: str, limit: int = 20) -> dict[str, Any]:
        if node_id not in self.nodes_by_id:
            return self._record('get_neighbors', {'node_id': node_id, 'limit': limit}, {'error': 'node_not_found', 'node_id': node_id})
        edges = []
        neighbor_ids = []
        for edge in self.analysis.get('edges', []):
            edge_mapping = cast(Mapping[str, Any], edge)
            source = str(edge_mapping.get('source', ''))
            target = str(edge_mapping.get('target', ''))
            if source == node_id or target == node_id:
                edges.append(dict(edge_mapping))
                neighbor_ids.append(target if source == node_id else source)
            if len(edges) >= limit:
                break
        result = {
            'node_id': node_id,
            'neighbors': [self.nodes_by_id[neighbor_id] for neighbor_id in neighbor_ids if neighbor_id in self.nodes_by_id],
            'edges': edges,
        }
        return self._record('get_neighbors', {'node_id': node_id, 'limit': limit}, result)

    def get_file_excerpt(self, file_path: str, start_line: int | None = None, end_line: int | None = None) -> dict[str, Any]:
        file_contents = cast(Mapping[str, str], self.analysis.get('file_contents', {}))
        code = file_contents.get(file_path)
        if code is None:
            return self._record('get_file_excerpt', {'file_path': file_path}, {'error': 'file_not_found', 'path': file_path})

        lines = code.splitlines()
        start = max(1, start_line or 1)
        end = min(len(lines), end_line or min(len(lines), start + 80))
        excerpt = '\n'.join(f'{line_number:>4}: {lines[line_number - 1]}' for line_number in range(start, end + 1))
        if len(excerpt) > self.max_excerpt_chars:
            excerpt = excerpt[:self.max_excerpt_chars]
        result = {'path': file_path, 'start_line': start, 'end_line': end, 'excerpt': excerpt}
        return self._record('get_file_excerpt', {'file_path': file_path, 'start_line': start_line, 'end_line': end_line}, result)

    def get_entrypoints(self) -> list[dict[str, Any]]:
        return self._record('get_entrypoints', {}, list(self.analysis.get('entrypoints', [])))

    def get_key_modules(self) -> list[dict[str, Any]]:
        return self._record('get_key_modules', {}, list(self.analysis.get('key_modules', [])))

    def get_summary(self, kind: str = 'repo_overview') -> dict[str, Any]:
        summaries = self.analysis.get('summaries', {})
        if not isinstance(summaries, Mapping):
            return self._record('get_summary', {'kind': kind}, {'error': 'summary_not_found', 'kind': kind})
        for summary in summaries.values():
            if isinstance(summary, Mapping) and summary.get('kind') == kind:
                return self._record('get_summary', {'kind': kind}, dict(summary))
        return self._record('get_summary', {'kind': kind}, {'error': 'summary_not_found', 'kind': kind})


def build_smolagents_tools(toolbox: ArtifactToolbox):
    from smolagents import Tool

    class SearchSymbolsTool(Tool):
        name = 'search_symbols'
        description = 'Search graph symbols by natural language or code token query.'
        inputs = {
            'query': {'type': 'string', 'description': 'Search query'},
            'limit': {'type': 'integer', 'description': 'Maximum results', 'nullable': True},
        }
        output_type = 'object'

        def forward(self, query: str, limit: int = 8):
            return toolbox.search_symbols(query, limit)

    class GetNodeTool(Tool):
        name = 'get_node'
        description = 'Return one graph node by id from the stored analysis artifact.'
        inputs = {'node_id': {'type': 'string', 'description': 'Graph node id'}}
        output_type = 'object'

        def forward(self, node_id: str):
            return toolbox.get_node(node_id)

    class GetNeighborsTool(Tool):
        name = 'get_neighbors'
        description = 'Return incoming and outgoing graph neighbors for a node id.'
        inputs = {
            'node_id': {'type': 'string', 'description': 'Graph node id'},
            'limit': {'type': 'integer', 'description': 'Maximum edges', 'nullable': True},
        }
        output_type = 'object'

        def forward(self, node_id: str, limit: int = 20):
            return toolbox.get_neighbors(node_id, limit)

    class GetFileExcerptTool(Tool):
        name = 'get_file_excerpt'
        description = 'Return a bounded excerpt from file_contents stored in the artifact. Never reads the filesystem.'
        inputs = {'file_path': {'type': 'string', 'description': 'Repository file path'}}
        output_type = 'object'

        def forward(self, file_path: str):
            return toolbox.get_file_excerpt(file_path)

    class GetEntrypointsTool(Tool):
        name = 'get_entrypoints'
        description = 'Return deterministic entrypoint metadata from the artifact.'
        inputs = {}
        output_type = 'object'

        def forward(self):
            return toolbox.get_entrypoints()

    class GetKeyModulesTool(Tool):
        name = 'get_key_modules'
        description = 'Return deterministic key module metadata from the artifact.'
        inputs = {}
        output_type = 'object'

        def forward(self):
            return toolbox.get_key_modules()

    class GetSummaryTool(Tool):
        name = 'get_summary'
        description = 'Return a cached artifact summary by kind if present.'
        inputs = {'kind': {'type': 'string', 'description': 'Summary kind such as repo_overview or onboarding_guide', 'nullable': True}}
        output_type = 'object'

        def forward(self, kind: str = 'repo_overview'):
            return toolbox.get_summary(kind)

    return [
        SearchSymbolsTool(),
        GetNodeTool(),
        GetNeighborsTool(),
        GetFileExcerptTool(),
        GetEntrypointsTool(),
        GetKeyModulesTool(),
        GetSummaryTool(),
    ]
