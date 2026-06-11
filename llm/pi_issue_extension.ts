import { defineTool, type AgentToolResult, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import fs from "node:fs";
import path from "node:path";
import { Type } from "typebox";

type IssueHarnessJob = {
  job_id?: string;
  repo?: {
    language?: string;
    primary_language?: string;
    languages?: string[];
    analysis_profile?: string;
  };
  issue?: {
    title?: string;
    body?: string;
  };
  comments?: Array<{ body?: string }>;
  evidence?: Record<string, unknown>;
  seed_candidates?: Array<{ node_id?: string; path?: string; score?: number; reason?: string }>;
  graph?: {
    nodes?: Array<{
      id?: string;
      kind?: string;
      type?: string;
      label?: string;
      symbol?: string;
      path?: string;
      parent_id?: string;
      start_line?: number;
      end_line?: number;
      language?: string;
      support_level?: string;
    }>;
    edges?: Array<{ source?: string; target?: string; kind?: string; type?: string; path?: string }>;
  };
  file_contents?: Record<string, string>;
  file_manifest?: Record<string, {
    path?: string;
    language?: string;
    language_family?: string;
    support_level?: string;
    content_stored?: boolean;
    byte_size?: number;
    truncated?: boolean;
  }>;
};

type ToolCall = {
  name: string;
  arguments: unknown;
};

type GraphNode = NonNullable<NonNullable<IssueHarnessJob["graph"]>["nodes"]>[number];

const toolCalls: ToolCall[] = [];
const SOFT_FINISH_TOOL_CALLS = 12;
const MAX_TOOL_CALLS = 80;
const GENERIC_TERMS = new Set([
  "api",
  "app",
  "core",
  "data",
  "main",
  "repo",
  "service",
  "services",
  "test",
  "tests",
  "util",
  "utils",
  "view",
  "views",
]);
const POSITIVE_NEGATION_OVERRIDES = [
  /\b(?:do not|don't|should not|must not)\s+(?:ignore|exclude)\b[^.\n;:]{0,48}$/i,
  /\bexcept\b[^.\n;:]{0,32}$/i,
];
const NEGATIVE_BEFORE_MENTION = [
  /\b(?:do not|don't)\s+(?:include|inspect)\b[^.\n;:]{0,48}$/i,
  /\b(?:ignore|exclude)\b[^.\n;:]{0,24}$/i,
  /\b(?:unrelated to|not related to|not part of|different request from)\b[^.\n;:]{0,48}$/i,
];
const NEGATIVE_AFTER_MENTION = [
  /^[^.\n;:]{0,48}\b(?:unrelated|not related|not part|different request)\b/i,
  /^[^.\n;:]{0,48}\b(?:should not|must not)\b[^.\n;:]{0,32}\b(?:include|inspect|use|touch|change|be part)\b/i,
];

function loadJob(): IssueHarnessJob {
  const jobPath = process.env.ISSUE_HARNESS_JOB_FILE || "";
  if (!jobPath) throw new Error("ISSUE_HARNESS_JOB_FILE is required");
  return JSON.parse(fs.readFileSync(jobPath, "utf8"));
}

function jsonToolResult(payload: Record<string, unknown>, terminate = false): AgentToolResult<Record<string, unknown>> {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(payload) }],
    details: payload,
    ...(terminate ? { terminate: true } : {}),
  };
}

function recordToolCall(name: string, args: unknown) {
  const countedCalls = toolCalls.filter((call) => call.name !== "finish_issue_map_transcript").length;
  if (name !== "finish_issue_map_transcript" && countedCalls >= MAX_TOOL_CALLS) {
    const job = loadJob();
    return jsonToolResult({
      sample_id: job.job_id,
      variant_id: "runtime-pi-issue-harness",
      tool_calls: toolCalls.filter((call) => call.name !== "finish_issue_map_transcript"),
      final: {
        hypotheses: [],
        investigation_path: [],
        confidence: {
          level: "none",
          score: 0,
          reasons: ["tool_call_budget_exceeded"],
        },
      },
      error: "tool_call_budget_exceeded",
      max_tool_calls: MAX_TOOL_CALLS,
      attempted_tool: name,
      attempted_arguments: args,
      instruction: "Stop the run; the backend will record this as a harness contract failure.",
    }, true);
  }
  toolCalls.push({ name, arguments: args });
  return null;
}

function finishGuidance() {
  const used = toolCalls.filter((call) => call.name !== "finish_issue_map_transcript").length;
  return {
    tool_call_budget: {
      used,
      soft_finish_by: SOFT_FINISH_TOOL_CALLS,
      hard_max: MAX_TOOL_CALLS,
      instruction:
        "Do not exhaustively search. Verify the best seed/search candidates, then call finish_issue_map_transcript. If used >= soft_finish_by, finish now with the best inspected nodes.",
    },
  };
}

function issueText(job: IssueHarnessJob): string {
  return [
    job.issue?.title || "",
    job.issue?.body || "",
    ...(job.comments || []).map((comment) => comment.body || ""),
    JSON.stringify(job.evidence || {}),
  ].join("\n").toLowerCase();
}

function tokenSet(value: string): Set<string> {
  const tokens = new Set<string>();
  const normalized = value.toLowerCase().replace(/[./:-]+/g, "_");
  for (const token of normalized.split(/[^0-9a-zA-Z가-힣_]+|_+/)) {
    if (token.length > 1 && !GENERIC_TERMS.has(token)) tokens.add(token);
  }
  for (const token of value.toLowerCase().match(/[A-Za-z0-9_./:-]+|[가-힣]+/g) || []) {
    if (token.length > 1 && !GENERIC_TERMS.has(token)) tokens.add(token);
  }
  return tokens;
}

function includesToken(text: string, token: string): boolean {
  if (!token) return false;
  if (/[./:-]/.test(token)) return text.includes(token);
  const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[^a-zA-Z0-9_])${escaped}([^a-zA-Z0-9_]|$)`).test(text);
}

function isNegatedMention(text: string, value: string): boolean {
  const needle = value.toLowerCase();
  if (!needle) return false;
  let index = text.indexOf(needle);
  let seen = false;
  let positive = false;
  while (index >= 0) {
    seen = true;
    const before = text.slice(Math.max(0, index - 96), index);
    const after = text.slice(index + needle.length, index + needle.length + 96);
    const hasPositiveOverride = POSITIVE_NEGATION_OVERRIDES.some((pattern) => pattern.test(before));
    const hasNegativeScope = !hasPositiveOverride && (
      NEGATIVE_BEFORE_MENTION.some((pattern) => pattern.test(before)) ||
      NEGATIVE_AFTER_MENTION.some((pattern) => pattern.test(after))
    );
    if (!hasNegativeScope) positive = true;
    index = text.indexOf(needle, index + needle.length);
  }
  return seen && !positive;
}

function nodeById(job: IssueHarnessJob, nodeId: string) {
  return (job.graph?.nodes || []).find((node) => node.id === nodeId);
}

function nodePath(job: IssueHarnessJob, nodeId: string): string | undefined {
  return nodeById(job, nodeId)?.path;
}

function safeNodeId(job: IssueHarnessJob, rawNodeId: string): string {
  const nodeId = String(rawNodeId || "");
  if (!nodeId || path.isAbsolute(nodeId) || nodeId.includes("..") || nodeId.includes("\\") || /[\r\n]/.test(nodeId)) {
    throw new Error("node_id must be a safe graph node id");
  }
  if (!nodeById(job, nodeId)) {
    throw new Error("node_id is not available in the bounded harness job");
  }
  return nodeId;
}

function validNodeIds(job: IssueHarnessJob): Set<string> {
  return new Set((job.graph?.nodes || []).map((node) => node.id || "").filter(Boolean));
}

function validPaths(job: IssueHarnessJob): Set<string> {
  return new Set([
    ...Object.keys(job.file_contents || {}),
    ...(job.graph?.nodes || []).map((node) => node.path || "").filter(Boolean),
  ]);
}

function priorToolNames(): string[] {
  return toolCalls.filter((call) => call.name !== "finish_issue_map_transcript").map((call) => call.name);
}

function safeFilePath(job: IssueHarnessJob, rawPath: string): string {
  const filePath = String(rawPath || "");
  if (!filePath || path.isAbsolute(filePath) || filePath.includes("..") || filePath.includes("\\")) {
    throw new Error("path must be a safe repository-relative path");
  }
  if (!Object.prototype.hasOwnProperty.call(job.file_contents || {}, filePath)) {
    throw new Error("path is not available in the bounded harness job");
  }
  return filePath;
}

function scoreNode(job: IssueHarnessJob, node: GraphNode, query = "") {
  const text = `${issueText(job)}\n${query}`.toLowerCase();
  const nodeId = String(node.id || "");
  const nodePathValue = String(node.path || "");
  const label = String(node.label || "");
  const symbol = String(node.symbol || nodeId.split("::").slice(-1)[0] || "");
  const basename = nodePathValue.split("/").slice(-1)[0] || "";
  const negated = [nodeId, nodePathValue, label, symbol].some((value) => isNegatedMention(text, value));
  let score = 0.05;
  const reasons: string[] = [];
  if (!negated) {
    for (const token of tokenSet(`${nodeId} ${nodePathValue} ${label} ${symbol}`)) {
      if (includesToken(text, token)) {
        score += token.includes("/") || token.includes("::") ? 0.22 : 0.12;
        reasons.push(`issue matched ${token}`);
      }
    }
    if (nodePathValue && text.includes(nodePathValue.toLowerCase())) {
      score += 0.28;
      reasons.push(`issue mentioned ${nodePathValue}`);
    }
    if (basename && includesToken(text, basename.toLowerCase())) {
      score += 0.12;
      reasons.push(`issue mentioned ${basename}`);
    }
  }
  for (const seed of job.seed_candidates || []) {
    if (seed.node_id === nodeId) {
      score += Math.min(0.25, Number(seed.score || 0) * 0.25);
      reasons.push("deterministic seed candidate");
    }
  }
  return {
    node_id: nodeId,
    path: nodePathValue || undefined,
    kind: node.kind || node.type,
    label,
    language: node.language,
    start_line: node.start_line,
    end_line: node.end_line,
    score: Math.min(1, Number(score.toFixed(3))),
    reasons: reasons.slice(0, 6),
  };
}

const getIssueContext = defineTool({
  name: "get_issue_context",
  label: "Get Issue Context",
  description: "Read bounded issue text, comments, extracted evidence, and seed candidate hints.",
  promptSnippet: "Read bounded issue context before searching repository artifacts",
  parameters: Type.Object({}),
  async execute(_toolCallId, params) {
    const budgetError = recordToolCall("get_issue_context", params);
    if (budgetError) return budgetError;
    const job = loadJob();
    const result = {
      issue: job.issue || {},
      comments: job.comments || [],
      evidence: job.evidence || {},
      seed_candidates: job.seed_candidates || [],
      recommended_workflow: [
        "Use seed_candidates as the primary ranked shortlist.",
        "Run at most one symbol search and one text search if seed evidence is insufficient.",
        "Inspect at most two likely nodes with read_node_context or read_repo_file.",
        "Call finish_issue_map_transcript with the best 1-3 inspected origin nodes; do not keep searching for certainty.",
      ],
      ...finishGuidance(),
    };
    return jsonToolResult(result);
  },
});

function rankedSymbols(job: IssueHarnessJob, query = "") {
  return (job.graph?.nodes || [])
    .filter((node) => node.id && node.path && !["directory"].includes(String(node.kind || node.type || "")))
    .map((node) => scoreNode(job, node, query))
    .filter((candidate) => candidate.score > 0.05 || (job.seed_candidates || []).some((seed) => seed.node_id === candidate.node_id))
    .sort((left, right) => right.score - left.score || String(left.path || "").localeCompare(String(right.path || "")) || left.node_id.localeCompare(right.node_id));
}

function searchText(job: IssueHarnessJob, query: string) {
  const terms = Array.from(tokenSet(query || issueText(job))).filter((term) => term.length >= 3).slice(0, 12);
  const matches: Array<{ path: string; line: number; text: string; terms: string[] }> = [];
  for (const [filePath, fileText] of Object.entries(job.file_contents || {})) {
    const lines = String(fileText || "").split(/\r?\n/);
    lines.forEach((line, index) => {
      const lowered = line.toLowerCase();
      const hitTerms = terms.filter((term) => lowered.includes(term));
      if (hitTerms.length) {
        matches.push({ path: filePath, line: index + 1, text: line.trim().slice(0, 500), terms: hitTerms });
      }
    });
  }
  return matches.slice(0, 40);
}

function compactNode(node: GraphNode | undefined) {
  if (!node) return undefined;
  return {
    id: node.id,
    kind: node.kind || node.type,
    type: node.type,
    label: node.label,
    symbol: node.symbol,
    path: node.path,
    parent_id: node.parent_id,
    start_line: node.start_line,
    end_line: node.end_line,
    language: node.language,
    support_level: node.support_level,
  };
}

function listedFiles(job: IssueHarnessJob) {
  return Object.keys(job.file_contents || {}).sort().map((filePath) => {
    const manifest = job.file_manifest?.[filePath] || {};
    return {
      path: filePath,
      language: manifest.language ?? null,
      support_level: manifest.support_level ?? null,
      byte_size: manifest.byte_size ?? String((job.file_contents || {})[filePath] || "").length,
      truncated: Boolean(manifest.truncated),
    };
  });
}

function findContainer(job: IssueHarnessJob, node: GraphNode) {
  const nodes = job.graph?.nodes || [];
  if (node.parent_id) {
    const parent = nodes.find((candidate) => candidate.id === node.parent_id);
    if (parent) return compactNode(parent);
  }
  const fileNode = nodes.find((candidate) => candidate.path === node.path && ["file", "module"].includes(String(candidate.kind || candidate.type || "")));
  if (fileNode && fileNode.id !== node.id) return compactNode(fileNode);
  const nodeId = String(node.id || "");
  const prefixContainers = nodes
    .filter((candidate) => candidate.id && candidate.id !== node.id && nodeId.startsWith(`${candidate.id}::`))
    .sort((left, right) => String(right.id || "").length - String(left.id || "").length);
  return compactNode(prefixContainers[0]);
}

const listRepoFiles = defineTool({
  name: "list_repo_files",
  label: "List Repo Files",
  description: "List source files available in the bounded analysis artifact.",
  promptSnippet: "List bounded repository files before investigating the issue",
  parameters: Type.Object({}),
  async execute(_toolCallId, params) {
    const budgetError = recordToolCall("list_repo_files", params);
    if (budgetError) return budgetError;
    const job = loadJob();
    const files = listedFiles(job);
    return jsonToolResult({ files });
  },
});

const searchRepoSymbols = defineTool({
  name: "search_repo_symbols",
  label: "Search Repo Symbols",
  description: "Search graph nodes against issue evidence and query terms.",
  promptSnippet: "Search graph symbols using issue terms, stack traces, and failure symptoms",
  parameters: Type.Object({ query: Type.Optional(Type.String()) }),
  async execute(_toolCallId, params) {
    const budgetError = recordToolCall("search_repo_symbols", params);
    if (budgetError) return budgetError;
    const job = loadJob();
    const candidates = rankedSymbols(job, String(params.query || "")).slice(0, 20);
    return jsonToolResult({
      candidates,
      instruction: "Pick the strongest candidate(s), inspect at most two, then call finish_issue_map_transcript.",
      ...finishGuidance(),
    });
  },
});

const searchRepoText = defineTool({
  name: "search_repo_text",
  label: "Search Repo Text",
  description: "Search bounded file text for stack traces, output strings, error messages, and issue symptoms.",
  promptSnippet: "Search code text when issue symptoms are not direct symbol names",
  parameters: Type.Object({ query: Type.String() }),
  async execute(_toolCallId, params) {
    const budgetError = recordToolCall("search_repo_text", params);
    if (budgetError) return budgetError;
    const job = loadJob();
    const matches = searchText(job, String(params.query || ""));
    return jsonToolResult({
      matches,
      instruction: "Use these text matches to choose a short candidate list. Do not continue broad text search after a plausible file is found.",
      ...finishGuidance(),
    });
  },
});

const readRepoFile = defineTool({
  name: "read_repo_file",
  label: "Read Repo File",
  description: "Read a bounded file excerpt from the analysis artifact, never the filesystem.",
  promptSnippet: "Read candidate files before naming final origin nodes",
  parameters: Type.Object({
    path: Type.String(),
    start_line: Type.Optional(Type.Number()),
    end_line: Type.Optional(Type.Number()),
  }),
  async execute(_toolCallId, params) {
    const budgetError = recordToolCall("read_repo_file", params);
    if (budgetError) return budgetError;
    const job = loadJob();
    const filePath = safeFilePath(job, params.path);
    const lines = String((job.file_contents || {})[filePath] || "").split(/\r?\n/);
    const start = Math.max(1, Number(params.start_line || 1));
    const end = Math.min(lines.length, Number(params.end_line || Math.min(lines.length, start + 120)));
    const excerpt = lines.slice(start - 1, end).map((line, index) => `${start + index}: ${line}`).join("\n").slice(0, 8000);
    const result = {
      path: filePath,
      start_line: start,
      end_line: end,
      excerpt,
      instruction: "If this file plausibly explains the issue, call finish_issue_map_transcript now.",
      ...finishGuidance(),
    };
    return jsonToolResult(result);
  },
});

const readNodeContext = defineTool({
  name: "read_node_context",
  label: "Read Node Context",
  description: "Read bounded code, container, and direct graph neighbors for one exact node.",
  promptSnippet: "Use after symbol search to inspect the most relevant candidate node in context",
  parameters: Type.Object({
    node_id: Type.String(),
    before: Type.Optional(Type.Number()),
    after: Type.Optional(Type.Number()),
  }),
  async execute(_toolCallId, params) {
    const budgetError = recordToolCall("read_node_context", params);
    if (budgetError) return budgetError;
    const job = loadJob();
    const nodeId = safeNodeId(job, params.node_id);
    const node = nodeById(job, nodeId)!;
    const warnings: string[] = [];

    const before = Math.max(0, Math.min(80, Number(params.before ?? 8)));
    const after = Math.max(0, Math.min(80, Number(params.after ?? 20)));
    const nodeStart = Number(node.start_line || 0);
    const nodeEnd = Number(node.end_line || node.start_line || 0);
    const filePath = node.path || "";
    let context: Record<string, unknown> = {
      path: filePath || undefined,
      start_line: node.start_line,
      end_line: node.end_line,
      excerpt: "",
    };

    if (!filePath || !Object.prototype.hasOwnProperty.call(job.file_contents || {}, filePath)) {
      warnings.push("file_content_unavailable");
    } else if (!nodeStart || !nodeEnd) {
      warnings.push("line_range_unavailable");
      const lines = String((job.file_contents || {})[filePath] || "").split(/\r?\n/);
      const end = Math.min(lines.length, 120);
      context = {
        path: filePath,
        start_line: 1,
        end_line: end,
        excerpt: lines.slice(0, end).map((line, index) => `${index + 1}: ${line}`).join("\n").slice(0, 8000),
      };
    } else {
      const lines = String((job.file_contents || {})[filePath] || "").split(/\r?\n/);
      const start = Math.max(1, nodeStart - before);
      const end = Math.min(lines.length, nodeEnd + after);
      context = {
        path: filePath,
        start_line: start,
        end_line: end,
        excerpt: lines.slice(start - 1, end).map((line, index) => `${start + index}: ${line}`).join("\n").slice(0, 8000),
      };
    }

    const incoming = (job.graph?.edges || []).filter((edge) => edge.target === nodeId).slice(0, 50);
    const outgoing = (job.graph?.edges || []).filter((edge) => edge.source === nodeId).slice(0, 50);
    const neighborIds = new Set(
      [...incoming, ...outgoing]
        .flatMap((edge) => [edge.source || "", edge.target || ""])
        .filter((candidateId) => candidateId && candidateId !== nodeId)
    );
    const neighborNodes = (job.graph?.nodes || []).filter((candidate) => neighborIds.has(candidate.id || "")).slice(0, 50).map(compactNode);
    const result = {
      node: compactNode(node),
      context,
      container: findContainer(job, node),
      neighbors: {
        incoming,
        outgoing,
        nodes: neighborNodes,
      },
      warnings,
      instruction: "If this node plausibly explains the issue, call finish_issue_map_transcript now. Otherwise inspect one more candidate at most.",
      ...finishGuidance(),
    };
    return jsonToolResult(result);
  },
});

const getNode = defineTool({
  name: "get_node",
  label: "Get Node",
  description: "Inspect exact graph node metadata from the analysis artifact.",
  promptSnippet: "Inspect exact candidate node metadata",
  parameters: Type.Object({ node_id: Type.String() }),
  async execute(_toolCallId, params) {
    const budgetError = recordToolCall("get_node", params);
    if (budgetError) return budgetError;
    const job = loadJob();
    const node = nodeById(job, params.node_id);
    const result = node
      ? { node, instruction: "Prefer read_node_context over repeated get_node calls; finish after inspecting the best candidate.", ...finishGuidance() }
      : { error: "node_not_found", node_id: params.node_id, ...finishGuidance() };
    return jsonToolResult(result);
  },
});

const getNeighbors = defineTool({
  name: "get_neighbors",
  label: "Get Neighbors",
  description: "Inspect incoming and outgoing graph neighbors for a node.",
  promptSnippet: "Inspect graph neighbors to understand call/import/container relationships",
  parameters: Type.Object({ node_id: Type.String(), limit: Type.Optional(Type.Number()) }),
  async execute(_toolCallId, params) {
    const budgetError = recordToolCall("get_neighbors", params);
    if (budgetError) return budgetError;
    const job = loadJob();
    const limit = Math.max(1, Math.min(50, Number(params.limit || 20)));
    const edges = (job.graph?.edges || [])
      .filter((edge) => edge.source === params.node_id || edge.target === params.node_id)
      .slice(0, limit);
    const neighborIds = new Set(edges.flatMap((edge) => [edge.source || "", edge.target || ""]).filter((nodeId) => nodeId && nodeId !== params.node_id));
    const nodes = (job.graph?.nodes || []).filter((node) => neighborIds.has(node.id || ""));
    const result = {
      node_id: params.node_id,
      nodes,
      edges,
      instruction: "Use neighbors only to confirm the best origin node; then call finish_issue_map_transcript.",
      ...finishGuidance(),
    };
    return jsonToolResult(result);
  },
});

const finishIssueMapTranscript = defineTool({
  name: "finish_issue_map_transcript",
  label: "Finish Issue Map Transcript",
  description: "Return final investigated issue origin nodes and a short investigation path.",
  promptSnippet: "Finish with exact node IDs and paths after bounded tool investigation",
  promptGuidelines: [
    "Call finish_issue_map_transcript only after using search_repo_symbols or search_repo_text and reading candidate code or graph neighbors.",
    "Use exact node_id values returned by graph tools. Do not invent node IDs or file paths.",
  ],
  parameters: Type.Object({
    hypotheses: Type.Array(Type.Object({
      kind: Type.Optional(Type.String()),
      node_id: Type.String(),
      confidence: Type.Number(),
      rationale: Type.String(),
    })),
    investigation_path: Type.Array(Type.Object({
      node_id: Type.String(),
      path: Type.String(),
      action: Type.Optional(Type.String()),
      why: Type.String(),
    })),
    confidence: Type.Object({
      level: Type.Optional(Type.String()),
      score: Type.Number(),
      reasons: Type.Optional(Type.Array(Type.String())),
      rationale: Type.Optional(Type.String()),
    }),
  }),
  async execute(_toolCallId, params) {
    const budgetError = recordToolCall("finish_issue_map_transcript", params);
    if (budgetError) return budgetError;
    const job = loadJob();
    const validNodes = validNodeIds(job);
    const allNodeIds = [
      ...(params.hypotheses || []).map((hypothesis) => hypothesis.node_id),
      ...(params.investigation_path || []).map((step) => step.node_id),
    ].filter(Boolean);
    const toolNames = priorToolNames();
    if (toolNames.length === 0) {
      const error = {
        error: "missing_tool_work",
        instruction: "Call get_issue_context, list_repo_files, then search_repo_symbols or search_repo_text before finishing.",
      };
      return jsonToolResult(error);
    }
    if (!toolNames.includes("get_issue_context")) {
      const error = {
        error: "missing_issue_context",
        instruction: "Retry after reading bounded issue context with get_issue_context.",
      };
      return jsonToolResult(error);
    }
    if (!toolNames.includes("list_repo_files")) {
      const error = {
        error: "missing_file_listing",
        instruction: "Retry after listing bounded repository files with list_repo_files.",
      };
      return jsonToolResult(error);
    }
    if (!toolNames.includes("search_repo_symbols") && !toolNames.includes("search_repo_text")) {
      const error = {
        error: "missing_search",
        instruction: "Retry after searching repository symbols or text for issue evidence.",
      };
      return jsonToolResult(error);
    }
    const hasInspection =
      toolNames.includes("read_repo_file") ||
      toolNames.includes("get_neighbors") ||
      toolNames.includes("read_node_context");
    if (allNodeIds.length && !hasInspection) {
      const error = {
        error: "missing_inspection",
        instruction: "Retry after inspecting a candidate with read_node_context, read_repo_file, or get_neighbors.",
      };
      return jsonToolResult(error);
    }

    const invalidNodes = Array.from(new Set(allNodeIds.filter((nodeId) => !validNodes.has(nodeId))));
    if (invalidNodes.length) {
      const error = {
        error: "invalid_node_ids",
        invalid_node_ids: invalidNodes,
        valid_node_id_examples: Array.from(validNodes).slice(0, 20),
        instruction: "Retry finish_issue_map_transcript using exact node_id values returned by search_repo_symbols/get_node.",
      };
      return jsonToolResult(error);
    }

    const text = issueText(job);
    const negatedNodes = Array.from(new Set(allNodeIds.filter((nodeId) => {
      const symbol = nodeId.includes("::") ? nodeId.split("::").slice(-1)[0] : nodeId;
      return [nodeId, symbol].some((value) => isNegatedMention(text, value));
    })));
    if (negatedNodes.length) {
      const error = {
        error: "negated_node_ids",
        negated_node_ids: negatedNodes,
        instruction: "Retry without nodes the issue text explicitly marks unrelated or excluded.",
      };
      return jsonToolResult(error);
    }

    const paths = validPaths(job);
    const invalidPaths = Array.from(new Set((params.investigation_path || [])
      .map((step) => step.path)
      .filter((filePath) => filePath && !paths.has(filePath))));
    const mismatchedPaths = (params.investigation_path || [])
      .filter((step) => nodePath(job, step.node_id) && nodePath(job, step.node_id) !== step.path)
      .map((step) => ({ node_id: step.node_id, path: step.path, expected_path: nodePath(job, step.node_id) }));
    if (invalidPaths.length || mismatchedPaths.length) {
      const error = {
        error: "invalid_paths",
        invalid_paths: invalidPaths,
        mismatched_paths: mismatchedPaths,
        valid_path_examples: Array.from(paths).slice(0, 20),
        instruction: "Retry using repository-relative paths that match each node_id.",
      };
      return { content: [{ type: "text", text: JSON.stringify(error) }], details: error };
    }

    const transcript = {
      sample_id: job.job_id,
      variant_id: "runtime-pi-issue-harness",
      tool_calls: toolCalls.filter((call) => call.name !== "finish_issue_map_transcript"),
      final: {
        hypotheses: params.hypotheses || [],
        investigation_path: params.investigation_path || [],
        confidence: params.confidence || {},
      },
    };
    return jsonToolResult(transcript, true);
  },
});

export default function (pi: ExtensionAPI) {
  pi.registerTool(getIssueContext);
  pi.registerTool(listRepoFiles);
  pi.registerTool(searchRepoSymbols);
  pi.registerTool(searchRepoText);
  pi.registerTool(readRepoFile);
  pi.registerTool(getNode);
  pi.registerTool(getNeighbors);
  pi.registerTool(readNodeContext);
  pi.registerTool(finishIssueMapTranscript);
}
