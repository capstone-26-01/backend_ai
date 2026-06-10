import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import fs from "node:fs";
import path from "node:path";
import { Type } from "typebox";

type IssueMapJob = {
  job_id?: string;
  repo?: {
    local_path?: string;
    language?: string;
  };
  artifact?: {
    nodes?: Array<{ id: string; path: string }>;
    edges?: Array<{ source: string; target: string; type?: string }>;
  };
  issue?: {
    title?: string;
    body?: string;
    comments?: Array<{ body?: string }>;
  };
};

type ToolCall = {
  name: string;
  arguments: unknown;
};

const GENERIC_SYMBOLS = new Set(["analysis", "view", "views", "service", "services"]);
const NEGATIVE_EVIDENCE = [
  "do not inspect",
  "do not include",
  "don't inspect",
  "don't include",
  "never enters",
  "never enter",
  "does not enter",
  "not enter",
  "not part",
  "should not",
  "different request",
  "ignore",
  "avoid",
  "unrelated",
];
const toolCalls: ToolCall[] = [];

function loadJob(): IssueMapJob {
  return JSON.parse(process.env.HARNESS_EVAL_JOB || "{}");
}

function nodeById(job: IssueMapJob, nodeId: string) {
  return (job.artifact?.nodes || []).find((node) => node.id === nodeId)
    || selectedNodes(job).find((node) => node.node_id === nodeId);
}

function issueText(job: IssueMapJob): string {
  return [
    job.issue?.title || "",
    job.issue?.body || "",
    ...(job.issue?.comments || []).map((comment) => comment.body || ""),
  ].join("\n").toLowerCase();
}

function includesToken(text: string, token: string): boolean {
  const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[^a-zA-Z0-9_])${escaped}([^a-zA-Z0-9_]|$)`).test(text);
}

function isNegatedMention(text: string, needle: string): boolean {
  if (!needle) return false;
  const loweredNeedle = needle.toLowerCase();
  let index = text.indexOf(loweredNeedle);
  let seen = false;
  let hasPositiveMention = false;
  while (index >= 0) {
    seen = true;
    const before = text.slice(Math.max(0, index - 48), index);
    const after = text.slice(index + loweredNeedle.length, index + loweredNeedle.length + 48);
    const context = `${before} ${after}`;
    if (!NEGATIVE_EVIDENCE.some((phrase) => context.includes(phrase))) {
      hasPositiveMention = true;
    }
    index = text.indexOf(loweredNeedle, index + loweredNeedle.length);
  }
  return seen && !hasPositiveMention;
}

function rankNodes(job: IssueMapJob) {
  if (job.repo?.local_path) return rankRepoSymbols(job);
  const text = issueText(job);
  const candidates = (job.artifact?.nodes || []).map((node) => {
    const id = String(node.id || "");
    const path = String(node.path || "");
    const symbol = id.includes("::") ? id.split("::").slice(-1)[0] : id;
    const basename = path.split("/").slice(-1)[0];
    const negated = [id, path, symbol].some((value) => isNegatedMention(text, value));
    let score = 0.1;
    if (!negated) {
      if (path && text.includes(path.toLowerCase())) score += 0.45;
      if (basename && includesToken(text, basename.toLowerCase())) score += 0.25;
      if (symbol && !GENERIC_SYMBOLS.has(symbol.toLowerCase()) && includesToken(text, symbol.toLowerCase())) score += 0.55;
    }
    return { node_id: id, path, score: Math.min(1, score) };
  });
  candidates.sort((left, right) => right.score - left.score || left.node_id.localeCompare(right.node_id));
  return candidates;
}

function selectedNodes(job: IssueMapJob) {
  const ranked = rankNodes(job);
  const strong = ranked.filter((candidate) => candidate.score >= 0.5);
  return strong.length > 0 ? strong : ranked.slice(0, 2);
}

function repoRoot(job: IssueMapJob): string {
  const localPath = job.repo?.local_path || "";
  const root = path.resolve(process.cwd(), localPath);
  if (!root.startsWith(process.cwd())) {
    throw new Error("repo local_path must stay inside the workspace");
  }
  return root;
}

function safeRelativePath(value: string): string {
  if (!value || path.isAbsolute(value) || value.includes("..")) {
    throw new Error("repo path must be a safe relative path");
  }
  return value;
}

function listPythonFiles(root: string): string[] {
  const result: string[] = [];
  const ignored = new Set([".git", "__pycache__", "venv", ".venv", "node_modules"]);
  function visit(directory: string) {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      if (ignored.has(entry.name)) continue;
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        visit(absolute);
      } else if (entry.isFile() && entry.name.endsWith(".py")) {
        result.push(path.relative(root, absolute).split(path.sep).join("/"));
      }
    }
  }
  visit(root);
  return result.sort();
}

function readRepoFileText(job: IssueMapJob, relativePath: string): string {
  const root = repoRoot(job);
  const safePath = safeRelativePath(relativePath);
  const absolute = path.resolve(root, safePath);
  if (!absolute.startsWith(root + path.sep)) {
    throw new Error("repo file path escaped repo root");
  }
  return fs.readFileSync(absolute, "utf8");
}

function extractSymbols(relativePath: string, text: string) {
  const symbols: Array<{ node_id: string; path: string; name: string; kind: string; line: number }> = [];
  text.split(/\r?\n/).forEach((line, index) => {
    const match = line.match(/^\s*(async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b/);
    if (!match) return;
    symbols.push({
      node_id: `${relativePath}::${match[2]}`,
      path: relativePath,
      name: match[2],
      kind: match[1] === "class" ? "class" : "function",
      line: index + 1,
    });
  });
  return symbols;
}

function rankRepoSymbols(job: IssueMapJob, query = "") {
  const text = `${issueText(job)}\n${query}`.toLowerCase();
  const root = repoRoot(job);
  const candidates = listPythonFiles(root).flatMap((relativePath) => {
    const fileText = readRepoFileText(job, relativePath);
    return extractSymbols(relativePath, fileText).map((symbol) => {
      const basename = symbol.path.split("/").slice(-1)[0];
      const negated = [symbol.node_id, symbol.path, symbol.name].some((value) => isNegatedMention(text, value));
      let score = 0.1;
      if (!negated) {
        if (symbol.path && text.includes(symbol.path.toLowerCase())) score += 0.25;
        if (basename && includesToken(text, basename.toLowerCase())) score += 0.1;
        if (symbol.name && !GENERIC_SYMBOLS.has(symbol.name.toLowerCase()) && includesToken(text, symbol.name.toLowerCase())) score += 0.65;
      }
      return { ...symbol, score: Math.min(1, score) };
    });
  });
  candidates.sort((left, right) => right.score - left.score || left.node_id.localeCompare(right.node_id));
  return candidates;
}

function validNodeIds(job: IssueMapJob): Set<string> {
  return new Set(validNodePathMap(job).keys());
}

function validNodePathMap(job: IssueMapJob): Map<string, string> {
  if (job.repo?.local_path) {
    const root = repoRoot(job);
    return new Map(
      listPythonFiles(root).flatMap((relativePath) =>
        extractSymbols(relativePath, readRepoFileText(job, relativePath)).map((symbol) => [symbol.node_id, symbol.path] as [string, string])
      )
    );
  }
  return new Map((job.artifact?.nodes || []).map((node) => [node.id, node.path]));
}

const listRepoFiles = defineTool({
  name: "list_repo_files",
  label: "List Repo Files",
  description: "List bounded Python files in the provided local repository fixture.",
  promptSnippet: "List Python files available in the selected repository",
  parameters: Type.Object({}),
  async execute(_toolCallId, params) {
    toolCalls.push({ name: "list_repo_files", arguments: params });
    const job = loadJob();
    const files = listPythonFiles(repoRoot(job));
    return {
      content: [{ type: "text", text: JSON.stringify({ files }) }],
      details: { files },
    };
  },
});

const searchRepoSymbols = defineTool({
  name: "search_repo_symbols",
  label: "Search Repo Symbols",
  description: "Search bounded repository symbols and rank them against the issue text.",
  promptSnippet: "Search Python functions and classes in the selected repository",
  parameters: Type.Object({
    query: Type.Optional(Type.String()),
  }),
  async execute(_toolCallId, params) {
    toolCalls.push({ name: "search_repo_symbols", arguments: params });
    const job = loadJob();
    const candidates = rankRepoSymbols(job, String(params.query || "")).slice(0, 12);
    return {
      content: [{ type: "text", text: JSON.stringify({ candidates }) }],
      details: { candidates },
    };
  },
});

const readRepoFile = defineTool({
  name: "read_repo_file",
  label: "Read Repo File",
  description: "Read a bounded Python file from the provided repository fixture.",
  promptSnippet: "Read a bounded repository file for code context",
  parameters: Type.Object({
    path: Type.String(),
  }),
  async execute(_toolCallId, params) {
    toolCalls.push({ name: "read_repo_file", arguments: params });
    const job = loadJob();
    const safePath = safeRelativePath(params.path);
    const text = readRepoFileText(job, safePath);
    return {
      content: [{ type: "text", text: JSON.stringify({ path: safePath, text: text.slice(0, 4000) }) }],
      details: { path: safePath, text: text.slice(0, 4000) },
    };
  },
});

const readNodeContext = defineTool({
  name: "read_node_context",
  label: "Read Node Context",
  description: "Read bounded source context for a repository fixture symbol node.",
  promptSnippet: "Inspect one exact candidate node with surrounding source lines",
  parameters: Type.Object({
    node_id: Type.String(),
    before: Type.Optional(Type.Number()),
    after: Type.Optional(Type.Number()),
  }),
  async execute(_toolCallId, params) {
    toolCalls.push({ name: "read_node_context", arguments: params });
    const job = loadJob();
    const nodeId = String(params.node_id || "");
    const node = rankRepoSymbols(job).find((candidate) => candidate.node_id === nodeId);
    if (!node) {
      const result = { error: "node_not_found", node_id: nodeId };
      return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
    }
    const before = Math.max(0, Math.min(80, Number(params.before ?? 8)));
    const after = Math.max(0, Math.min(80, Number(params.after ?? 20)));
    const text = readRepoFileText(job, node.path);
    const lines = text.split(/\r?\n/);
    const start = Math.max(1, Number(node.line || 1) - before);
    const end = Math.min(lines.length, Number(node.line || 1) + after);
    const excerpt = lines.slice(start - 1, end).map((line, index) => `${start + index}: ${line}`).join("\n").slice(0, 4000);
    const result = {
      node: { id: node.node_id, kind: node.kind, label: node.name, path: node.path, start_line: node.line, end_line: node.line },
      context: { path: node.path, start_line: start, end_line: end, excerpt },
      neighbors: { incoming: [], outgoing: [], nodes: [] },
      warnings: [],
    };
    return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
  },
});

const rankIssueCandidates = defineTool({
  name: "rank_issue_candidates",
  label: "Rank Issue Candidates",
  description: "Return candidate repo graph nodes for the selected GitHub issue from the provided issue-map job.",
  promptSnippet: "Rank likely issue origin nodes from the bounded issue-map artifact",
  parameters: Type.Object({
    query: Type.Optional(Type.String()),
  }),
  async execute(_toolCallId, params) {
    toolCalls.push({ name: "rank_issue_candidates", arguments: params });
    const job = loadJob();
    const result = { candidates: rankNodes(job).slice(0, 3) };
    return {
      content: [{ type: "text", text: JSON.stringify(result) }],
      details: result,
    };
  },
});

const loadFocusGraph = defineTool({
  name: "load_focus_graph",
  label: "Load Focus Graph",
  description: "Return a bounded focus graph for selected node IDs from the issue-map artifact.",
  promptSnippet: "Load graph neighbors for selected candidate node IDs",
  parameters: Type.Object({
    node_ids: Type.Array(Type.String()),
  }),
  async execute(_toolCallId, params) {
    toolCalls.push({ name: "load_focus_graph", arguments: params });
    const job = loadJob();
    const selected = new Set(params.node_ids || []);
    const nodes = (job.artifact?.nodes || []).filter((node) => selected.has(node.id));
    const edges = (job.artifact?.edges || []).filter((edge) => selected.has(edge.source) || selected.has(edge.target));
    const result = { nodes, edges };
    return {
      content: [{ type: "text", text: JSON.stringify(result) }],
      details: result,
    };
  },
});

const loadCodeContext = defineTool({
  name: "load_code_context",
  label: "Load Code Context",
  description: "Return bounded synthetic code context for selected candidate paths.",
  promptSnippet: "Load bounded code context for selected candidate paths",
  parameters: Type.Object({
    paths: Type.Array(Type.String()),
  }),
  async execute(_toolCallId, params) {
    toolCalls.push({ name: "load_code_context", arguments: params });
    const job = loadJob();
    const artifactPaths = new Set((job.artifact?.nodes || []).map((node) => node.path));
    const excerpts = (params.paths || [])
      .filter((path) => artifactPaths.has(path))
      .map((path) => ({ path, excerpt: `Bounded context for ${path}` }));
    const result = { excerpts };
    return {
      content: [{ type: "text", text: JSON.stringify(result) }],
      details: result,
    };
  },
});

const searchRepoText = defineTool({
  name: "search_repo_text",
  label: "Search Repo Text",
  description: "Search text across bounded Python files in the provided repository fixture.",
  promptSnippet: "Search code text for symptoms, log strings, and output keys",
  parameters: Type.Object({
    query: Type.String(),
  }),
  async execute(_toolCallId, params) {
    toolCalls.push({ name: "search_repo_text", arguments: params });
    const job = loadJob();
    const root = repoRoot(job);
    const query = String(params.query || "").toLowerCase();
    const terms = query.split(/[^a-zA-Z0-9_]+/).filter((term) => term.length >= 3);
    const matches = listPythonFiles(root).flatMap((relativePath) => {
      const text = readRepoFileText(job, relativePath);
      return text.split(/\r?\n/).flatMap((line, index) => {
        const lowered = line.toLowerCase();
        if (!terms.some((term) => lowered.includes(term))) return [];
        return [{ path: relativePath, line: index + 1, text: line.trim() }];
      });
    }).slice(0, 20);
    return {
      content: [{ type: "text", text: JSON.stringify({ matches }) }],
      details: { matches },
    };
  },
});

const finishIssueMapTranscript = defineTool({
  name: "finish_issue_map_transcript",
  label: "Finish Issue Map Transcript",
  description: "Emit the final issue-map transcript JSON. Use this as the final action.",
  promptSnippet: "Finish with the final issue-map transcript JSON",
  promptGuidelines: [
    "Use finish_issue_map_transcript as the final action after issue-map tools are called.",
    "After calling finish_issue_map_transcript, do not emit another assistant response in the same turn.",
  ],
  parameters: Type.Object({
    hypotheses: Type.Array(Type.Object({
      node_id: Type.String(),
      confidence: Type.Number(),
      rationale: Type.String(),
    })),
    investigation_path: Type.Array(Type.Object({
      node_id: Type.String(),
      path: Type.String(),
      why: Type.String(),
    })),
    confidence: Type.Object({
      score: Type.Number(),
      rationale: Type.String(),
    }),
  }),
  async execute(_toolCallId, params) {
    toolCalls.push({ name: "finish_issue_map_transcript", arguments: params });
    const job = loadJob();
    const nodePaths = validNodePathMap(job);
    const valid = new Set(nodePaths.keys());
    const submittedIds = [
      ...params.hypotheses.map((hypothesis) => hypothesis.node_id),
      ...params.investigation_path.map((step) => step.node_id),
    ];
    const invalidIds = Array.from(new Set(submittedIds.filter((nodeId) => nodeId && !valid.has(nodeId))));
    if (invalidIds.length > 0) {
      const error = {
        error: "invalid_node_ids",
        invalid_node_ids: invalidIds,
        valid_node_id_examples: Array.from(valid).slice(0, 16),
        instruction: "Retry finish_issue_map_transcript using exact node_id values returned by search_repo_symbols. Do not use file paths or Class.method spellings as node_id values.",
      };
      return {
        content: [{ type: "text", text: JSON.stringify(error) }],
        details: error,
      };
    }
    const text = issueText(job);
    const negatedIds = Array.from(new Set(submittedIds.filter((nodeId) => {
      if (!nodeId) return false;
      const symbol = nodeId.includes("::") ? nodeId.split("::").slice(-1)[0] : nodeId;
      return [nodeId, symbol].some((value) => isNegatedMention(text, value));
    })));
    if (negatedIds.length > 0) {
      const error = {
        error: "negated_node_ids",
        negated_node_ids: negatedIds,
        instruction: "Retry finish_issue_map_transcript without node_id values the issue describes as unrelated, from another request, or not part of this failure.",
      };
      return {
        content: [{ type: "text", text: JSON.stringify(error) }],
        details: error,
      };
    }
    const validPaths = new Set(nodePaths.values());
    const invalidPaths = Array.from(new Set(params.investigation_path.map((step) => step.path).filter((filePath) => filePath && !validPaths.has(filePath))));
    const mismatchedPaths = params.investigation_path
      .filter((step) => nodePaths.has(step.node_id) && nodePaths.get(step.node_id) !== step.path)
      .map((step) => ({ node_id: step.node_id, path: step.path, expected_path: nodePaths.get(step.node_id) }));
    if (invalidPaths.length > 0 || mismatchedPaths.length > 0) {
      const error = {
        error: "invalid_paths",
        invalid_paths: invalidPaths,
        mismatched_paths: mismatchedPaths,
        valid_path_examples: Array.from(validPaths).slice(0, 16),
        instruction: "Retry finish_issue_map_transcript using exact repository-relative file paths that match each node_id.",
      };
      return {
        content: [{ type: "text", text: JSON.stringify(error) }],
        details: error,
      };
    }
    const transcript = {
      sample_id: job.job_id,
      variant_id: "pi-opencode-kimi-k25-issue-tools",
      tool_calls: toolCalls.filter((call) => call.name !== "finish_issue_map_transcript"),
      final: {
        hypotheses: params.hypotheses,
        investigation_path: params.investigation_path,
        confidence: params.confidence,
      },
    };
    return {
      content: [{ type: "text", text: JSON.stringify(transcript) }],
      details: transcript,
      terminate: true,
    };
  },
});

export default function (pi: ExtensionAPI) {
  pi.registerTool(listRepoFiles);
  pi.registerTool(searchRepoSymbols);
  pi.registerTool(searchRepoText);
  pi.registerTool(readRepoFile);
  pi.registerTool(readNodeContext);
  pi.registerTool(rankIssueCandidates);
  pi.registerTool(loadFocusGraph);
  pi.registerTool(loadCodeContext);
  pi.registerTool(finishIssueMapTranscript);
}
