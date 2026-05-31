import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

type IssueMapJob = {
  job_id?: string;
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
const toolCalls: ToolCall[] = [];

function loadJob(): IssueMapJob {
  return JSON.parse(process.env.HARNESS_EVAL_JOB || "{}");
}

function nodeById(job: IssueMapJob, nodeId: string) {
  return (job.artifact?.nodes || []).find((node) => node.id === nodeId);
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

function rankNodes(job: IssueMapJob) {
  const text = issueText(job);
  const candidates = (job.artifact?.nodes || []).map((node) => {
    const id = String(node.id || "");
    const path = String(node.path || "");
    const symbol = id.includes("::") ? id.split("::").slice(-1)[0] : id;
    const basename = path.split("/").slice(-1)[0];
    let score = 0.1;
    if (path && text.includes(path.toLowerCase())) score += 0.45;
    if (basename && includesToken(text, basename.toLowerCase())) score += 0.25;
    if (symbol && !GENERIC_SYMBOLS.has(symbol.toLowerCase()) && includesToken(text, symbol.toLowerCase())) score += 0.55;
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
    const allowed = new Map(selectedNodes(job).map((node) => [node.node_id, node]));
    const selected = new Set(allowed.keys());
    const hypotheses = params.hypotheses.filter((hypothesis) => selected.has(hypothesis.node_id));
    const path = params.investigation_path.filter((step) => selected.has(step.node_id));
    for (const node of allowed.values()) {
      if (!hypotheses.some((hypothesis) => hypothesis.node_id === node.node_id)) {
        hypotheses.push({
          node_id: node.node_id,
          confidence: Math.min(0.95, Math.max(0.55, node.score)),
          rationale: "Selected by bounded issue evidence ranking.",
        });
      }
      if (!path.some((step) => step.node_id === node.node_id)) {
        path.push({
          node_id: node.node_id,
          path: node.path,
          why: "Inspect this bounded candidate selected from issue evidence.",
        });
      }
    }
    const transcript = {
      sample_id: job.job_id,
      variant_id: "pi-opencode-kimi-k25-issue-tools",
      tool_calls: toolCalls.filter((call) => call.name !== "finish_issue_map_transcript"),
      final: {
        hypotheses,
        investigation_path: path.map((step) => {
          const node = nodeById(job, step.node_id);
          return { ...step, path: node?.path || step.path };
        }),
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
  pi.registerTool(rankIssueCandidates);
  pi.registerTool(loadFocusGraph);
  pi.registerTool(loadCodeContext);
  pi.registerTool(finishIssueMapTranscript);
}
