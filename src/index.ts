import type { Plugin, Hooks } from "@opencode-ai/plugin";

export interface MlflowPluginOptions {
  trackingUri?: string;
  experimentName?: string;
  logSpans?: boolean;
  logTokens?: boolean;
  logToolDetails?: boolean;
  requestTimeoutMs?: number;
}

const PLUGIN_VERSION = "1.1.0";

const DEFAULT_OPTIONS: Required<MlflowPluginOptions> = {
  trackingUri: "http://localhost:5000",
  experimentName: "opencode-sessions",
  logSpans: true,
  logTokens: true,
  logToolDetails: false,
  requestTimeoutMs: 5000,
};

// ---------- MLflow REST client (v2 metrics/runs + v3 traces) ----------

let versionTagCounter = 0;

async function mlflowRequest<T = any>(
  trackingUri: string,
  endpoint: string,
  body?: any,
  timeoutMs = 5000
): Promise<T> {
  const url = `${trackingUri}${endpoint}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`MLflow API error ${response.status} ${response.statusText}: ${text.slice(0, 300)}`);
    }
    if (response.status === 204) return undefined as T;
    const text = await response.text();
    return text ? JSON.parse(text) : (undefined as T);
  } finally {
    clearTimeout(timeout);
  }
}

// ---------- Trace ID helpers ----------

function randomHex(bytes: number): string {
  const arr = new Uint8Array(bytes);
  crypto.getRandomValues(arr);
  return Array.from(arr, (b) => b.toString(16).padStart(2, "0")).join("");
}

function newTraceId(): string {
  return `tr-${randomHex(16)}`;
}

function newSpanId(): string {
  return randomHex(8);
}

function isoTime(ms: number): string {
  return new Date(ms).toISOString().replace(/\.(\d{3})Z$/, ".$1Z");
}

function durationString(ms: number): string {
  if (ms < 1000) return `0.${String(ms).padStart(3, "0")}s`;
  return `${(ms / 1000).toFixed(3)}s`;
}

// ---------- Session state ----------

interface PendingSpan {
  traceId: string;
  spanId: string;
  parentId: string | null;
  name: string;
  spanType: string;
  startTime: number;
  inputs?: unknown;
}

interface SessionData {
  runId: string;
  startTime: number;
  toolCount: number;
  messageCount: number;
  errors: number;
  toolTimings: Map<string, number>;
  pendingSpans: Map<string, PendingSpan>;
  activeTraces: Map<string, { traceId: string; startTime: number; rootSpanId: string }>;
  tokens: { input: number; output: number; reasoning: number; cacheRead: number; cacheWrite: number };
  cost: number;
  agent?: string;
  providerId?: string;
  modelId?: string;
}

function newSessionData(runId: string): SessionData {
  return {
    runId,
    startTime: Date.now(),
    toolCount: 0,
    messageCount: 0,
    errors: 0,
    toolTimings: new Map(),
    pendingSpans: new Map(),
    activeTraces: new Map(),
    tokens: { input: 0, output: 0, reasoning: 0, cacheRead: 0, cacheWrite: 0 },
    cost: 0,
  };
}

// ---------- Trace payload builders ----------

type AttrValue = { string_value?: string; int_value?: number; double_value?: number; bool_value?: boolean };

function attr(key: string, value: string | number | boolean): { key: string; value: AttrValue } {
  if (typeof value === "string") return { key, value: { string_value: value } };
  if (typeof value === "boolean") return { key, value: { bool_value: value } };
  if (Number.isInteger(value)) return { key, value: { int_value: value } };
  return { key, value: { double_value: value } };
}

function truncate(content: string, max = 4000): string {
  if (content.length <= max) return content;
  return content.slice(0, max) + `...[truncated ${content.length - max} chars]`;
}

function jsonSafe(value: unknown, max = 4000): string | undefined {
  if (value === undefined || value === null) return undefined;
  try {
    const s = typeof value === "string" ? value : JSON.stringify(value);
    return truncate(s, max);
  } catch {
    return String(value).slice(0, max);
  }
}

interface SpanPayload {
  span_id: string;
  trace_id: string;
  parent_id?: string | null;
  name: string;
  start_time_millis: number;
  end_time_millis: number;
  attributes: { key: string; value: AttrValue }[];
  status: { status_code: string; description?: string };
}

interface TracePayload {
  trace_info: {
    trace_id: string;
    trace_location: { type: string; mlflow_experiment: { experiment_id: string } };
    request_time: string;
    execution_duration: string;
    state: string;
    trace_metadata: Record<string, string>;
    tags: Record<string, string>;
  };
  data: {
    spans: any[];
    request?: string;
    response?: string;
  };
}

function buildTracePayload(params: {
  experimentId: string;
  traceId: string;
  sessionId: string;
  runId: string;
  startTime: number;
  endTime: number;
  state: "OK" | "ERROR";
  spans: any[];
  request?: string;
  response?: string;
  errorMessage?: string;
}): TracePayload {
  const {
    experimentId,
    traceId,
    sessionId,
    runId,
    startTime,
    endTime,
    state,
    spans,
    request,
    response,
    errorMessage,
  } = params;

  const trace_metadata: Record<string, string> = {
    "mlflow.sourceRun": runId,
    "opencode.session_id": sessionId,
  };
  if (errorMessage) trace_metadata["opencode.error"] = truncate(errorMessage, 300);

  const tags: Record<string, string> = {
    "mlflow.traceName": "opencode-turn",
    "opencode.plugin_version": PLUGIN_VERSION,
  };

  return {
    trace_info: {
      trace_id: traceId,
      trace_location: { type: "MLFLOW_EXPERIMENT", mlflow_experiment: { experiment_id: experimentId } },
      request_time: isoTime(startTime),
      execution_duration: durationString(Math.max(0, endTime - startTime)),
      state,
      trace_metadata,
      tags,
    },
    data: {
      spans,
      request: request ?? undefined,
      response: response ?? undefined,
    },
  };
}

// ---------- Main plugin ----------

const plugin: Plugin = async (input, options?: MlflowPluginOptions): Promise<Hooks> => {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const sessions = new Map<string, SessionData>();
  let experimentId: string | null = null;
  let experimentReady: Promise<string> | null = null;

  // Never let a tracking failure break the user's session.
  async function safe(name: string, fn: () => Promise<void>): Promise<void> {
    try {
      await fn();
    } catch (err) {
      console.warn(`[opencode-mlflow] ${name} failed:`, err instanceof Error ? err.message : err);
    }
  }

  async function ensureExperiment(): Promise<string> {
    if (experimentId) return experimentId;
    if (experimentReady) return experimentReady;

    experimentReady = (async () => {
      const data = await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/experiments/search", {
        max_results: 100,
      });
      const existing = data.experiments?.find((e: any) => e.name === opts.experimentName);
      if (existing) {
        const id = existing.experiment_id as string;
        experimentId = id;
        return id;
      }
      const result = await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/experiments/create", {
        name: opts.experimentName,
        artifact_location: "",
      });
      const created = result.experiment_id as string;
      experimentId = created;
      return created;
    })();

    try {
      return await experimentReady;
    } finally {
      experimentReady = null;
    }
  }

  async function logMetric(runId: string, key: string, value: number): Promise<void> {
    await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/runs/log-metric", {
      run_id: runId,
      key,
      value,
      timestamp: Date.now(),
    });
  }

  async function logParam(runId: string, key: string, value: string): Promise<void> {
    await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/runs/log-parameter", {
      run_id: runId,
      key,
      value,
    });
  }

  async function logBatch(
    runId: string,
    metrics: { key: string; value: number }[]
  ): Promise<void> {
    const timestamp = Date.now();
    await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/runs/log-batch", {
      run_id: runId,
      metrics: metrics.map((m) => ({ key: m.key, value: m.value, timestamp })),
      params: [],
      tags: [],
    });
  }

  async function logTag(runId: string, key: string, value: string): Promise<void> {
    await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/runs/set-tag", {
      run_id: runId,
      key,
      value,
    });
  }

  async function endRun(runId: string, status: string): Promise<void> {
    await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/runs/update", {
      run_id: runId,
      status,
      end_time: Date.now(),
    });
  }

  async function getOrCreateSession(sessionId: string): Promise<SessionData> {
    let session = sessions.get(sessionId);
    if (!session) {
      const expId = await ensureExperiment();
      const result = await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/runs/create", {
        experiment_id: expId,
        user_id: "opencode",
        start_time: Date.now(),
        tags: [
          { key: "opencode.session_id", value: sessionId },
          { key: "opencode.plugin_version", value: PLUGIN_VERSION },
        ],
      });
      const runId = result.run?.info?.run_id || "";
      session = newSessionData(runId);
      sessions.set(sessionId, session);
      await logParam(runId, "session_id", sessionId);
    }
    return session;
  }

  // ---------- Trace lifecycle ----------

  // Turn lifecycle: a "trace" spans one user prompt -> assistant completion.
  // message parts (tool calls, text) become child spans under the LLM span.

  function startTurnTrace(session: SessionData, messageID: string): { traceId: string; rootSpanId: string } {
    const traceId = newTraceId();
    const rootSpanId = newSpanId();
    session.pendingSpans.set(rootSpanId, {
      traceId,
      spanId: rootSpanId,
      parentId: null,
      name: "opencode-turn",
      spanType: "LLM",
      startTime: Date.now(),
    });
    session.activeTraces.set(messageID, { traceId, startTime: Date.now(), rootSpanId });
    return { traceId, rootSpanId };
  }

  async function finishTurnTrace(
    session: SessionData,
    sessionID: string,
    messageID: string,
    extra: {
      request?: string;
      response?: string;
      error?: string;
      spanAttributes?: { key: string; value: AttrValue }[];
    } = {}
  ): Promise<void> {
    const active = session.activeTraces.get(messageID);
    if (!active) return;
    session.activeTraces.delete(messageID);

    if (!opts.logSpans) {
      session.pendingSpans.clear();
      return;
    }

    const expId = await ensureExperiment();
    const endTime = Date.now();
    const root = session.pendingSpans.get(active.rootSpanId);
    session.pendingSpans.delete(active.rootSpanId);

    const rootSpan: SpanPayload = {
      span_id: active.rootSpanId,
      trace_id: active.traceId,
      name: "opencode-turn",
      start_time_millis: active.startTime,
      end_time_millis: endTime,
      attributes: [
        attr("mlflow.spanType", "LLM"),
        attr("opencode.session_id", sessionID),
        ...(extra.spanAttributes ?? []),
      ],
      status: { status_code: extra.error ? "ERROR" : "OK", ...(extra.error ? { description: truncate(extra.error, 300) } : {}) },
    };

    const payload = buildTracePayload({
      experimentId: expId,
      traceId: active.traceId,
      sessionId: sessionID,
      runId: session.runId,
      startTime: active.startTime,
      endTime,
      state: extra.error ? "ERROR" : "OK",
      spans: [rootSpan],
      request: extra.request,
      response: extra.response,
      errorMessage: extra.error,
    });

    await mlflowRequest(opts.trackingUri, "/api/3.0/mlflow/traces", { trace: payload }, 10000);
  }

  return {
    config: async () => {},

    dispose: async () => {
      for (const [sessionId, session] of sessions) {
        sessions.delete(sessionId);
        if (!session.runId) continue;
        await safe("dispose:endRun", () => endRun(session.runId, "FINISHED"));
      }
    },

    "chat.message": async (msg) => {
      await safe("chat.message", async () => {
        const session = await getOrCreateSession(msg.sessionID);
        session.messageCount++;
        session.agent = msg.agent ?? session.agent;
        if (msg.model) {
          session.providerId = msg.model.providerID;
          session.modelId = msg.model.modelID;
        }

        // Start a new turn trace for this user prompt.
        const messageID = msg.messageID ?? `msg-${session.messageCount}`;
        startTurnTrace(session, messageID);

        await logBatch(session.runId, [
          { key: "message_count", value: session.messageCount },
        ]);

        if (session.agent) await logTag(session.runId, "agent", session.agent);
        if (session.providerId) await logTag(session.runId, "provider_id", session.providerId);
        if (session.modelId) await logTag(session.runId, "model_id", session.modelId);
      });
    },

    "tool.execute.before": async (toolInput, output) => {
      await safe("tool.execute.before", async () => {
        const session = await getOrCreateSession(toolInput.sessionID);
        session.toolCount++;
        session.toolTimings.set(toolInput.callID, Date.now());
        await logBatch(session.runId, [{ key: "tool_count", value: session.toolCount }]);
        if (opts.logToolDetails) {
          await logMetric(session.runId, `tool_${toolInput.tool}_count`, 1);
        }
      });
    },

    "tool.execute.after": async (toolInput, toolOutput) => {
      await safe("tool.execute.after", async () => {
        const session = sessions.get(toolInput.sessionID);
        if (!session) return;

        const started = session.toolTimings.get(toolInput.callID);
        if (started !== undefined) {
          session.toolTimings.delete(toolInput.callID);
          const duration = Date.now() - started;
          if (opts.logToolDetails) {
            await logMetric(session.runId, `tool_${toolInput.tool}_duration_ms`, duration);
          }
        }
      });
    },

    event: async (evt) => {
      await safe("event", async () => {
        const e = evt.event as any;
        const sessionId = e.sessionID ?? e.sessionId;
        if (!sessionId) return;

        const session = sessions.get(sessionId);
        if (!session) return;

        if (e.type === "session.end" || e.type === "session.interrupt" || e.type === "session.idle") {
          const duration = Date.now() - session.startTime;
          await safe("event:metrics", async () => {
            await logBatch(session.runId, [
              { key: "duration_ms", value: duration },
              { key: "final_tool_count", value: session.toolCount },
              { key: "final_message_count", value: session.messageCount },
              { key: "final_error_count", value: session.errors },
            ]);
          });

          const status = e.type === "session.interrupt" ? "INTERRUPTED" : "FINISHED";
          await safe("event:endRun", () => endRun(session.runId, status));
          sessions.delete(sessionId);
        }
      });
    },
  };
};

export default plugin;
