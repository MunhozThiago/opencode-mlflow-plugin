import type { Plugin, PluginInput, Hooks } from "@opencode-ai/plugin";

interface MlflowPluginOptions {
  trackingUri?: string;
  experimentName?: string;
  logToolDetails?: boolean;
}

interface SessionData {
  runId: string;
  startTime: number;
  toolCount: number;
  messageCount: number;
  errors: number;
  toolTimings: Map<string, number>;
}

interface MlflowResponse {
  experiments?: any[];
  experiment_id?: string;
  run?: { info?: { run_id?: string } };
  run_info?: any;
}

const DEFAULT_OPTIONS: Required<MlflowPluginOptions> = {
  trackingUri: "http://localhost:5000",
  experimentName: "opencode-sessions",
  logToolDetails: false,
};

async function mlflowRequest(
  trackingUri: string,
  endpoint: string,
  method: string = "GET",
  body?: any
): Promise<MlflowResponse> {
  const url = `${trackingUri}/api/2.0/mlflow${endpoint}`;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);

  try {
    const options: RequestInit = { method, headers, signal: controller.signal };

    if (body && method !== "GET") {
      options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`MLflow API error: ${response.status} ${response.statusText} - ${text}`);
    }

    return response.json();
  } finally {
    clearTimeout(timeout);
  }
}

const plugin: Plugin = async (input: PluginInput, options?: MlflowPluginOptions): Promise<Hooks> => {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const sessions = new Map<string, SessionData>();
  let experimentId: string | null = null;

  async function ensureExperiment(): Promise<string> {
    if (experimentId) return experimentId;

    const data = await mlflowRequest(
      opts.trackingUri,
      "/experiments/search",
      "POST",
      { max_results: 100 }
    );

    const existing = data.experiments?.find((e: any) => e.name === opts.experimentName);

    if (existing) {
      experimentId = existing.experiment_id!;
    } else {
      const result = await mlflowRequest(
        opts.trackingUri,
        "/experiments/create",
        "POST",
        { name: opts.experimentName, artifact_location: "" }
      );
      experimentId = result.experiment_id!;
    }

    return experimentId!;
  }

  async function startRun(sessionId: string): Promise<string> {
    const expId = await ensureExperiment();

    const result = await mlflowRequest(
      opts.trackingUri,
      "/runs/create",
      "POST",
      {
        experiment_id: expId,
        user_id: "opencode",
        start_time: Date.now(),
        tags: [
          { key: "opencode.session_id", value: sessionId },
          { key: "opencode.plugin_version", value: "0.1.0" },
        ],
      }
    );

    return result.run?.info?.run_id || "";
  }

  async function endRun(runId: string, status: string = "FINISHED"): Promise<void> {
    await mlflowRequest(
      opts.trackingUri,
      "/runs/update",
      "POST",
      {
        run_id: runId,
        status,
        end_time: Date.now(),
      }
    );
  }

  async function logMetric(runId: string, key: string, value: number): Promise<void> {
    await mlflowRequest(
      opts.trackingUri,
      "/runs/log-metric",
      "POST",
      {
        run_id: runId,
        key,
        value,
        timestamp: Date.now(),
      }
    );
  }

  async function logParam(runId: string, key: string, value: string): Promise<void> {
    await mlflowRequest(
      opts.trackingUri,
      "/runs/log-parameter",
      "POST",
      {
        run_id: runId,
        key,
        value,
      }
    );
  }

  async function logTag(runId: string, key: string, value: string): Promise<void> {
    await mlflowRequest(
      opts.trackingUri,
      "/runs/set-tag",
      "POST",
      {
        run_id: runId,
        key,
        value,
      }
    );
  }

  async function getOrCreateSession(sessionId: string): Promise<SessionData> {
    let session = sessions.get(sessionId);
    if (!session) {
      const runId = await startRun(sessionId);
      session = {
        runId,
        startTime: Date.now(),
        toolCount: 0,
        messageCount: 0,
        errors: 0,
        toolTimings: new Map(),
      };
      sessions.set(sessionId, session);
      await logParam(runId, "session_id", sessionId);
    }
    return session;
  }

  return {
    config: async () => {},

    dispose: async () => {
      for (const [sessionId, session] of sessions) {
        await endRun(session.runId, "INTERRUPTED");
        sessions.delete(sessionId);
      }
    },

    "chat.message": async (input) => {
      const session = await getOrCreateSession(input.sessionID);
      session.messageCount++;
      await logMetric(session.runId, "message_count", session.messageCount);

      if (input.agent) {
        await logTag(session.runId, "agent", input.agent);
      }

      if (input.model) {
        await logTag(session.runId, "provider_id", input.model.providerID);
        await logTag(session.runId, "model_id", input.model.modelID);
      }
    },

    "tool.execute.before": async (input) => {
      const session = await getOrCreateSession(input.sessionID);
      session.toolCount++;
      session.toolTimings.set(input.callID, Date.now());
      await logMetric(session.runId, "tool_count", session.toolCount);

      if (opts.logToolDetails) {
        const toolKey = `tool_${input.tool}_count`;
        await logMetric(session.runId, toolKey, 1);
      }
    },

    "tool.execute.after": async (input, output) => {
      const session = sessions.get(input.sessionID);
      if (!session) return;

      const startTime = session.toolTimings.get(input.callID);
      if (startTime) {
        const duration = Date.now() - startTime;
        await logMetric(session.runId, `tool_${input.tool}_duration_ms`, duration);
        session.toolTimings.delete(input.callID);
      }

      if (output.metadata?.error || (output.output && output.output.includes("Error"))) {
        session.errors++;
        await logMetric(session.runId, "error_count", session.errors);
      }
    },

    event: async (input) => {
      const evt = input.event as any;
      const sessionId = evt.sessionID || evt.sessionId;
      if (!sessionId) return;

      const session = sessions.get(sessionId);
      if (!session) return;

      if (evt.type === "session.end" || evt.type === "session.interrupt") {
        const duration = Date.now() - session.startTime;
        await logMetric(session.runId, "duration_ms", duration);
        await logMetric(session.runId, "final_tool_count", session.toolCount);
        await logMetric(session.runId, "final_message_count", session.messageCount);
        await logMetric(session.runId, "final_error_count", session.errors);

        const status = evt.type === "session.interrupt" ? "INTERRUPTED" : "FINISHED";
        await endRun(session.runId, status);
        sessions.delete(sessionId);
      }
    },
  };
};

export default plugin;
