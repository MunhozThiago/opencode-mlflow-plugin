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
}

interface MLflowClient {
  Experiments: {
    list(): Promise<{ experiments: any[] }>;
    create(opts: { name: string; artifact_location: string }): Promise<{ experiment_id: string }>;
  };
  Runs: {
    create(opts: { experiment_id: string; user_id: string; start_time: number; tags: Record<string, string> }): Promise<{ run: { info: { run_id: string } } }>;
    update(opts: { run_id: string; status: string; end_time: number }): Promise<any>;
    logMetric(opts: { run_id: string; key: string; value: number; timestamp: number }): Promise<any>;
    logParameter(opts: { run_id: string; key: string; value: string }): Promise<any>;
  };
}

const DEFAULT_OPTIONS: Required<MlflowPluginOptions> = {
  trackingUri: "http://localhost:5000",
  experimentName: "opencode-sessions",
  logToolDetails: false,
};

const plugin: Plugin = async (input: PluginInput, options?: MlflowPluginOptions): Promise<Hooks> => {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const sessions = new Map<string, SessionData>();
  let client: MLflowClient | null = null;
  let experimentId: string | null = null;

  async function getClient(): Promise<MLflowClient> {
    if (client) return client;

    const mlflowModule = await import("mlflow");
    const MLflowClass = (mlflowModule as any).default || mlflowModule;
    client = new MLflowClass({ endpoint: opts.trackingUri }) as MLflowClient;
    return client;
  }

  async function ensureExperiment(): Promise<string> {
    if (experimentId) return experimentId;

    const mlflow = await getClient();
    const { experiments } = await mlflow.Experiments.list();
    const existing = experiments.find((e: any) => e.name === opts.experimentName);

    if (existing) {
      experimentId = existing.experiment_id!;
    } else {
      const result = await mlflow.Experiments.create({
        name: opts.experimentName,
        artifact_location: "",
      });
      experimentId = result.experiment_id;
    }

    return experimentId!;
  }

  async function startRun(sessionId: string): Promise<string> {
    const mlflow = await getClient();
    const expId = await ensureExperiment();

    const result = await mlflow.Runs.create({
      experiment_id: expId,
      user_id: "opencode",
      start_time: Date.now(),
      tags: {
        "opencode.session_id": sessionId,
        "opencode.plugin_version": "0.1.0",
      },
    });

    return result.run.info!.run_id!;
  }

  async function endRun(runId: string, status: string = "FINISHED"): Promise<void> {
    const mlflow = await getClient();
    await mlflow.Runs.update({
      run_id: runId,
      status,
      end_time: Date.now(),
    });
  }

  async function logMetric(runId: string, key: string, value: number): Promise<void> {
    const mlflow = await getClient();
    await mlflow.Runs.logMetric({
      run_id: runId,
      key,
      value,
      timestamp: Date.now(),
    });
  }

  async function logParam(runId: string, key: string, value: string): Promise<void> {
    const mlflow = await getClient();
    await mlflow.Runs.logParameter({
      run_id: runId,
      key,
      value,
    });
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
      };
      sessions.set(sessionId, session);
      await logParam(runId, "session_id", sessionId);
    }
    return session;
  }

  return {
    config: async () => {},

    "chat.message": async (input) => {
      const session = await getOrCreateSession(input.sessionID);
      session.messageCount++;
      await logMetric(session.runId, "message_count", session.messageCount);
    },

    "tool.execute.before": async (input) => {
      const session = await getOrCreateSession(input.sessionID);
      session.toolCount++;
      await logMetric(session.runId, "tool_count", session.toolCount);

      if (opts.logToolDetails) {
        const toolKey = `tool_${input.tool}_count`;
        await logMetric(session.runId, toolKey, 1);
      }
    },

    "tool.execute.after": async (input, output) => {
      const session = sessions.get(input.sessionID);
      if (!session) return;

      if (output.output && output.output.includes("error")) {
        session.errors++;
        await logMetric(session.runId, "error_count", session.errors);
      }
    },

    event: async (input) => {
      const evt = input.event as any;
      if (evt.type === "session.end") {
        const sessionId = evt.sessionID || evt.sessionId;
        if (!sessionId) return;

        const session = sessions.get(sessionId);
        if (!session) return;

        const duration = Date.now() - session.startTime;
        await logMetric(session.runId, "duration_ms", duration);
        await logMetric(session.runId, "final_tool_count", session.toolCount);
        await logMetric(session.runId, "final_message_count", session.messageCount);
        await logMetric(session.runId, "final_error_count", session.errors);

        await endRun(session.runId, "FINISHED");
        sessions.delete(sessionId);
      }
    },
  };
};

export default plugin;
