import type { Plugin } from "@opencode-ai/plugin";
import * as mlflow from "mlflow";

interface MlflowPluginOptions {
  /** MLflow tracking URI (default: http://localhost:5000) */
  trackingUri?: string;
  /** Experiment name for opencode sessions */
  experimentName?: string;
  /** Whether to log tool inputs/outputs (default: false) */
  logToolDetails?: boolean;
  /** Whether to log chat messages (default: false) */
  logMessages?: boolean;
}

interface SessionData {
  runId: string;
  startTime: number;
  toolCount: number;
  messageCount: number;
  errors: number;
}

const DEFAULT_OPTIONS: Required<MlflowPluginOptions> = {
  trackingUri: "http://localhost:5000",
  experimentName: "opencode-sessions",
  logToolDetails: false,
  logMessages: false,
};

const plugin: Plugin = async (input, options?: MlflowPluginOptions) => {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const sessions = new Map<string, SessionData>();
  let client: mlflow.MlflowClient | null = null;

  async function getClient(): Promise<mlflow.MlflowClient> {
    if (!client) {
      client = new mlflow.MlflowClient({
        trackingUri: opts.trackingUri,
      });
    }
    return client;
  }

  async function ensureExperiment(): Promise<number> {
    const c = await getClient();
    try {
      const experiment = await c.getExperimentByName(opts.experimentName);
      return experiment.experimentId!;
    } catch {
      const experimentId = await c.createExperiment(opts.experimentName);
      return experimentId;
    }
  }

  async function startRun(sessionId: string): Promise<string> {
    const c = await getClient();
    const experimentId = await ensureExperiment();

    const run = await c.createRun({
      experimentId,
      runName: `session-${sessionId}`,
      tags: {
        "opencode.session_id": sessionId,
        "opencode.plugin_version": "0.1.0",
      },
    });

    return run.info.runId!;
  }

  async function endRun(runId: string, status: string = "FINISHED"): Promise<void> {
    const c = await getClient();
    await c.endRun(runId, status);
  }

  async function logMetric(
    runId: string,
    key: string,
    value: number,
    step?: number
  ): Promise<void> {
    const c = await getClient();
    await c.logMetric(runId, key, value, step);
  }

  async function logParam(
    runId: string,
    key: string,
    value: string
  ): Promise<void> {
    const c = await getClient();
    await c.logParam(runId, key, value);
  }

  return {
    config: (cfg) => {
      // Configuration hook - runs once on init
    },

    "chat.message": async (input, output) => {
      // Track chat messages if enabled
      if (!opts.logMessages) return;

      const sessionId = input.context?.sessionId;
      if (!sessionId) return;

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

      session.messageCount++;
      await logMetric(session.runId, "message_count", session.messageCount);
    },

    "tool.execute.before": async (input, output) => {
      const sessionId = input.context?.sessionId;
      if (!sessionId) return;

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

      session.toolCount++;
      await logMetric(session.runId, "tool_count", session.toolCount);

      if (opts.logToolDetails) {
        await logMetric(
          session.runId,
          `tool_${input.tool}_count`,
          1
        );
      }
    },

    "tool.execute.after": async (input, output) => {
      const sessionId = input.context?.sessionId;
      if (!sessionId) return;

      const session = sessions.get(sessionId);
      if (!session) return;

      if (output.error) {
        session.errors++;
        await logMetric(session.runId, "error_count", session.errors);
      }
    },

    event: async (input) => {
      // Handle session end events
      if (input.type === "session.end") {
        const sessionId = input.sessionId;
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
