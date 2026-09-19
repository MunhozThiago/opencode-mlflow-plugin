import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { appendFileSync } from "node:fs";

const PLUGIN_VERSION = "2.0.0";
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PYTHON_SCRIPT = join(__dirname, "..", "send_trace.py");
const LOG_FILE = join(__dirname, "..", "debug.log");

function dbg(msg) { try { appendFileSync(LOG_FILE, `[${new Date().toISOString()}] ${msg}\n`); } catch {} }

const DEFAULT_OPTIONS = {
    trackingUri: "http://localhost:5000",
    experimentName: "opencode-sessions",
    logSpans: true,
    logTokens: true,
    logToolDetails: false,
};

// ---------- MLflow REST client ----------
async function mlflowRequest(trackingUri, endpoint, body, timeoutMs = 5000) {
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
            throw new Error(`MLflow API ${response.status}: ${text.slice(0, 200)}`);
        }
        if (response.status === 204) return undefined;
        const text = await response.text();
        return text ? JSON.parse(text) : undefined;
    } finally {
        clearTimeout(timeout);
    }
}

// ---------- Python subprocess for trace sending via MLflow SDK ----------
function sendViaPython(traceData) {
    return new Promise((resolve, reject) => {
        // Convert BigInt to strings before JSON.stringify
        const safeData = { ...traceData };
        if (typeof safeData.start_time_ns === "bigint") safeData.start_time_ns = safeData.start_time_ns.toString();
        if (typeof safeData.end_time_ns === "bigint") safeData.end_time_ns = safeData.end_time_ns.toString();
        const jsonData = JSON.stringify(safeData);
        const proc = execFile(
            "python",
            [PYTHON_SCRIPT],
            { timeout: 120000, maxBuffer: 1024 * 1024 },
            (error, stdout, stderr) => {
                if (error) {
                    reject(new Error(`Python: ${error.message}${stderr ? " " + stderr.slice(0, 200) : ""}`));
                    return;
                }
                try {
                    const result = JSON.parse(stdout.trim());
                    if (result.error) reject(new Error(`Python: ${result.error}`));
                    else resolve(result);
                } catch {
                    reject(new Error(`Python output: ${stdout.slice(0, 200)}`));
                }
            }
        );
        proc.stdin.write(jsonData);
        proc.stdin.end();
    });
}

// ---------- Helpers ----------
function randomHex(bytes) {
    const arr = new Uint8Array(bytes);
    crypto.getRandomValues(arr);
    return Array.from(arr, (b) => b.toString(16).padStart(2, "0")).join("");
}
function truncate(s, max = 4000) {
    return s.length <= max ? s : s.slice(0, max) + `...[truncated ${s.length - max}]`;
}
function jsonSafe(value, max = 4000) {
    if (value == null) return undefined;
    try {
        const s = typeof value === "string" ? value : JSON.stringify(value);
        return truncate(s, max);
    } catch { return String(value).slice(0, max); }
}

// ---------- Main plugin ----------
const plugin = async (input, options) => {
    const opts = { ...DEFAULT_OPTIONS, ...options };
    dbg(`PLUGIN LOADED v${PLUGIN_VERSION} script=${PYTHON_SCRIPT}`);
    const sessions = new Map();

    async function safe(name, fn) {
        try { await fn(); } catch (err) { dbg(`ERROR [${name}]: ${err?.message ?? err}`); }
    }

    async function ensureExperiment() {
        const data = await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/experiments/search", { max_results: 100 });
        const existing = data.experiments?.find((e) => e.name === opts.experimentName);
        if (existing) return existing.experiment_id;
        const result = await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/experiments/create", { name: opts.experimentName });
        return result.experiment_id;
    }

    async function logBatch(runId, metrics) {
        const timestamp = Date.now();
        await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/runs/log-batch", {
            run_id: runId, metrics: metrics.map((m) => ({ key: m.key, value: m.value, timestamp })), params: [], tags: [],
        });
    }

    async function logTag(runId, key, value) {
        await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/runs/set-tag", { run_id: runId, key, value });
    }

    async function logParam(runId, key, value) {
        await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/runs/log-parameter", { run_id: runId, key, value });
    }

    async function endRun(runId, status) {
        await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/runs/update", { run_id: runId, status, end_time: Date.now() });
    }

    async function getOrCreateSession(sessionId) {
        let session = sessions.get(sessionId);
        if (!session) {
            const expId = await ensureExperiment();
            const result = await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/runs/create", {
                experiment_id: expId, user_id: "opencode", start_time: Date.now(),
                tags: [{ key: "opencode.session_id", value: sessionId }],
            });
            const runId = result.run?.info?.run_id || "";
            session = {
                runId, startTime: Date.now(),
                toolCount: 0, messageCount: 0, errors: 0,
                toolTimings: new Map(),
                responseText: new Map(),
                activeTrace: null,
                userMessage: "",
                agent: undefined, providerId: undefined, modelId: undefined,
                lastCompletedAt: null,
                // Token counts (accumulated across all assistant messages in turn)
                inputTokens: 0, outputTokens: 0, totalTokens: 0,
                // Generation params
                temperature: undefined, topP: undefined, topK: undefined, maxTokens: undefined,
                // Cost in dollars
                cost: 0,
                // Model info from message.updated events
                lastModelID: undefined, lastProviderID: undefined,
                // Finish reason
                finishReason: undefined,
                // Reasoning tokens and cache
                reasoningTokens: 0, cacheRead: 0, cacheWrite: 0,
            };
            sessions.set(sessionId, session);
            await logParam(runId, "session_id", sessionId);
        }
        return session;
    }

    async function finishTurnTrace(session, sessionId, extra = {}) {
        const trace = session.activeTrace;
        if (!trace) { return; }
        session.activeTrace = null;
        if (!opts.logSpans) { return; }

        // Send via Python MLflow SDK (handles preview data + spans)
        try {
            const modelName = session.lastModelID || session.modelId || "unknown";
            const providerName = session.lastProviderID || session.providerId || "unknown";
            dbg(`finishTurnTrace: sending via Python. traceId=${trace.traceId} model=${modelName} tokens=${session.inputTokens}/${session.outputTokens}/${session.totalTokens}`);
            await sendViaPython({
                tracking_uri: opts.trackingUri,
                experiment_id: await ensureExperiment(),
                trace_id: trace.traceId,
                span_id: trace.spanId,
                name: "opencode-turn",
                kind: 3,
                start_time_ns: BigInt(trace.startTime) * 1000000n,
                end_time_ns: BigInt(Date.now()) * 1000000n,
                attributes: {
                    "mlflow.spanType": "LLM",
                    "opencode.session_id": sessionId,
                    // GenAI semantic conventions
                    "opencode.agent": session.agent ?? "unknown",
                    "opencode.model": modelName,
                    "opencode.provider": providerName,
                    "opencode.input_tokens": session.inputTokens,
                    "opencode.output_tokens": session.outputTokens,
                    "opencode.total_tokens": session.totalTokens,
                    "opencode.reasoning_tokens": session.reasoningTokens,
                    "opencode.cache_read": session.cacheRead,
                    "opencode.cache_write": session.cacheWrite,
                    "opencode.cost_usd": session.cost,
                    "opencode.finish_reason": session.finishReason ?? "unknown",
                    // Generation params
                    ...(session.temperature != null ? { "opencode.temperature": session.temperature } : {}),
                    ...(session.topP != null ? { "opencode.top_p": session.topP } : {}),
                    ...(session.topK != null ? { "opencode.top_k": session.topK } : {}),
                    ...(session.maxTokens != null ? { "opencode.max_tokens": session.maxTokens } : {}),
                },
                status_code: extra.error ? 2 : 0,
                request_preview: extra.request ?? undefined,
                response_preview: extra.response ?? undefined,
            });
            dbg(`finishTurnTrace: sent successfully`);
        } catch (err) {
            dbg(`finishTurnTrace ERROR: ${err?.message ?? err}`);
        }
    }

    // Helper to finalize the current turn's trace and end the run
    async function finalizeTrace(session, sessionId, trigger) {
        // Cancel any pending debounce timer
        if (session._finalizeTimer) {
            clearTimeout(session._finalizeTimer);
            session._finalizeTimer = null;
        }
        if (!session.activeTrace) {
            return;
        }
        let responseText = "";
        for (const [key, text] of session.responseText.entries()) {
            if (key.startsWith(sessionId + ":")) responseText += text;
        }
        // dbg(`finalizeTrace: creating trace for ${sessionId} trigger=${trigger} userMsg=${session.userMessage?.slice(0,50)} respLen=${responseText.length}`);
        await safe("finishTurnTrace", async () => {
            await finishTurnTrace(session, sessionId, {
                request: session.userMessage ? jsonSafe(session.userMessage) : undefined,
                response: responseText ? jsonSafe(responseText) : undefined,
            });
        });
        session.userMessage = "";
        session.responseText.clear();
        await safe("endRun", () => endRun(session.runId, "FINISHED"));
        // dbg(`finalizeTrace: done for ${sessionId}`);
    }

    return {
        dispose: async () => {
            for (const [sid, session] of sessions) {
                if (session.runId) await safe("dispose", () => endRun(session.runId, "FINISHED"));
            }
            sessions.clear();
        },

        "chat.message": async (msg, out) => {
            await safe("chat.message", async () => {
                // dbg(`chat.message: sessionID=${msg.sessionID}`);
                const session = await getOrCreateSession(msg.sessionID);

                // Finalize previous turn's trace before starting new one
                if (session.activeTrace) {
                    await finalizeTrace(session, msg.sessionID, "next-chat-message");
                }

                session.messageCount++;
                session.agent = msg.agent ?? session.agent;
                if (msg.model) { session.providerId = msg.model.providerID; session.modelId = msg.model.modelID; }

                let userMessage = "";
                if (out?.parts) {
                    userMessage = out.parts.filter((p) => p.type === "text").map((p) => p.text).join("\n");
                }

                session.activeTrace = { traceId: `tr-${randomHex(16)}`, spanId: randomHex(8), startTime: Date.now() };
                session.userMessage = userMessage;
                session.responseText.clear();
                // Reset token counts and generation params for new turn
                session.inputTokens = 0;
                session.outputTokens = 0;
                session.totalTokens = 0;
                session.reasoningTokens = 0;
                session.cacheRead = 0;
                session.cacheWrite = 0;
                session.temperature = undefined;
                session.topP = undefined;
                session.topK = undefined;
                session.maxTokens = undefined;
                session.cost = 0;
                session.finishReason = undefined;

                await logBatch(session.runId, [{ key: "message_count", value: session.messageCount }]);
                if (session.agent) await logTag(session.runId, "agent", session.agent);
                if (session.providerId) await logTag(session.runId, "provider_id", session.providerId);
                if (session.modelId) await logTag(session.runId, "model_id", session.modelId);
            });
        },

        "chat.params": async (input, output) => {
            await safe("chat.params", async () => {
                const session = sessions.get(input.sessionID);
                if (!session) return;
                session.temperature = output.temperature;
                session.topP = output.topP;
                session.topK = output.topK;
                session.maxTokens = output.maxOutputTokens;
                // dbg(`chat.params: temp=${output.temperature} topP=${output.topP} topK=${output.topK} maxTokens=${output.maxOutputTokens}`);
            });
        },

        "tool.execute.before": async (toolInput) => {
            await safe("tool.execute.before", async () => {
                const session = await getOrCreateSession(toolInput.sessionID);
                session.toolCount++;
                session.toolTimings.set(toolInput.callID, Date.now());
                await logBatch(session.runId, [{ key: "tool_count", value: session.toolCount }]);
            });
        },

        "tool.execute.after": async (toolInput) => {
            await safe("tool.execute.after", async () => {
                const session = sessions.get(toolInput.sessionID);
                if (!session) return;
                const started = session.toolTimings.get(toolInput.callID);
                if (started !== undefined) {
                    session.toolTimings.delete(toolInput.callID);
                    if (opts.logToolDetails) {
                        await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/runs/log-metric", {
                            run_id: session.runId, key: `tool_${toolInput.tool}_duration_ms`,
                            value: Date.now() - started, timestamp: Date.now(),
                        });
                    }
                }
            });
        },

        event: async (evt) => {
            await safe("event", async () => {
                const e = evt.event;
                const sessionId = e.properties?.sessionID;
                if (!sessionId) return;
                const session = sessions.get(sessionId);
                if (!session) return;

                // Accumulate assistant text via message.part.updated (full snapshot)
                if (e.type === "message.part.updated" && e.properties?.part?.type === "text") {
                    const part = e.properties.part;
                    if (part.text) {
                        session.responseText.set(`${sessionId}:${part.messageID}`, part.text);
                    }
                    // Reset debounce on any text update activity
                    if (session._finalizeTimer) {
                        clearTimeout(session._finalizeTimer);
                        session._finalizeTimer = null;
                    }
                }

                // Accumulate streaming text deltas
                if (e.type === "message.part.delta") {
                    const { messageID, delta } = e.properties;
                    if (delta && typeof delta === "string" && messageID) {
                        const key = `${sessionId}:${messageID}`;
                        session.responseText.set(key, (session.responseText.get(key) ?? "") + delta);
                    }
                    // Reset debounce on any streaming activity
                    if (session._finalizeTimer) {
                        clearTimeout(session._finalizeTimer);
                        session._finalizeTimer = null;
                    }
                }

                // Finish trace when assistant message completes — accumulate token counts and metadata
                if (e.type === "message.updated" && e.properties?.info?.role === "assistant") {
                    const msg = e.properties.info;
                    // dbg(`assistant msg.updated: time=${JSON.stringify(msg.time)} keys=${Object.keys(msg).join(",")}`);
                    if (msg.time?.completed) {
                        session.lastCompletedAt = Date.now();

                        // Accumulate token counts from each assistant message
                        if (msg.tokens) {
                            const tok = msg.tokens;
                            // dbg(`tokens: input=${tok.input} output=${tok.output} reasoning=${tok.reasoning} cache_read=${tok.cache?.read} cache_write=${tok.cache?.write}`);
                            session.inputTokens += tok.input ?? 0;
                            session.outputTokens += tok.output ?? 0;
                            session.reasoningTokens += tok.reasoning ?? 0;
                            if (tok.cache) {
                                session.cacheRead += tok.cache.read ?? 0;
                                session.cacheWrite += tok.cache.write ?? 0;
                            }
                            // totalTokens = input + output + reasoning (opencode doesn't have a total field)
                            session.totalTokens = session.inputTokens + session.outputTokens + session.reasoningTokens;
                        }

                        // Track cost
                        if (typeof msg.cost === "number") {
                            session.cost += msg.cost;
                            // dbg(`cost: +${msg.cost} total=${session.cost}`);
                        }

                        // Track model/provider from message
                        if (msg.modelID) session.lastModelID = msg.modelID;
                        if (msg.providerID) session.lastProviderID = msg.providerID;

                        // Track finish reason
                        if (msg.finish) session.finishReason = msg.finish;

                        // dbg(`completed event recorded for session=${sessionId} tokens=${session.inputTokens}/${session.outputTokens}/${session.totalTokens} cost=${session.cost} model=${session.lastModelID || session.modelId} finish=${session.finishReason}`);

                        // Clear any existing debounce timer
                        if (session._finalizeTimer) clearTimeout(session._finalizeTimer);

                        // Debounce: wait 2s for more messages, then finalize
                        session._finalizeTimer = setTimeout(async () => {
                            session._finalizeTimer = null;
                            // dbg(`debounce timer fired for session=${sessionId}`);
                            // Collect response text
                            let responseText = "";
                            for (const [key, text] of session.responseText.entries()) {
                                if (key.startsWith(sessionId + ":")) responseText += text;
                            }
                            await safe("debounce-finalize", async () => {
                                if (!session.activeTrace) return;
                                dbg(`debounce finalize: creating trace for ${sessionId} respLen=${responseText.length}`);
                                await finishTurnTrace(session, sessionId, {
                                    request: session.userMessage ? jsonSafe(session.userMessage) : undefined,
                                    response: responseText ? jsonSafe(responseText) : undefined,
                                });
                                session.userMessage = "";
                                session.responseText.clear();
                            });
                        }, 2000);
                    }
                }

                // Session lifecycle metrics
                if (e.type === "session.end" || e.type === "session.interrupt" || e.type === "session.idle") {
                    dbg(`session lifecycle: ${e.type}`);
                    const duration = Date.now() - session.startTime;
                    await safe("session:metrics", async () => {
                        const timestamp = Date.now();
                        await mlflowRequest(opts.trackingUri, "/api/2.0/mlflow/runs/log-batch", {
                            run_id: session.runId,
                            metrics: [
                                { key: "duration_ms", value: duration, timestamp },
                                { key: "tool_count", value: session.toolCount, timestamp },
                                { key: "message_count", value: session.messageCount, timestamp },
                            ],
                            params: [], tags: [],
                        });
                    });

                    // Cancel debounce timer and finalize immediately
                    if (session._finalizeTimer) {
                        clearTimeout(session._finalizeTimer);
                        session._finalizeTimer = null;
                    }
                    await finalizeTrace(session, sessionId, `session:${e.type}`);
                }
            });
        },
    };
};
export default plugin;
