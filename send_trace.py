#!/usr/bin/env python3
"""
Sends trace/span to MLflow using the MLflow Python SDK.
Reads JSON from stdin, outputs JSON result to stdout.
"""
import sys
import os
import json
import time

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"

def main():
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)

        tracking_uri = data["tracking_uri"]
        experiment_id = str(data["experiment_id"])
        trace_id = data["trace_id"]
        span_id = data["span_id"]
        name = data.get("name", "opencode-turn")
        kind = data.get("kind", 3)
        start_time_ns = int(data["start_time_ns"])
        end_time_ns = int(data["end_time_ns"])
        attributes = data.get("attributes", {})
        status_code = data.get("status_code", 0)
        request_preview = data.get("request_preview")
        response_preview = data.get("response_preview")

        import mlflow
        from mlflow.entities import SpanStatusCode

        mlflow.set_tracking_uri(tracking_uri)

        # Parse inputs/outputs from attributes
        inputs = {}
        outputs = {}
        if request_preview:
            try:
                inputs = json.loads(request_preview)
            except (json.JSONDecodeError, TypeError):
                inputs = {"request": request_preview}
        if response_preview:
            try:
                outputs = json.loads(response_preview)
            except (json.JSONDecodeError, TypeError):
                outputs = {"response": response_preview}

        # Build GenAI semantic convention attributes
        span_attributes = {}

        # Core opencode metadata
        if "opencode.session_id" in attributes:
            span_attributes["opencode.session_id"] = attributes["opencode.session_id"]
        if "opencode.agent" in attributes:
            span_attributes["opencode.agent"] = attributes["opencode.agent"]
        if "opencode.model" in attributes:
            span_attributes["opencode.model"] = attributes["opencode.model"]
            # Also set the GenAI model name attribute
            span_attributes["mlflow.model"] = attributes["opencode.model"]
        if "opencode.provider" in attributes:
            span_attributes["opencode.provider"] = attributes["opencode.provider"]

        # Token counts (GenAI semantic conventions)
        input_tokens = attributes.get("opencode.input_tokens", 0)
        output_tokens = attributes.get("opencode.output_tokens", 0)
        total_tokens = attributes.get("opencode.total_tokens", 0)
        reasoning_tokens = attributes.get("opencode.reasoning_tokens", 0)
        cache_read = attributes.get("opencode.cache_read", 0)
        cache_write = attributes.get("opencode.cache_write", 0)
        if input_tokens:
            span_attributes["llm.token_count.prompt"] = input_tokens
            span_attributes["mlflow.inputTokens"] = input_tokens
        if output_tokens:
            span_attributes["llm.token_count.completion"] = output_tokens
            span_attributes["mlflow.outputTokens"] = output_tokens
        if total_tokens:
            span_attributes["llm.token_count.total"] = total_tokens
            span_attributes["mlflow.totalTokens"] = total_tokens
        if reasoning_tokens:
            span_attributes["llm.token_count.reasoning"] = reasoning_tokens
        if cache_read:
            span_attributes["llm.token_count.cache_read"] = cache_read
        if cache_write:
            span_attributes["llm.token_count.cache_write"] = cache_write
        # Full token usage breakdown (as dict for MLflow aggregation)
        span_attributes["mlflow.chat.tokenUsage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "total_tokens": total_tokens,
        }

        # Cost
        cost = attributes.get("opencode.cost_usd", 0)
        if cost:
            span_attributes["opencode.cost_usd"] = cost

        # Finish reason
        finish_reason = attributes.get("opencode.finish_reason", "unknown")
        if finish_reason and finish_reason != "unknown":
            span_attributes["llm.response.finish_reason"] = finish_reason

        # Generation parameters
        if "opencode.temperature" in attributes:
            span_attributes["llm.request.temperature"] = attributes["opencode.temperature"]
        if "opencode.top_p" in attributes:
            span_attributes["llm.request.top_p"] = attributes["opencode.top_p"]
        if "opencode.top_k" in attributes:
            span_attributes["llm.request.top_k"] = attributes["opencode.top_k"]
        if "opencode.max_tokens" in attributes:
            span_attributes["llm.request.max_tokens"] = attributes["opencode.max_tokens"]

        # Store prompt/completion in attributes for UI display
        if request_preview:
            span_attributes["mlflow.spanInputs"] = request_preview
            span_attributes["llm.prompt_template.template"] = request_preview
        if response_preview:
            span_attributes["mlflow.spanOutputs"] = response_preview

        # Create client and start trace with inputs
        client = mlflow.MlflowClient()

        root_span = client.start_trace(
            name=name,
            span_type="LLM",
            inputs=inputs if inputs else None,
            attributes=span_attributes if span_attributes else None,
            experiment_id=experiment_id,
            start_time_ns=start_time_ns,
        )

        # Build events list for LLM prompt/completion display in UI
        # Note: MLflow 3.16.0 end_trace doesn't support events param,
        # so we store prompt/completion in span attributes instead

        # End the trace with outputs and status
        status = SpanStatusCode.OK if status_code == 0 else SpanStatusCode.ERROR
        client.end_trace(
            trace_id=root_span.trace_id,
            end_time_ns=end_time_ns,
            outputs=outputs if outputs else None,
            status=status,
        )

        # With MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=false, end_trace() is
        # synchronous — the trace is written to the server before returning.
        # No additional flush needed.

        sys.stdout.write(json.dumps({
            "success": True,
            "trace_id": root_span.trace_id,
            "span_id": root_span.span_id,
        }))
        sys.stdout.write("\n")
        sys.stdout.flush()

    except Exception as e:
        sys.stdout.write(json.dumps({"error": str(e)}))
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
