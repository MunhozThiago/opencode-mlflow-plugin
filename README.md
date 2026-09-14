# opencode-mlflow-plugin

MLflow plugin for [opencode](https://opencode.ai) - tracks sessions and tool executions in your local MLflow server.

## Features

- Automatically tracks each opencode session as an MLflow run
- Logs tool execution counts and errors
- Tracks session duration and message counts
- Optional detailed logging of tool inputs/outputs
- Connects to your local MLflow tracking server

## Installation

### Via npm (recommended)

Add to your `opencode.json`:

```json
{
  "plugin": ["opencode-mlflow-plugin"]
}
```

### Local development

1. Clone this repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Build the plugin:
   ```bash
   npm run build
   ```
4. Reference locally in `opencode.json`:
   ```json
   {
     "plugin": ["./path/to/opencode-mlflow-plugin"]
   }
   ```

## Configuration

### Prerequisites

1. Install MLflow:
   ```bash
   pip install mlflow
   ```

2. Start the MLflow tracking server:
   ```bash
   mlflow ui
   ```
   This starts the UI at http://localhost:5000

### Plugin Options

Configure the plugin in `opencode.json`:

```json
{
  "plugin": [
    ["opencode-mlflow-plugin", {
      "trackingUri": "http://localhost:5000",
      "experimentName": "opencode-sessions",
      "logToolDetails": false,
      "logMessages": false
    }]
  ]
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `trackingUri` | string | `http://localhost:5000` | MLflow tracking server URL |
| `experimentName` | string | `opencode-sessions` | Name of the MLflow experiment |
| `logToolDetails` | boolean | `false` | Log individual tool execution counts |
| `logMessages` | boolean | `false` | Log chat message counts |

## What Gets Tracked

Each opencode session creates an MLflow run with:

### Parameters
- `session_id`: Unique identifier for the opencode session

### Metrics
- `tool_count`: Number of tool executions
- `message_count`: Number of chat messages
- `error_count`: Number of failed tool executions
- `duration_ms`: Session duration in milliseconds
- `final_tool_count`: Total tools executed
- `final_message_count`: Total messages
- `final_error_count`: Total errors

### Tags
- `opencode.session_id`: Session identifier
- `opencode.plugin_version`: Plugin version

## Viewing Results

1. Start MLflow UI:
   ```bash
   mlflow ui
   ```

2. Open http://localhost:5000 in your browser

3. Select the "opencode-sessions" experiment

4. Click on any run to see metrics, parameters, and tags

## Development

### Build

```bash
npm run build
```

### Watch mode

```bash
npm run dev
```

### Publish to npm

```bash
npm publish
```

## License

MIT
