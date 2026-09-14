# UMI ARX Observation MCP

Read-only MCP server for synchronized camera, arm, and gripper observations.
The fixed payload protocol is `umi-arx.observation.v1`.

## Mock smoke test

The server uses the existing `umi-arx` Conda environment and stdio transport:

```bash
conda run --no-capture-output -n umi-arx \
  python observation_mcp/server.py --mock
```

An MCP host can launch it with:

```json
{
  "mcpServers": {
    "umi-arx-observation": {
      "command": "conda",
      "args": [
        "run",
        "--no-capture-output",
        "-n",
        "umi-arx",
        "python",
        "/home/hjadmin/projects/umi-arx/observation_mcp/server.py",
        "--mock"
      ]
    }
  }
}
```

## MCP surface

- `get_observation`: returns one JSON text block followed by 1-8 JPEG image
  blocks in chronological order. `image.content_index` addresses the matching
  MCP content block.
- `get_observation_status`: returns source readiness without image data.
- `umi-arx://observation/schema/v1`: JSON Schema for the metadata packet.

The server implements MCP JSON-RPC over stdio and exposes no robot command.
Never write logs to stdout because stdout is reserved for MCP messages.
