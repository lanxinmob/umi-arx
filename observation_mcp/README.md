# UMI ARX Observation MCP

Read-only MCP server for synchronized camera, arm, and gripper observations.
The fixed payload protocol is `umi-arx.observation.v1`.

## Run

The checked-in configuration matches camera index `0` from the existing ARX5
evaluation commands and the loopback robot server on port `8765`.

Start the robot-host state server first, then run the MCP server:

```bash
conda run --no-capture-output -n umi-arx \
  python observation_mcp/server.py
```

The MCP process only sends `GET_STATE`; it never resets or commands the arm.
For a hardware-free smoke test:

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

The server continuously reads the configured UVC device so frames remain fresh,
samples complete image and robot state pairs at 5 Hz, and keeps 10 seconds in
memory. It implements MCP JSON-RPC over stdio and exposes no robot command.
Never write logs to stdout because stdout is reserved for MCP messages.

`GET_STATE` currently counts as activity in the ARX5 safe server watchdog. Do
not use this observation service as a substitute for the control process's
action heartbeat, and do not leave an armed robot unattended while it runs.
