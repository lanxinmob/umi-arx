# UMI ARX Observation MCP

Read-only MCP server for synchronized camera, arm, and gripper observations.
The fixed payload protocol is `umi-arx.observation.v1`.

For the new [bounded action bridge](../action_mcp/README.md), run this server
with `--action-port 8766`. It then reads JSON state from that bridge instead of
the legacy controller below. Do not run both motor controllers on can0. Camera
configuration and the observation protocol remain unchanged; mock or unhealthy
action-bridge feedback is rejected in live observation mode.

## Run

The checked-in [config.json](config.json) matches camera index `0` from the
existing ARX5 evaluation commands and the loopback robot server on port `8765`.

On the robot host, start the guarded server with measured workspace bounds:

```bash
conda activate arx5
cd /home/hjadmin/projects/arx5-sdk/python
python communication/zmq_server_agent_safe.py X5_umi can0 \
  --workspace-min X_MIN Y_MIN Z_MIN \
  --workspace-max X_MAX Y_MAX Z_MAX
```

This server initializes the hardware. Keep the workspace clear and the
physical emergency stop reachable. In a second terminal, start the MCP server:

```bash
conda run --no-capture-output -n umi-arx \
  python observation_mcp/server.py
```

The MCP process only sends `GET_STATE`; it never resets or commands the arm.
For a hardware-free smoke test, add `--mock`:

```bash
conda run --no-capture-output -n umi-arx \
  python observation_mcp/server.py --mock
```

An MCP host can launch live mode with:

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
        "/home/hjadmin/projects/umi-arx/observation_mcp/server.py"
      ]
    }
  }
}
```

Add `"--mock"` after `server.py` in that array for a hardware-free client
integration test. `--no-capture-output` is required because MCP reserves stdout
for protocol messages.

Edit `config.json` when the UVC by-id path, resolution, or robot port changes.
The default sampler produces 224x224 JPEG observations at 5 Hz and retains 10
seconds. The robot endpoint is deliberately restricted to loopback because its
existing transport deserializes Python objects.

## MCP surface

- `get_observation`: returns one JSON text block followed by 1-8 JPEG image
  blocks in chronological order. `image.content_index` addresses the matching
  MCP content block. Its defaults return four samples spanning 2.4 seconds.
- `get_observation_status`: returns source readiness without image data.
- `umi-arx://observation/schema/v1`: JSON Schema for the metadata packet.

Each sample contains capture time, relative time, EE pose in base-frame XYZ/RPY,
TCP pose in base-frame XYZ/rotation-vector, joint position/velocity/torque, and
gripper width/velocity/torque. Units are encoded in the field names. Read the
schema resource for the exact fixed contract.

`GET_STATE` currently counts as activity in the ARX5 safe server watchdog. Do
not use this observation service as a substitute for the control process's
action heartbeat, and do not leave an armed robot unattended while it runs.
