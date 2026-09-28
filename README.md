Please follow [This Notion Site](https://yihuai.notion.site/UMI-ARX-Cup-Task-Deployment-Tutorial-112e4c9a5c4980f1ac20fe4da261d9a1?pvs=4) for detailed deployment instructions.

Local agent interfaces: [read-only observations](observation_mcp/README.md) and
[bounded action MCP](action_mcp/README.md). The action bridge defaults to mock;
read its live commissioning limits before enabling hardware.

## Robot State Startup Errors

Start `python communication/zmq_server.py X5_umi can0` in the ARX5 SDK
environment before starting `scripts/eval_arx5.py`. The client waits for a live
controller and at least two robot state samples before reporting readiness.
If startup times out or the controller exits, inspect the earlier controller
traceback and the robot server terminal; verify the configured robot IP/port
(default port 8765) and CAN connection. A policy server listening on port 8766
does not establish the robot connection.

An empty robot observation buffer is a robot state acquisition failure, not a
model inference failure. Do not substitute zero poses or change the checkpoint
to bypass it. Restart the client after resolving the reported connection or
controller error.
