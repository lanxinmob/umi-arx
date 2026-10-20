Please follow [This Notion Site](https://yihuai.notion.site/UMI-ARX-Cup-Task-Deployment-Tutorial-112e4c9a5c4980f1ac20fe4da261d9a1?pvs=4) for detailed deployment instructions.

Local agent interfaces: [read-only observations](observation_mcp/README.md) and
[bounded action MCP](action_mcp/README.md). The action bridge defaults to mock;
read its live commissioning limits before enabling hardware.

## Robot State Startup Errors

For guarded Cartesian checkpoints, start the updated
`python communication/zmq_server_agent_safe.py X5_umi can0` in the ARX5 SDK
environment before starting `scripts/eval_arx5.py`. For `absolute_joint`
checkpoints, use the joint server documented below. In either mode the client
waits for a live controller and at least two robot state samples before
reporting readiness.
On reset, the client initializes its hold target from `ee_cmd_pose` and
`gripper_cmd_pos` in `GET_STATE`, the same reset command pose used by the
guarded server's 2 mm command limit. `ee_pose` remains the measured observation.
Older ZMQ servers without these fields must be restarted from the updated SDK.
If startup times out or the controller exits, inspect the earlier controller
traceback and the robot server terminal; verify the configured robot IP/port
(default port 8765) and CAN connection. A policy server listening on port 8766
does not establish the robot connection.

An empty robot observation buffer is a robot state acquisition failure, not a
model inference failure. Do not substitute zero poses or change the checkpoint
to bypass it. Restart the client after resolving the reported connection or
controller error.

## Absolute-joint ARX5 policy

For a checkpoint whose action representation is `absolute_joint` with seven
values (`q1..q6` in radians followed by gripper width in meters), use the joint
server on port 8765:

```bash
cd /home/hjadmin/projects/arx5-sdk/python
python communication/zmq_joint_server.py X5_umi can0 --port 8765
```

Do not use `communication/zmq_server.py` for this checkpoint; that entry point
serves Cartesian `SET_EE_POSE` requests and does not implement the joint
trajectory protocol.

Run the DGM policy service on port 8766, then start the client from this
repository. The client uses a 5 Hz action clock and submits each future joint
chunk after linear resampling to 100 Hz as one `SET_JOINT_TRAJ` request. The
first waypoint is kept at least one action step (0.2 s at the default 5 Hz
clock) in the future so camera and inference latency cannot collapse the first
transition. The first 0.8 s of a replanned chunk is blended with the previous
chunk by default. The SDK keeps the resulting trajectory and interpolates it in
its 500 Hz servo loop; the client polls feedback at 200 Hz.
With the default `--steps_per_inference 4`, policy replanning is 1.25 Hz (every
0.8 s), while each response contains a 16-step horizon. Overlapping chunks are
blended by default; use `--chunk-blend 0` to disable that boundary smoothing.
