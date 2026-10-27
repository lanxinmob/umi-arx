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

## Keyboard evaluation

`scripts/eval_arx5.py` defaults to `--teleop-device keyboard`. Click the camera
preview to focus it and use an English input layout with Caps Lock off.
For Cartesian checkpoints (including `umi_multitask.ckpt`), hold these keys in
the `Human in control` view:

| Keys | Action |
| --- | --- |
| Up/Down or `w`/`s` | Move along +X/-X |
| Left/Right or `a`/`d` | Move along +Y/-Y |
| PageUp/PageDown or `z`/`x` | Move along +Z/-Z |
| `i`/`k`, `j`/`l`, `u`/`o` | Rotate around end-effector X, Y, Z |
| `f` / `r` | Close / open gripper |
| `f` + `r` | Return the selected arm(s) home |
| `c` | Start policy control |
| `s` during policy control | Stop policy and return to human control |
| `q` during human control | Return home and quit |

On Linux Wayland desktops, the script uses Qt's `xcb` platform for the preview
through XWayland because `pynput` listens for X11 keyboard events. This setting
only affects this process. Keep the preview focused: keys typed into a native
Wayland terminal are invisible to the X11 listener.
Absolute-joint checkpoints use hand dragging instead of keyboard movement, as
described below.

## Absolute-joint ARX5 policy

For a checkpoint whose action representation is `absolute_joint` with seven
values (`q1..q6` in radians followed by gripper width in meters), use the joint
server on port 8765:

```bash
cd /home/hjadmin/projects/arx5-sdk/python
python communication/zmq_joint_server.py X5_umi can0 --port 8765
```

The joint server enters gravity-compensated drag mode when `eval_arx5.py`
returns control with `s`. If the released arm rises or leans backward, restart
the server with a lower feedforward multiplier, for example
`--gravity-scale 0.85`; keep the arm supported while tuning this value. The
server clears the active policy trajectory before switching to drag, so episode
recording cannot continue to drive the arm during the handoff.

Do not use `communication/zmq_server.py` for this checkpoint; that entry point
serves Cartesian `SET_EE_POSE` requests and does not implement the joint
trajectory protocol.

Run the DGM policy service on port 8766, then start the client from this
repository. The client uses a 5 Hz action clock and submits each future joint
chunk after linear resampling to 100 Hz as one `SET_JOINT_TRAJ` request. The
first waypoint is kept at least two action steps (0.4 s at the default 5 Hz
clock) in the future so camera and inference latency cannot collapse the first
transition. The first 0.8 s of a replanned chunk is blended with the previous
chunk by default. The SDK anchors each replacement to measured joint feedback
and uses cubic interpolation in its 500 Hz servo loop; the client polls
feedback at 200 Hz.
Press `q` in the human-control view to stop recording and return the arm to the
SDK home pose before exiting.
With the default `--steps_per_inference 4`, policy replanning is 1.25 Hz (every
0.8 s), while each response contains a 16-step horizon. Overlapping chunks are
blended by default; use `--chunk-blend 0` to disable that boundary smoothing.
The joint server ramps HOLD gains for 0.3 s after `C`; keep the arm still until
the `HOLD confirmed` message appears.

`scripts/eval_arx5.py` applies the manually selected field-of-view scale to each
policy observation batch and desktop preview. Pass `--image-scale 0.7` for
zoom-in, `--image-scale 1.3` for zoom-out, or omit it to use `1.0` without a
FOV transform. Values are limited to `0.7–1.3`. Camera preprocessing applies
this scale before `--no-mirror` and `--gripper-mask`, then feeds the same
processed frame to the policy and display; the selected value is saved as
`image_scale_factor` with each recorded observation.
