# ARX5 Multimodal Agent Control

This is a parallel control path. It does not use `umi_multitask.ckpt` and does
not replace any existing entry point.

## Process layout

1. `agent_planner_server.py` receives wrist-image history, measured TCP history,
   gripper history, timestamps, and previously accepted actions. It calls the
   OpenAI Responses API and returns one normalized TCP tool-frame increment.
2. `eval_arx5_agent.py` validates the response identity and schema, projects it
   into a bounded absolute TCP target, and retains the original human/agent
   control workflow.
3. `zmq_server_agent_safe.py` is the robot-host guard. It independently checks
   every interpolated EE command before passing it to the ARX5 SDK.

The planner link uses JSON metadata plus raw RGB bytes. It does not deserialize
Python pickle. The robot SDK's existing pickle protocol is retained only on the
loopback-bound robot server. The agent client refuses to start unless port 8765
identifies itself as `arx5_agent_safe`.

## 1. Planner server

First run the deterministic hold backend:

```bash
cd /home/hjadmin/projects/unified_video_action
bash scripts/eval/eval_agent_real.sh --mock
```

For the API backend, explicitly select an image-capable model available to the
account:

```bash
cd /home/hjadmin/projects/unified_video_action
export OPENAI_API_KEY='...'
export OPENAI_MODEL='...'
bash scripts/eval/eval_agent_real.sh --timeout 15
```

The default bind is `tcp://127.0.0.1:8766`. When the planner runs in a separate
container/network namespace, bind the published interface explicitly:

```bash
bash scripts/eval/eval_agent_real.sh --bind tcp://0.0.0.0:8766
```

Do not expose this API-backed port to an untrusted network.

## 2. Robot safety server

Use measured, site-specific base-frame workspace bounds in meters:

```bash
conda activate arx5
cd /home/hjadmin/projects/arx5-sdk/python
python communication/zmq_server_agent_safe.py X5_umi can0 \
  --workspace-min X_MIN Y_MIN Z_MIN \
  --workspace-max X_MAX Y_MAX Z_MAX
```

The server binds `127.0.0.1:8765`, requires `RESET_TO_HOME` before motion,
allows the SDK's reset-only `0.086 m` fully open gripper state, and applies these
default per-command limits:

| Limit | Default |
|---|---:|
| EE translation delta | `0.002 m` |
| wrapped RPY delta | `0.02 rad` |
| gripper delta | `0.002 m` |
| no-message damping timeout | `2.0 s` |

On timeout it enters damping and disarms. It never moves home as a timeout
reaction. A new reset is required to arm it again.

## 3. Agent client

Start with mock plus shadow mode. No checkpoint argument is needed:

```bash
cd /home/hjadmin/projects/umi-arx
python scripts/eval_arx5_agent.py \
  -o data/experiments/uva_agent \
  --no_mirror \
  --task_name 'pick up the mouse and place it on the mouse pad' \
  --gripper-mask
```

Shadow mode suppresses human and agent waypoint submission, but it still starts
the existing `Arx5Env` robot bridge. That bridge performs its normal startup
`RESET_TO_HOME`. Shadow mode is therefore not a no-hardware simulator.

After checking mock/shadow logs and workspace calibration, enable submission
explicitly:

```bash
python scripts/eval_arx5_agent.py \
  -o data/experiments/uva_agent \
  --no_mirror \
  --task_name 'pick up the mouse and place it on the mouse pad' \
  --gripper-mask \
  --arm \
  --max-position-step 0.005 \
  --max-rotation-step 0.035 \
  --max-gripper-step 0.002 \
  --episode-radius 0.12 \
  --min-z-below-start 0.08
```

Treat these as upper bounds, not recommended values for every setup. Reduce
them for first hardware trials. `--min-z-below-start` is relative to the TCP
pose at `C`; it is not a calibrated table-collision boundary.

## Controls

Human-control behavior is retained:

- `C`: start a new agent episode.
- `Q`: hold and exit.
- `M`: move to a matched dataset pose when match options are supplied.
- Existing keyboard, SpaceMouse, or gamepad initial-pose adjustment remains.

Agent-control behavior:

- `S`: cancel an outstanding planner request, submit a measured hold, end the
  episode, and return to human control.
- `Q`: cancel, hold, end the episode, and exit.

As in the original workflow, `S` does not resume the same episode. Pressing `C`
starts a new one.

## Safety behavior

The API model never returns meters, radians, joint values, or absolute poses. It
returns normalized `[-1,1]` tool-frame directions. The client independently
checks finite values, exact fields, confidence, translation/rotation/gripper
step norms, the previous submitted target, episode radius, and downward travel.
Invalid, stale, dark, timed-out, mismatched, or low-confidence results become a
hold. Three consecutive failures return control to the operator.

Each inference schedules at most one micro-step. No long action horizon is
queued, so operator cancellation is checked while waiting for the API.

Logs are written under the selected output directory:

```text
obs/<episode>/<step>.npy
action/<episode>/<step>.npy
```

They include raw planner responses, measured TCP state, safety decisions,
submission state, and action history.

## Residual risk

This path is deliberately conservative, but it is not a certified safety
controller. A monocular wrist image does not establish depth, table height,
collision geometry, payload state, or human proximity. API latency also makes
this a stop-and-observe controller rather than a 5 Hz real-time policy.

Keep the physical emergency stop immediately reachable. Validate workspace
bounds and tool/camera axes with the robot unladen, use mock and shadow logs
first, and do not run unattended. Existing SDK/C++ IK and emergency-state code
was not changed because this implementation is restricted to new files.
