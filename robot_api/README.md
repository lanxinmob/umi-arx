# Robot API

One MCP stdio endpoint and one Python class for camera history, measured arm pose,
gripper width, and bounded relative commands. Protocol: `umi-arx.robot.v1`.
Target identity is **X5_umi / can0**, never the demonstration arm on can1.

## Start

Run from the repository root, using the existing camera environment (Python 3.10,
numpy, opencv-python, pyzmq). No additional MCP SDK is required.

```bash
cd /home/hjadmin/projects/umi-arx
/home/hjadmin/miniconda3/envs/umi-arx/bin/python -m robot_api.server
```

Default startup creates an isolated **mock** bridge on a free loopback port.
It does not import the hardware SDK or open CAN/camera devices. Images are synthetic
fixtures; proprioception is sampled from the running mock bridge at 5 Hz, so
history reflects actual mock commands rather than fabricated past trajectories.
This tests software, not camera geometry, grasping or physical dynamics.

MCP client configuration (set its working directory to this repository):

```json
{
  "mcpServers": {
    "umi-arx-robot": {
      "command": "/home/hjadmin/miniconda3/envs/umi-arx/bin/python",
      "args": ["-m", "robot_api.server"],
      "cwd": "/home/hjadmin/projects/umi-arx"
    }
  }
}
```

For clients without `cwd`, set `PYTHONPATH=/home/hjadmin/projects/umi-arx` in that
server's environment instead. JSON-RPC messages are newline-delimited on stdin/stdout;
diagnostics go to stderr. EOF/SIGINT/SIGTERM requests damping if this API engaged
the session and terminates its owned mock process. Hard kills rely on the bridge's
30 s idle and 120 s session deadlines; this is not an emergency-stop substitute.

## Tools

| Tool | Arguments | Result |
| --- | --- | --- |
| `status` | none | Connection, source, contract/limits, feedback, readiness blockers |
| `observe` | `sample_count=4`, `history_seconds=2.4`, `wait_timeout_seconds=5` | 1-8 distinct chronological images and measured state |
| `arm` | none | Explicit holding engagement, then a fresh observation |
| `move` | `request_id`, `observation_id`, `delta_xyz_m`; optional `delta_rotvec_rad=[0,0,0]`, `frame="base"`, `duration_s=5` | One bounded action and post-action observation |
| `gripper` | `request_id`, `observation_id`, `delta_m`, optional `duration_s=5` | Positive opens, negative closes; post-action observation |
| `stop` | none | Latched damping, **not motor power removal** |

Call `status`, explicitly `arm`, inspect its observation, then `move` or `gripper`.
An `observation_id` is single-use, must be the latest issued ID, must match the
bridge session/sequence, and expires 10 s after the newest image capture.
Call `observe` again when necessary; reads do not extend the controller lease.
The observation stream itself must still be healthy and fresh (1 s by the
workspace configuration) immediately before motion, even if an ID has not expired.
Only the API that armed the bridge can submit motion. It never automatically
rearms or resets a stopped/faulted controller.

Example `tools/call` after `arm` supplies an observation ID:

```json
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"move","arguments":{"request_id":"lift-001","observation_id":"ID_FROM_ARM","delta_xyz_m":[0,0,0.005],"duration_s":5}}}
```

Every tool returns MCP `content`, `structuredContent`, and `isError`. The first
content block is the same API JSON as `structuredContent`. When images exist,
`observation` contains the unchanged `umi-arx.observation.v1` packet;
`observation.samples[i].image.content_index` identifies the actual MCP image block.
The `state` is a separate, more recent controller read, not a claim that all
historical images were captured at the current pose.

Pose units are metres/radians. Motion uses SDK `eef_link`, **not** observation TCP
or camera axes. `delta_rotvec_rad` is a right-hand rotation vector, not Euler
angles. Measured `ee_pose_base_xyz_rpy` uses XYZ plus roll/pitch/yaw (`Rz Ry Rx`).
The gripper has one width DOF; its world pose follows the arm end effector.

Read the active contract from `status`: visual mode permits at most 5 mm
translation per command, strict mode 2 mm, rotation 0.02 rad, width increment
2 mm, and duration 1-5 s. Visual translations cannot combine with rotation.
Commands are never silently clipped, subdivided, or retried. Existing speed,
feedback, load, workspace, direction, tracking and session guards remain active.
`action.status="executed"` means bounded progress, **not target arrival**;
inspect `target_reached`, measured feedback and images. Gripper command success
does not mean an object was grasped. There is no pretend `pick_place` tool.

Retry an uncertain API response using the **same request_id and identical
arguments**. Within that API process the stored response is returned without
moving again. Changed arguments with the same ID are rejected. IDs are not
deduplicated across process restarts: reconnect requires explicit engagement of
a new, disarmed bridge; never replay an old command after restart. Transport or
post-action observation failure requests stop and is never reported as success.

Images, observation metadata, contract, and pending/final action receipts are
written under `robot_api/runs/<session-id>/` (gitignored). `image_files` contains
absolute paths to all requested JPEGs, not just a montage. `--output PATH` changes
the parent directory. Files are retained for diagnosis; test fixtures need not be
kept. One serialized API instance per controlling model session is intended.

## Python

```python
from robot_api.api import RobotAPI
from robot_api.server import start_mock, stop_mock

bridge, port = start_mock()
try:
    with RobotAPI(port) as robot:
        observation = robot.arm()["structuredContent"]
        result = robot.move(
            request_id="lift-001",
            observation_id=observation["observation_id"],
            delta_xyz_m=[0, 0, 0.005],
        )
        print(result["structuredContent"])
finally:
    stop_mock(bridge)
```

## Physical Acceptance

**Not yet accepted for immediate cup-to-tray execution.** The physical gripper
tripped the former measured 6 mm/s guard in four trials. Recorded peaks reached
7.50-9.17 mm/s despite smooth commands. Lower ramp speed and alternative gains
did not resolve it. Stable loaded holding and a transfer workspace beyond the
current 30 mm commissioning radius are also unverified. Restarting an API does
not remove these facts. See [commissioning.json](commissioning.json) and
[physical test history](../action_mcp/README.md).

The measured gripper-speed threshold is now independently set to a provisional
12 mm/s (`contract.limits.gripper_measured_speed_limit_m_s`). The command-speed
limit remains 4 mm/s (`gripper_speed_m_s`), including the SDK ramp. The subsequent
live 1 mm / 5 s opening retest stayed below the speed threshold but moved only
0.116 mm and failed target completion. It does not clear the recorded failed
commissioning or establish reliable grasping. See the latest physical test history.

The API can attach **read-only** to an explicitly started existing live bridge:

```bash
/home/hjadmin/miniconda3/envs/umi-arx/bin/python -m robot_api.server --live --port 8766
```

This command never starts or enables hardware. Without a live bridge it fails
with a connection error; it never falls back to mock. Starting the underlying
hardware bridge enables motors even before `arm`, so follow the separate
[action bridge startup procedure](../action_mcp/README.md), with freshly checked
clearance and explicit bounds. Do not run the legacy controller concurrently.

`commissioning.json` is operator-owned evidence, not a tuning switch. After the
physical issue is diagnosed and corrected, each failed/unverified item needs an
actual passing test and a retained evidence reference; only then may its status
be recorded as `passed`. The model tools cannot approve or edit this record.
Until then `arm/move/gripper` reject live operation, while `observe/status/stop`
remain available. `task_ready` can be true only for live, healthy, fresh,
explicitly armed operation with passing commissioning records. `simulation_ready`
is a separate flag; neither flag certifies collision safety or successful grasp.

## Verification Record (2026-09-15)

Executed against the existing Python environment, without enabling hardware:

- TCP conversion: finite identity/pi cases and matrix round trips; reusable client import.
- Python API: four distinct timestamped JPEGs; engagement; 5 mm motion; gripper
  increment reflected in sampled state; identical replay without a second action.
- Rejection paths: wrong live/mock source, replaced bridge session, wrong schema,
  oversized/nonfinite/boolean commands, expired/superseded observation, stale stream.
- Fault paths: lost action reply and post-action camera failure request stop;
  known live commissioning failures block engagement.
- Real stdio MCP subprocess: initialize, six tool schemas, image indices, move,
  gripper, replay; notification-shaped action requests do not execute.
- Lifecycle: EOF and SIGTERM stop the engaged mock session and reap its child bridge.

The first short-duration test correctly hit the planned-speed guard. The facade
now reports a confirmed pre-execution rejection separately from an unknown
transport outcome; the normal 5 s move passed without changing the guard.
Tests used inline scripts and temporary directories, which were removed.
Physical cup-to-tray acceptance remains **not passed** for the reasons above.
