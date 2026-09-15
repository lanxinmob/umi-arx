# UMI ARX Action MCP

Small, bounded actions for the task arm **X5_umi / can0**. `can1` is never used.
Protocol: `umi-arx.action.v1`. The existing observation MCP stays read-only.

This is commissioning software, not a certified robot safety controller or a
trained VLA policy. It does not locate objects, avoid collisions, or implement
a pick-and-place planner. Software checks do not establish physical safety.

## Run without hardware

From `/home/hjadmin/projects/umi-arx`, start the bridge:

```bash
conda run --no-capture-output -n arx5 python action_mcp/bridge.py
```

It defaults to **mock**, binds only `127.0.0.1:8766`, and never imports the SDK
or opens CAN. The mock is a deterministic protocol simulator, not robot physics.
In another terminal, start the MCP proxy:

```bash
conda run --no-capture-output -n umi-arx python action_mcp/server.py
```

MCP host configuration (the bridge must be running separately):

```json
{
  "mcpServers": {
    "umi-arx-action": {
      "command": "/home/hjadmin/miniconda3/envs/umi-arx/bin/python",
      "args": ["/home/hjadmin/projects/umi-arx/action_mcp/server.py"]
    }
  }
}
```

Dependencies are `numpy` and `pyzmq`, already installed in both environments.
No MCP package or model API key is required. All modules support Python 3.9.
Use `--port` on both processes to change the port.

## Tools and sequence

1. `get_action_contract`: verify `source`, frames and limits.
2. `arm_action_session`: ramp position gains at the current measured pose over
   one second. It does not reset the arm. Returns `ready` or a failure.
3. Acquire a fresh image with the observation MCP and `get_action_state` with
   the action MCP. Use the returned `session`, `next_sequence`, `state_token`.
4. `execute_action`: submit one bounded delta. The MCP tool waits for measured
   completion or failure, and returns the target, actual pose and errors.
5. Repeat observations/actions without requiring a human confirmation for each
   step. End with `stop_action_session`.

Example `execute_action` arguments; the identifiers must come from step 3:

```json
{
  "session": "SESSION_FROM_STATE",
  "sequence": 1,
  "state_token": "TOKEN_FROM_STATE",
  "frame": "base",
  "delta_xyz_m": [0, 0, 0.002],
  "delta_rotvec_rad": [0, 0, 0],
  "gripper_delta_m": 0,
  "duration_s": 1
}
```

The bridge wire envelope is JSON, never pickle:

```json
{"protocol":"umi-arx.action.v1","command":"state","data":{}}
```

Commands: `info`, `state`, `arm`, `action`, `stop`. `action.data` uses the schema
above; other commands require `{}`. Replies are `{"ok":true,"result":{...}}`
or `{"ok":false,"error":"..."}`. Direct bridge actions initially return
`running`, NOT success. Read `state.last_action.status` for the outcome.
Only trusted local clients should access this endpoint; session tokens prevent
stale/replayed actions, not malicious local access.

## Coordinate contract

- Position deltas: **meters**. Rotation deltas: right-hand **rotation vectors
  in radians**, not three Euler-angle increments. Positive gripper delta opens.
- `base`: SDK URDF `base_link`; `p_target = p + dp`, `R_target = dR * R`.
- `tool`: SDK URDF `eef_link` at action start; `p_target = p + R * dp`,
  `R_target = R * dR`. This is **not** the observation TCP frame or camera frame.
- Feedback `ee_pose_base_xyz_rpy`: XYZ in meters, RPY in radians with
  `R = Rz(yaw) * Ry(pitch) * Rx(roll)`. Do not feed TCP rotvec as SDK RPY.
- `X5_umi.urdf` places `eef_link` 0.225 m along link6's X axis. Base Z is the
  URDF up axis; its agreement with the physical installation still needs
  commissioning. Pixel directions are not assumed to match robot axes.

## Fixed protection limits

Limits live in `control.Limits`; model requests cannot change them or gains.

| Check | Default |
| --- | --- |
| Single translation / rotation / gripper delta | 2 mm / 0.02 rad / 2 mm |
| Smallest nonzero translation / rotation / gripper delta | 0.5 mm / 0.001 rad / 0.5 mm |
| Planned translation / rotation / gripper speed | 5 mm/s / 0.05 rad/s / 4 mm/s |
| Joint speed | 0.15 rad/s |
| Duration / settling deadline | 1-5 s / additional 1 s |
| Motor feedback age / control-loop gap | 100 ms / 100 ms |
| State-token validity | 2 s, also invalidated by intervening motion |
| Ready-state idle lease / entire session | 30 s / 120 s |
| Cumulative distance from session origin | 30 mm plus explicit XYZ bounds |
| Joint load thresholds | [10, 12, 10, 3, 3, 3] SDK Nm |
| Gripper load threshold | 0.25 in SDK gripper-torque units |

The bridge ticks independently of MCP/model requests, approximately every 5 ms.
It preflights 100 joint-path intervals, IK/FK residual and smoothstep peak speed.
Runtime checks cover state finiteness, joint/workspace limits, measured speed,
load, tracking error, SDK latch and per-motor freshness. Raw CAN monitoring uses
kernel receive timestamps, ignores local TX, and rejects missing timestamps or
receive-queue overflow. Neither the SDK sampling timestamp nor `GET_STATE`
alone proves motor feedback freshness.

Final tolerances are at most 0.5 mm / 0.005 rad / 1 mm, tightened to 25% of each
nonzero requested displacement. The target must remain reached for 150 ms;
staying still cannot pass a microstep simply because it is smaller than the
absolute tolerance. High contact load is a failure, NOT a successful grasp.

Only one action is allowed at a time. Identical latest requests are idempotent;
other old sequences are rejected. State/info polling does not renew the idle
lease. Model loss during a move allows only the already accepted, finite action
to finish; then the idle lease expires. MCP EOF requests stop if that proxy
armed the session. Explicit stop, feedback failure, load/tracking failure or
deadline expiration latches damping. There is no remote reset, fault-clear,
gain-change or auto-rearm command. Restart the bridge only after resolving the
cause and checking the scene.

## Live commissioning

**Not yet passed on hardware; see the latest commissioning result below.**
Do not confuse mock or offline kinematics results with physical direction,
tracking or stop tests.
Ensure the arm is supported, the local movement envelope is clear, and the
physical stop is accessible. Stop every other controller on can0. The file
lock prevents duplicate instances of this bridge; the initial CAN sender check
does not prevent an unrelated program from starting later.

Live startup **enables motors**, even before the MCP arm tool. It requires
explicit enable acknowledgement and bounds in `base_link` meters. Replace the
six `X_MIN ... Z_MAX` placeholders with the assessed local commissioning
envelope, not example values or guessed object coordinates:

```bash
conda run --no-capture-output -n arx5 python action_mcp/bridge.py \
  --live --allow-hardware-enable \
  --workspace-min X_MIN Y_MIN Z_MIN \
  --workspace-max X_MAX Y_MAX Z_MAX
```

The backend loads only the patched local SDK in
`/home/hjadmin/projects/arx5-sdk/python`, not the installed wheel. Gripper
calibration remains `open_readout=-3.4`, width 0.086 m. No motor zeroing or
calibration-sign changes are performed. Arming requires measured width in
[0, 0.086] m; out-of-range readings are not clamped away.

Gravity feedforward is disabled throughout this service because this SDK's
ordinary `set_to_damping()` does not disable it. Gains are ramped from zero;
an arm that cannot hold within the tracking limits will stop. **Damping is not
power removal, and an unsupported arm or object can settle or fall.**

Python/OS stalls, process SIGKILL, host power loss and a failed CAN path cannot
be guaranteed safe by this software. Motor-side command timeout behavior and
physical stop effectiveness must be verified independently. TCP bounds do not
check swept links, fingers, cables, people or the held object's geometry.
Measured load thresholds are not a certified force limit.

Connect the existing observation MCP to this bridge, without a second motor
controller, using:

```bash
conda run --no-capture-output -n umi-arx \
  python observation_mcp/server.py --action-port 8766
```

This changes only its read-only telemetry source; the camera config and
`umi-arx.observation.v1` output stay unchanged. The live observation provider
rejects mock, stale or unhealthy bridge feedback. **Do not also launch the old
controller on 8765/can0.** Without `--action-port`, the observation MCP retains
its legacy configuration, including the old watchdog's read-as-activity issue.

## Verification performed

- 16 temporary unittest cases passed, including 200 frame-composition cases
  against SciPy, fault/staleness injection, loop and idle deadlines, replay,
  invalid payloads, workspace limits, IK rejection and small-step false success.
- The read-only observation adapter preserves its output fields and legacy
  configuration; it rejects mock and unhealthy action-bridge feedback.
- Actual stdio MCP to JSON bridge to mock execution passed, including EOF stop
  and normal process teardown. Temporary test sources are removed after use.
- Patched SDK / real URDF were exercised without creating a CAN controller.
  From the historical joint pose, base XYZ 0.5 mm targets passed preflight;
  0.005 rad rotations around each base axis were correctly rejected because
  native IK returned excessive orientation residual. Full six-axis physical
  control is **not** verified. The native solver header uses `EPS_=1e-4`; the
  guard independently verifies the unweighted geometric residual.
- The initial software verification did not enable hardware or test live motion.

### Live commissioning, 2026-09-15

The SDK's startup clear/enable sequence produced a DM status-0 reply, for
example `057f807fe7fc1e1b`. Status 0 is disabled, not a hardware error. The CAN
monitor now allows it only during initialization, then requires fresh feedback
from all seven motors and enabled status from all four DM motors. Actual fault
codes still latch during startup, and any subsequent disable latches a fault.
Eight focused, hardware-free checks covered these transitions and passed;
no temporary test source files were retained.

After that fix, startup and the one-second holding-gain ramp passed on can0.
A 1 mm base-Z upward request was accepted. Measured upward travel was only
0.114 mm; final position error was 0.886 mm (allowed 0.25 mm for this request).
The action failed with `target not reached before deadline`, entered latched
damping, and the controller disconnected normally. The return action and
pick-and-place were not executed. Gripper width remained about 18.45 mm.

Offline FK of the corresponding IK target predicts 0.947 mm upward travel,
with 0.093 mm position residual. Actual transmitted motor targets and physical
position-loop response still need diagnosis; do not assume the cause is gain,
friction or gravity without further evidence. Accurate motion and a successful
pick-and-place remain unverified. Software damping was exercised; a physical
emergency stop/power-loss test was not performed.

Files: `control.py` defines geometry and execution rules; `backend.py` isolates
mock/SDK/CAN access; `bridge.py` owns execution; `server.py` exposes MCP tools.
