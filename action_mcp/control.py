"""Bounded relative actions and a model-independent execution state machine."""

from dataclasses import asdict, dataclass, replace
from collections import deque
import math
import time
import uuid

import numpy as np


PROTOCOL = "umi-arx.action.v1"


def scalar(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a finite JSON number")
    return float(value)


def vector(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(name + " must be a JSON array of length 3")
    return np.array([scalar(v, name) for v in value])


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def matrix_rpy(m):
    pitch = math.atan2(-m[2, 0], math.hypot(m[0, 0], m[1, 0]))
    if abs(math.cos(pitch)) < 1e-4:
        raise ValueError("RPY singularity; action rejected")
    return np.array([math.atan2(m[2, 1], m[2, 2]), pitch,
                     math.atan2(m[1, 0], m[0, 0])])


def rotvec_matrix(v):
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle)*k + (1-math.cos(angle))*(k @ k)


def rotation_error(a, b):
    relative = rpy_matrix(a).T @ rpy_matrix(b)
    return math.acos(float(np.clip((np.trace(relative)-1)/2, -1, 1)))


@dataclass(frozen=True)
class Limits:
    position_step_m: float = .002
    rotation_step_rad: float = .02
    gripper_step_m: float = .002
    translation_speed_m_s: float = .005
    rotation_speed_rad_s: float = .05
    gripper_speed_m_s: float = .004
    joint_speed_rad_s: float = .15
    motor_age_s: float = .1
    snapshot_age_s: float = 2.
    loop_gap_s: float = .1
    idle_timeout_s: float = 30.
    session_timeout_s: float = 120.
    radius_m: float = .03
    position_tracking_m: float = .003
    rotation_tracking_rad: float = .025
    joint_tracking_rad: float = .04
    gripper_tracking_m: float = .004
    position_tolerance_m: float = .0005
    rotation_tolerance_rad: float = .005
    gripper_tolerance_m: float = .001
    settle_timeout_s: float = 1.
    minimum_position_step_m: float = .0005
    minimum_rotation_step_rad: float = .001
    minimum_gripper_step_m: float = .0005
    settle_window_s: float = .2
    settle_position_m: float = .00025
    settle_joint_rad: float = .0008
    settle_rotation_rad: float = .0015
    settle_gripper_m: float = .0001


def action_target(payload, pose, width, limits):
    if payload.get("frame") not in ("base", "tool"):
        raise ValueError("frame must be base or tool (SDK eef_link, not camera/TCP)")
    dp = vector(payload["delta_xyz_m"], "delta_xyz_m")
    dr = vector(payload["delta_rotvec_rad"], "delta_rotvec_rad")
    dg = scalar(payload["gripper_delta_m"], "gripper_delta_m")
    duration = scalar(payload["duration_s"], "duration_s")
    if not 1. <= duration <= 5.:
        raise ValueError("duration_s must be in [1, 5]")
    for actual, bound, name in ((np.linalg.norm(dp), limits.position_step_m, "translation"),
                                (np.linalg.norm(dr), limits.rotation_step_rad, "rotation"),
                                (abs(dg), limits.gripper_step_m, "gripper")):
        if actual > bound + 1e-12:
            raise ValueError(name + " step exceeds limit; not silently clipped")
    for actual, minimum, name in ((np.linalg.norm(dp), limits.minimum_position_step_m, "translation"),
                                  (np.linalg.norm(dr), limits.minimum_rotation_step_rad, "rotation"),
                                  (abs(dg), limits.minimum_gripper_step_m, "gripper")):
        if 0 < actual < minimum - 1e-12:
            raise ValueError(name + " step is below commissioning resolution")
    if not 0 <= width + dg <= .086:
        raise ValueError("gripper target outside [0, 0.086] m")
    rotation = rpy_matrix(pose[3:])
    delta_rotation = rotvec_matrix(dr)
    if payload["frame"] == "tool":
        target = np.r_[pose[:3] + rotation @ dp, matrix_rpy(rotation @ delta_rotation)]
    else:
        target = np.r_[pose[:3] + dp, matrix_rpy(delta_rotation @ rotation)]
    return target, width + dg, duration


class Executor:
    """Single owner. Call tick at 100 Hz even when no model is connected."""

    def __init__(self, backend, workspace_min, workspace_max, clock=time.monotonic, visual_steps=False):
        self.backend, self.clock = backend, clock
        self.visual_steps = visual_steps
        self.limits = replace(Limits(), position_step_m=.005) if visual_steps else Limits()
        self.lower, self.upper = np.array(workspace_min), np.array(workspace_max)
        if (self.lower.shape != (3,) or self.upper.shape != (3,) or
                not np.isfinite(np.r_[self.lower, self.upper]).all() or
                np.any(self.lower >= self.upper)):
            raise ValueError("valid workspace bounds are required")
        self.mode, self.reason = "disarmed", "not armed"
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.tokens = {}
        self.last_action = None
        self.active = None
        self.last_tick = clock()
        self.idle_since = self.last_tick
        self.session_start = self.last_tick
        self.origin = None
        self.reference = None
        self.previous = None
        self.motion_check = None

    def check(self, state):
        now = self.clock()
        for name, size in (("pose", 6), ("q", 6), ("dq", 6), ("torque", 6),
                           ("motor_received", 7)):
            value = np.asarray(state[name])
            if value.shape != (size,) or not np.isfinite(value).all():
                raise RuntimeError("invalid feedback: " + name)
        if not np.isfinite([state["width"], state["gripper_speed"], state["gripper_load"]]).all():
            raise RuntimeError("nonfinite gripper feedback")
        ages = now - np.array(state["motor_received"])
        if np.any(ages < 0) or np.max(ages) > self.limits.motor_age_s:
            raise RuntimeError("stale motor feedback")
        if state["emergency"] or state["motor_fault"]:
            raise RuntimeError("hardware or SDK fault")
        if np.any(np.abs(state["dq"]) > self.limits.joint_speed_rad_s):
            raise RuntimeError("measured joint speed exceeded")
        if abs(state["gripper_speed"]) > self.limits.gripper_speed_m_s * 1.5:
            if self.last_action is not None and self.last_action["status"] == "running":
                self.last_action["gripper_stop_sample"] = {
                    "width_m": state["width"], "speed_m_s": state["gripper_speed"],
                    "load": state["gripper_load"],
                    "speed_limit_m_s": self.limits.gripper_speed_m_s * 1.5}
            raise RuntimeError("measured gripper speed exceeded")
        if (np.any(np.abs(state["torque"]) > [10, 12, 10, 3, 3, 3]) or
                abs(state["gripper_load"]) > .25):
            raise RuntimeError("load exceeded; not classified as successful grasp")
        if not -.005 <= state["width"] <= .091:
            raise RuntimeError("gripper feedback outside calibrated range")
        self.backend.check_joints(state["q"])
        self.check_workspace(state["pose"])
        if self.previous is not None:
            prev_time, prev = self.previous
            dt = now - prev_time
            if dt >= .04:
                translation_speed = float(np.linalg.norm(state["pose"][:3]-prev[:3])/dt)
                angular_speed = rotation_error(state["pose"][3:], prev[3:])/dt
                self.motion_check = {"interval_s": dt,
                                     "previous_pose_base_xyz_rpy": prev.tolist(),
                                     "current_pose_base_xyz_rpy": state["pose"].tolist(),
                                     "translation_speed_m_s": translation_speed,
                                     "rotation_speed_rad_s": angular_speed,
                                     "joint_velocity_rad_s": state["dq"].tolist(),
                                     "motor_feedback_age_s": ages.tolist()}
                if translation_speed > self.limits.translation_speed_m_s * 1.5:
                    raise RuntimeError("measured translation speed exceeded")
                if angular_speed > self.limits.rotation_speed_rad_s * 1.5:
                    raise RuntimeError("measured rotation speed exceeded")
                self.previous = (now, state["pose"].copy())
        else:
            self.previous = (now, state["pose"].copy())

    def check_workspace(self, pose):
        if np.any(pose[:3] < self.lower) or np.any(pose[:3] > self.upper):
            raise RuntimeError("workspace exceeded")
        if self.origin is not None and np.linalg.norm(pose[:3]-self.origin[:3]) > self.limits.radius_m:
            raise RuntimeError("session translation envelope exceeded")

    def stop(self, reason="explicit stop"):
        self.mode, self.reason = "stopped", reason
        self.tokens.clear()
        if self.last_action is not None and self.last_action["status"] == "running":
            self.last_action.update(status="failed", reason=reason,
                                    last_motion_check=self.motion_check)
        self.active = None
        self.backend.stop()

    def snapshot(self):
        state = self.backend.read()
        healthy, error = True, None
        try:
            self.check(state)
        except Exception as exc:
            healthy, error = False, str(exc)
            if self.mode not in ("disarmed", "stopped"):
                self.stop(error)
        token = uuid.uuid4().hex if healthy and self.mode == "ready" else None
        if token:
            self.tokens[token] = (self.clock(), state)
            while len(self.tokens) > 32:
                del self.tokens[next(iter(self.tokens))]
        return {"protocol": PROTOCOL, "source": self.backend.source,
                "session": self.session, "mode": self.mode, "reason": self.reason,
                "healthy": healthy, "feedback_error": error,
                "next_sequence": self.sequence + 1, "state_token": token,
                "ee_pose_base_xyz_rpy": state["pose"].tolist(),
                "joint_position_rad": state["q"].tolist(),
                "joint_velocity_rad_s": state["dq"].tolist(),
                "joint_torque_nm": state["torque"].tolist(),
                "gripper_width_m": state["width"],
                "gripper_velocity_m_s": state["gripper_speed"],
                "gripper_torque": state["gripper_load"],
                "source_timestamp_s": float(np.min(state["motor_received"])),
                "motor_feedback_age_s": (self.clock()-state["motor_received"]).tolist(),
                "last_action": self.last_action}

    def arm(self):
        if self.mode != "disarmed":
            raise ValueError("arm is allowed only once after process startup")
        state = self.backend.read()
        self.check(state)
        if not 0 <= state["width"] <= .086:
            raise ValueError("measured gripper width must be in [0, .086] before arming")
        self.origin = state["pose"].copy()
        self.reference = state
        self.mode, self.reason = "arming", "ramping gains at measured pose"
        self.session_start = self.clock()
        return {"session": self.session, "mode": self.mode}

    def submit(self, payload):
        keys = {"session", "sequence", "state_token", "frame", "delta_xyz_m",
                "delta_rotvec_rad", "gripper_delta_m", "duration_s"}
        if not isinstance(payload, dict) or set(payload) != keys:
            raise ValueError("action fields must match protocol exactly")
        if payload["session"] != self.session:
            raise ValueError("wrong session")
        if type(payload["sequence"]) is not int:
            raise ValueError("sequence must be an integer")
        if self.last_action is not None and payload == self.last_action["request"]:
            return self.last_action
        if self.mode != "ready":
            raise ValueError("executor is not ready: " + self.mode)
        if payload["sequence"] != self.sequence + 1:
            raise ValueError("out-of-order/replayed sequence")
        token = payload["state_token"]
        if not isinstance(token, str) or token not in self.tokens:
            raise ValueError("unknown or consumed state_token")
        created, state = self.tokens[token]
        if self.clock()-created > self.limits.snapshot_age_s:
            raise ValueError("expired state_token; get fresh state")
        current = self.backend.read()
        self.check(current)
        if (np.linalg.norm(current["pose"][:3]-state["pose"][:3]) > .0005 or
                rotation_error(current["pose"][3:], state["pose"][3:]) > .005 or
                abs(current["width"]-state["width"]) > .001):
            raise ValueError("state changed since observation")
        target, width, duration = action_target(payload, current["pose"], current["width"], self.limits)
        visual = self.visual_steps and any(payload["delta_xyz_m"])
        if visual and (any(payload["delta_rotvec_rad"]) or payload["gripper_delta_m"] != 0):
            raise ValueError("visual translation cannot be combined with rotation or gripper motion")
        self.check_workspace(target)
        arm_delta = any(payload["delta_xyz_m"]) or any(payload["delta_rotvec_rad"])
        q_target = self.backend.solve(target, current["q"]) if arm_delta else current["q"].copy()
        self.backend.check_joints(q_target)
        q_start = current["q"].copy()
        # Validate the actual joint-interpolated route, not only its endpoint.
        previous_pose = self.backend.fk(q_start)
        for alpha in np.linspace(0, 1, 101)[1:]:
            pose = self.backend.fk(q_start + alpha*(q_target-q_start))
            self.check_workspace(pose)
            if visual:
                self.translation_envelope(current["pose"], target, pose, planned=True)
            if (np.linalg.norm(pose[:3]-previous_pose[:3])*150/duration > self.limits.translation_speed_m_s or
                    rotation_error(pose[3:], previous_pose[3:])*150/duration > self.limits.rotation_speed_rad_s):
                raise ValueError("planned Cartesian speed exceeded")
            previous_pose = pose
        if (np.max(np.abs(q_target-q_start))*1.5/duration > self.limits.joint_speed_rad_s or
                abs(width-current["width"])*1.5/duration > self.limits.gripper_speed_m_s):
            raise ValueError("planned joint/gripper speed exceeded")
        if not visual and (np.linalg.norm(previous_pose[:3]-target[:3]) > .00025 or
                rotation_error(previous_pose[3:], target[3:]) > .0025):
            raise ValueError("IK returned inaccurate FK result")
        now = self.clock()
        if now-self.last_tick > self.limits.loop_gap_s or now-created > self.limits.snapshot_age_s:
            raise RuntimeError("planning exceeded control/snapshot deadline")
        self.sequence = payload["sequence"]
        self.tokens.clear()
        self.active = dict(start=now, duration=duration, q_start=q_start,
                           q_target=q_target, width_start=current["width"], width_target=width,
                           target=target, settled_since=None, settling_samples=deque(),
                           visual=visual, start_pose=current["pose"].copy())
        requested = [np.linalg.norm(target[:3]-current["pose"][:3]),
                     rotation_error(target[3:], current["pose"][3:]), abs(width-current["width"])]
        defaults = [self.limits.position_tolerance_m, self.limits.rotation_tolerance_rad,
                    self.limits.gripper_tolerance_m]
        # A requested microstep cannot succeed merely by staying within a larger
        # fixed tolerance. Require at least 75% of each requested displacement.
        self.active["tolerances"] = [min(default, .25*distance) if distance > 1e-9 else default
                                     for distance, default in zip(requested, defaults)]
        self.last_action = {"sequence": self.sequence, "status": "running", "request": payload,
                            "completion_mode": "visual_step" if visual else "target",
                            "start_pose_base_xyz_rpy": current["pose"].tolist(),
                            "target_pose_base_xyz_rpy": target.tolist(), "target_width_m": width,
                            "planned_pose_base_xyz_rpy": previous_pose.tolist()}
        self.mode, self.reason = "moving", "bounded action"
        return self.last_action

    def tick(self):
        now = self.clock()
        gap, self.last_tick = now-self.last_tick, now
        if self.mode in ("disarmed", "stopped"):
            # The SDK still receives motor feedback while no action is armed.
            # Drain the independent monitor to avoid stale queued CAN frames.
            self.backend.read()
            return
        try:
            if gap > self.limits.loop_gap_s:
                raise RuntimeError("control loop deadline missed")
            if now-self.session_start > self.limits.session_timeout_s:
                raise RuntimeError("session expired")
            state = self.backend.read()
            self.check(state)
            if self.mode == "arming":
                self.track(state, self.reference["pose"], self.reference["q"], self.reference["width"])
                fraction = min(1., now-self.session_start)
                self.backend.command(self.reference["q"], self.reference["width"])
                self.backend.gains(fraction)
                if fraction == 1.:
                    self.mode, self.reason, self.idle_since = "ready", "holding measured startup pose", now
                return
            if self.mode == "ready":
                self.track(state, self.reference["pose"], self.reference["q"], self.reference["width"])
                if now-self.idle_since > self.limits.idle_timeout_s:
                    raise RuntimeError("action idle timeout; state reads do not renew it")
                return
            a = self.active
            elapsed = now-a["start"]
            t = min(1., elapsed/a["duration"])
            alpha = t*t*(3-2*t)
            q = a["q_start"] + alpha*(a["q_target"]-a["q_start"])
            width = a["width_start"] + alpha*(a["width_target"]-a["width_start"])
            pose = self.backend.fk(q)
            progress = None
            if a["visual"]:
                progress = self.translation_envelope(a["start_pose"], a["target"], state["pose"])
            self.track(state, pose, q, width,
                       position_limit=self.limits.position_step_m+.0005 if a["visual"] else None)
            self.backend.command(q, width)
            errors = {"position_m": float(np.linalg.norm(state["pose"][:3]-a["target"][:3])),
                      "rotation_rad": rotation_error(state["pose"][3:], a["target"][3:]),
                      "gripper_m": abs(state["width"]-a["width_target"])}
            self.last_action.update(errors=errors, actual_pose_base_xyz_rpy=state["pose"].tolist(),
                                    actual_width_m=state["width"])
            if "q_command" in state:
                self.last_action["joint_command_rad"] = state["q_command"].tolist()
                self.last_action["joint_actual_rad"] = state["q"].tolist()
            if progress is not None:
                self.last_action["measured_translation"] = progress
            stationary = self.motion_settled(a, state, now)
            reached = (errors["position_m"] <= a["tolerances"][0] and
                       errors["rotation_rad"] <= a["tolerances"][1] and
                       errors["gripper_m"] <= a["tolerances"][2] and
                       stationary)
            visual_complete = (a["visual"] and progress["along_m"] >= .0005 and stationary
                               and errors["gripper_m"] <= self.limits.gripper_tolerance_m)
            if elapsed >= a["duration"] and (reached if not a["visual"] else visual_complete):
                if a["settled_since"] is None:
                    a["settled_since"] = now
                if now-a["settled_since"] >= .15:
                    self.last_action["status"] = "executed" if a["visual"] else "succeeded"
                    self.last_action["target_reached"] = bool(reached)
                    if a["visual"]:
                        # Drop the remaining position demand; the next action
                        # must be based on a fresh observation, not catch-up.
                        self.backend.command(state["q"], state["width"])
                        self.reference = dict(pose=state["pose"].copy(), q=state["q"].copy(), width=state["width"])
                    else:
                        self.reference = dict(pose=a["target"], q=a["q_target"], width=a["width_target"])
                    self.mode, self.reason, self.idle_since = "ready", "visual step executed" if a["visual"] else "measured target reached", now
                    self.active = None
            else:
                a["settled_since"] = None
            if self.active is not None and elapsed > a["duration"]+self.limits.settle_timeout_s:
                raise RuntimeError("no stable forward progress before deadline" if a["visual"] else "target not reached before deadline")
        except Exception as exc:
            self.stop(str(exc))

    def motion_settled(self, action, state, now):
        # Raw motor velocity bins can cross the completion threshold at rest.
        # Use measured excursion for completion; check() still enforces speeds.
        samples = action["settling_samples"]
        if now-action["start"] < action["duration"]:
            return False
        samples.append((now, state["pose"].copy(), state["q"].copy(), state["width"]))
        cutoff = now-self.limits.settle_window_s
        while len(samples) > 1 and samples[1][0] <= cutoff:
            samples.popleft()
        if now-samples[0][0] < self.limits.settle_window_s:
            return False
        poses = np.array([s[1] for s in samples])
        return (np.linalg.norm(np.ptp(poses[:, :3], axis=0)) <= self.limits.settle_position_m and
                np.max(np.ptp(np.array([s[2] for s in samples]), axis=0)) <= self.limits.settle_joint_rad and
                max(rotation_error(poses[0, 3:], p[3:]) for p in poses) <= self.limits.settle_rotation_rad and
                np.ptp([s[3] for s in samples]) <= self.limits.settle_gripper_m)

    def track(self, state, pose, q, width, position_limit=None):
        if (np.linalg.norm(state["pose"][:3]-pose[:3]) > (self.limits.position_tracking_m if position_limit is None else position_limit) or
                rotation_error(state["pose"][3:], pose[3:]) > self.limits.rotation_tracking_rad or
                np.max(np.abs(state["q"]-q)) > self.limits.joint_tracking_rad or
                abs(state["width"]-width) > self.limits.gripper_tracking_m):
            raise RuntimeError("tracking error exceeded")

    def translation_envelope(self, start, target, actual, planned=False):
        wanted = target[:3]-start[:3]
        distance = float(np.linalg.norm(wanted))
        direction = wanted/distance
        displacement = actual[:3]-start[:3]
        along = float(displacement @ direction)
        cross = float(np.linalg.norm(displacement-along*direction))
        margin = 0. if planned else .0005
        if (np.linalg.norm(displacement) > distance+margin+1e-12 or along < -.0005 or
                cross > .001 or rotation_error(start[3:], actual[3:]) > .02):
            raise RuntimeError("visual step direction/amplitude envelope exceeded")
        return {"along_m": along, "cross_m": cross, "delta_base_m": displacement.tolist()}

    def info(self):
        return {"protocol": PROTOCOL, "source": self.backend.source, "model": "X5_umi", "interface": "can0",
                "execution_mode": "visual_steps" if self.visual_steps else "strict_targets",
                "visual_translation": {"enabled": self.visual_steps,
                                       "minimum_progress_m": .0005,
                                       "measured_amplitude_margin_m": .0005,
                                       "reverse_tolerance_m": .0005,
                                       "cross_track_limit_m": .001,
                                       "orientation_change_limit_rad": .02,
                                       "completion_status": "executed",
                                       "requires_reobservation": True},
                "limits": asdict(self.limits), "workspace_min_m": self.lower.tolist(),
                "workspace_max_m": self.upper.tolist(),
                "tool_frame": "SDK eef_link; not observation TCP or camera frame",
                "rotation": "right-hand rotation vector; base: dR*R, tool: R*dR",
                "positive_gripper_delta": "opens", "stop": "latched damping, NOT power removal"}
