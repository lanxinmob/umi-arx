"""Single-owner, observation-conditioned facade over the bounded action bridge."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from action_mcp.control import Limits, action_target
from action_mcp.server import Client
from observation_mcp.live import (
    LiveObservationProvider, LiveObservationSample, _normalize_robot_state,
)
from observation_mcp.server import MockObservationProvider, _observation_content


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "observation_mcp/config.json"
PROTOCOL = "umi-arx.robot.v1"


class MockHistory(LiveObservationProvider):
    """Real-time mock bridge telemetry; images are explicitly synthetic fixtures."""

    source = "mock"

    def _run(self):
        port = int(self.config.robot.endpoint.rsplit(":", 1)[1])
        client = Client(port)
        pictures = MockObservationProvider()
        try:
            while not self._stop.is_set():
                try:
                    state = client.request("state")
                    if state.get("source") != "mock" or not state.get("healthy"):
                        raise RuntimeError("expected healthy mock telemetry")
                    fixture = pictures.snapshot(1, 0, 0)[0]
                    now_ns, now = time.time_ns(), time.monotonic()
                    fields = {
                        "timestamp": "source_timestamp_s", "ee_pose": "ee_pose_base_xyz_rpy",
                        "joint_pos": "joint_position_rad", "joint_vel": "joint_velocity_rad_s",
                        "joint_torque": "joint_torque_nm", "gripper_pos": "gripper_width_m",
                        "gripper_vel": "gripper_velocity_m_s", "gripper_torque": "gripper_torque",
                    }
                    robot = _normalize_robot_state(
                        {key: state[value] for key, value in fields.items()},
                        received_at_unix_ns=now_ns, skew_ms=0,
                    )
                    sample = LiveObservationSample(
                        self._sequence, now_ns, fixture.image_jpeg,
                        fixture.image_width, fixture.image_height, robot, now,
                    )
                    with self._condition:
                        self._samples.append(sample)
                        self._sequence += 1
                        self._last_camera_frame_unix_ns = now_ns
                        self._last_error = None
                        self._condition.notify_all()
                except Exception as exc:
                    self._set_error(str(exc))
                self._stop.wait(1 / self.config.sampling.frequency_hz)
        finally:
            client.close()


class RobotAPI:
    """Use from one model session; concurrent callers are serialized."""

    def __init__(self, port=8766, *, source="mock", config=CONFIG, output=None):
        if source not in ("mock", "live"):
            raise ValueError("source must be mock or live")
        self.source = source
        self.client = Client(port)
        self.provider = None
        self._lock = threading.RLock()
        self._closed = False
        self._observation = None
        self._requests = {}
        self._session = None
        self.output = Path(output or ROOT / "robot_api/runs") / uuid.uuid4().hex
        try:
            self.contract = self.client.request("info")
            if (self.contract.get("protocol") != "umi-arx.action.v1"
                    or self.contract.get("source") != source
                    or self.contract.get("model") != "X5_umi"
                    or self.contract.get("interface") != "can0"):
                raise RuntimeError("bridge identity mismatch; expected X5_umi/can0/" + source)
            self._session = self._state()["session"]
            provider = MockHistory if source == "mock" else LiveObservationProvider
            self.provider = provider(str(config), action_port=port)
            self.output.mkdir(parents=True, exist_ok=False)
            self._save("contract.json", self.contract)
        except Exception:
            self.close()
            raise

    def _save(self, name, value):
        path = self.output / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        return str(path)

    def _state(self):
        state = self.client.request("state")
        if state.get("protocol") != "umi-arx.action.v1" or state.get("source") != self.source or (
                self._session is not None and state.get("session") != self._session):
            raise RuntimeError("bridge source/session changed; reconnect explicitly")
        return state

    def _blockers(self):
        if self.source == "mock":
            return []
        # Evidence survives process restart; no model-accessible approval tool.
        evidence = json.loads((Path(__file__).with_name("commissioning.json")).read_text())
        checks = ("gripper_bounded_motion", "arm_stable_hold", "task_workspace")
        if evidence.get("robot") != "X5_umi/can0":
            return ["commissioning identity mismatch"]
        return [name + ": " + evidence.get(name, {}).get("detail", "missing evidence")
                for name in checks if evidence.get(name, {}).get("status") != "passed"
                or not evidence.get(name, {}).get("evidence")]

    def _result(self, data, images=(), failed=False):
        packet = {"protocol": PROTOCOL, "source": self.source, **data}
        return {"content": [{"type": "text", "text": json.dumps(packet, allow_nan=False)}, *images],
                "structuredContent": packet, "isError": failed}

    def _fresh(self, observation_status):
        age = observation_status.get("latest_observation_age_ms")
        maximum = self.provider.config.sampling.max_observation_age_seconds * 1000
        return (observation_status.get("ready") is True
                and age is not None and 0 <= age <= maximum)

    def status(self):
        with self._lock:
            try:
                state = self._state()
                obs = self.provider.status()
                fresh = self._fresh(obs)
                blockers = self._blockers()
                if not state.get("healthy"):
                    blockers.append(state.get("feedback_error") or "unhealthy feedback")
                if not fresh:
                    blockers.append("fresh complete observation unavailable")
                if state.get("mode") != "ready":
                    blockers.append("controller mode: " + state.get("mode", "unknown"))
                return self._result({"connected": True, "state": state,
                                     "observation_status": obs, "contract": self.contract,
                                     "simulation_ready": self.source == "mock" and not blockers,
                                     "task_ready": self.source == "live" and not blockers,
                                     "blockers": blockers, "output_directory": str(self.output)})
            except Exception as exc:
                return self._result({"connected": False, "task_ready": False,
                                     "simulation_ready": False, "blockers": [str(exc)]}, failed=True)

    def observe(self, sample_count=4, history_seconds=2.4, wait_timeout_seconds=5.0):
        with self._lock:
            return self._observe(sample_count, history_seconds, wait_timeout_seconds)

    def _observe(self, sample_count=4, history_seconds=2.4, wait_timeout_seconds=5.0,
                 after_ns=0):
        self._state()
        deadline = time.monotonic() + wait_timeout_seconds
        while True:
            content = _observation_content(self.provider, sample_count, history_seconds,
                                           wait_timeout_seconds)
            packet = json.loads(content[0]["text"])
            newest = packet["samples"][-1]["captured_at_unix_ns"]
            if newest > after_ns:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("no new observation after action completion")
            time.sleep(.05)
        state = self._state()
        observation_id = uuid.uuid4().hex
        age_s = max(0, (time.time_ns() - newest) / 1e9)
        self._observation = (observation_id, time.monotonic() - age_s, state["next_sequence"])
        files = []
        for index, block in enumerate(content[1:], 1):
            path = self.output / (observation_id + "_" + str(index) + ".jpg")
            path.write_bytes(base64.b64decode(block["data"]))
            files.append(str(path))
        result = self._result({"observation_id": observation_id, "observation": packet,
                               "state": state, "image_files": files,
                               "observation_max_age_s": 10}, content[1:])
        self._save(observation_id + ".json", result["structuredContent"])
        return result

    def arm(self):
        with self._lock:
            blockers = self._blockers()
            if blockers:
                raise RuntimeError("live commissioning incomplete: " + "; ".join(blockers))
            state = self._state()
            if state["mode"] != "disarmed" or not state["healthy"]:
                raise RuntimeError("arm requires a healthy disarmed bridge; never auto-rearms")
            self._observe(1, 0, 3)
            self._observation = None
            try:
                state = self.client.call("arm", {})
                if state.get("mode") != "ready":
                    raise RuntimeError("arming failed: " + str(state))
                result = self._observe(after_ns=time.time_ns())
                self._save("arm.json", result["structuredContent"])
                return result
            except Exception:
                self._stop_best_effort()
                raise

    def move(self, request_id, observation_id, delta_xyz_m, delta_rotvec_rad=None,
             frame="base", duration_s=5.0):
        return self._action(request_id, observation_id, {
            "delta_xyz_m": delta_xyz_m,
            "delta_rotvec_rad": [0, 0, 0] if delta_rotvec_rad is None else delta_rotvec_rad,
            "gripper_delta_m": 0, "frame": frame, "duration_s": duration_s,
        })

    def gripper(self, request_id, observation_id, delta_m, duration_s=5.0):
        return self._action(request_id, observation_id, {
            "delta_xyz_m": [0, 0, 0], "delta_rotvec_rad": [0, 0, 0],
            "gripper_delta_m": delta_m, "frame": "base", "duration_s": duration_s,
        })

    def _action(self, request_id, observation_id, command):
        with self._lock:
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
                raise ValueError("request_id must be a nonempty string, at most 128 characters")
            fingerprint = json.dumps([observation_id, command], sort_keys=True, allow_nan=False)
            if request_id in self._requests:
                previous, path = self._requests[request_id]
                if previous != fingerprint:
                    raise ValueError("request_id reused with different arguments")
                return json.loads(Path(path).read_text())
            if len(self._requests) >= 1000:
                raise RuntimeError("session request capacity reached; stop and reconnect")
            blockers = self._blockers()
            if blockers:
                raise RuntimeError("live commissioning incomplete: " + "; ".join(blockers))
            obs = self._observation
            if not obs or observation_id != obs[0] or time.monotonic() - obs[1] > 10:
                raise ValueError("missing, superseded, consumed or expired observation; call observe")
            if not self._fresh(self.provider.status()):
                self._stop_best_effort()
                raise RuntimeError("observation stream is stale/unhealthy; stop requested")
            state = self._state()
            if (state["mode"] != "ready" or not state["healthy"]
                    or state["next_sequence"] != obs[2] or not self.client.owns_session):
                raise RuntimeError("session is not owned/ready or changed since observation")
            payload = {**command, "session": state["session"],
                       "sequence": state["next_sequence"], "state_token": state["state_token"]}
            action_target(payload, np.asarray(state["ee_pose_base_xyz_rpy"]),
                          state["gripper_width_m"], Limits(**self.contract["limits"]))
            self._observation = None
            name = "request_" + uuid.uuid4().hex + ".json"
            pending = self._result({"request_id": request_id, "outcome": "unknown",
                                    "action_request": payload}, failed=True)
            path = self._save(name, pending)
            self._requests[request_id] = (fingerprint, path)
            try:
                action = self.client.call("action", payload)
                failed = action.get("status") not in ("succeeded", "executed")
                try:
                    observation = self._observe(after_ns=time.time_ns())
                    data = {**observation["structuredContent"], "request_id": request_id,
                            "action": action}
                    result = self._result(data, observation["content"][1:], failed=failed)
                except Exception as exc:
                    result = self._result({"request_id": request_id, "action": action,
                                           "observation_error": str(exc),
                                           "stop": self._stop_best_effort()}, failed=True)
            except Exception as exc:
                rejected = False
                if isinstance(exc, ValueError):
                    try:
                        current = self._state()
                        rejected = (current["mode"] == "ready" and current["healthy"]
                                    and current["next_sequence"] == payload["sequence"])
                    except Exception:
                        pass
                result = self._result({"request_id": request_id,
                                       "outcome": "rejected" if rejected else "unknown",
                                       "error": str(exc),
                                       "stop": None if rejected else self._stop_best_effort()}, failed=True)
            try:
                self._save(name, result)
            except Exception:
                self._stop_best_effort()
                raise
            return result

    def _stop_best_effort(self):
        self._observation = None
        try:
            return self.client.request("stop")
        except Exception as exc:
            return {"confirmed": False, "error": str(exc)}

    def stop(self):
        with self._lock:
            result = self._stop_best_effort()
            return self._result({"stop": result}, failed=result.get("confirmed") is False)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.client.close()
            if self.provider is not None:
                self.provider.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
