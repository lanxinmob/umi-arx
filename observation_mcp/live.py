"""Live UVC and ARX5 GET_STATE observation provider."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import json
import math
import os
import threading
import time
from typing import Any

import cv2
import numpy as np
import zmq


@dataclass(frozen=True)
class LiveObservationSample:
    sequence: int
    captured_at_unix_ns: int
    image_jpeg: bytes
    image_width: int
    image_height: int
    robot: dict[str, Any]
    captured_at_monotonic_s: float


@dataclass(frozen=True)
class CameraConfig:
    device: str
    width: int
    height: int
    fps: float
    fourcc: str
    buffer_size: int
    output_width: int
    output_height: int
    jpeg_quality: int


@dataclass(frozen=True)
class RobotConfig:
    endpoint: str
    timeout_ms: int
    action_protocol: bool = False


@dataclass(frozen=True)
class SamplingConfig:
    frequency_hz: float
    buffer_seconds: float
    max_observation_age_seconds: float


@dataclass(frozen=True)
class LiveConfig:
    camera: CameraConfig
    robot: RobotConfig
    sampling: SamplingConfig

    @classmethod
    def from_path(cls, path: str) -> "LiveConfig":
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("config root must be an object")
        camera = _object(raw, "camera")
        robot = _object(raw, "robot")
        sampling = _object(raw, "sampling")
        result = cls(
            camera=CameraConfig(
                device=_text(camera, "device"),
                width=_integer(camera, "width", 1),
                height=_integer(camera, "height", 1),
                fps=_number(camera, "fps", 0.1),
                fourcc=_text(camera, "fourcc"),
                buffer_size=_integer(camera, "buffer_size", 1),
                output_width=_integer(camera, "output_width", 1),
                output_height=_integer(camera, "output_height", 1),
                jpeg_quality=_integer(camera, "jpeg_quality", 1, 100),
            ),
            robot=RobotConfig(
                endpoint=_text(robot, "endpoint"),
                timeout_ms=_integer(robot, "timeout_ms", 1, 10_000),
            ),
            sampling=SamplingConfig(
                frequency_hz=_number(sampling, "frequency_hz", 0.1, 60.0),
                buffer_seconds=_number(sampling, "buffer_seconds", 1.0, 120.0),
                max_observation_age_seconds=_number(
                    sampling, "max_observation_age_seconds", 0.1, 30.0
                ),
            ),
        )
        if len(result.camera.fourcc) != 4:
            raise ValueError("camera.fourcc must contain exactly four characters")
        if not result.robot.endpoint.startswith(
            ("tcp://127.0.0.1:", "tcp://localhost:")
        ):
            raise ValueError("robot.endpoint must use trusted loopback")
        return result


def _object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config.{key} must be an object")
    return value


def _text(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config.{key} must be a non-empty string")
    return value


def _number(
    parent: dict[str, Any], key: str, minimum: float, maximum: float | None = None
) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config.{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"config.{key} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"config.{key} must be at most {maximum}")
    return number


def _integer(
    parent: dict[str, Any], key: str, minimum: int, maximum: int | None = None
) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"config.{key} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"config.{key} is outside the supported range")
    return value


class RobotStateClient:
    """Restricted client whose only outbound command is GET_STATE."""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self.context = zmq.Context()
        self.socket: zmq.Socket[Any] | None = None
        self._connect()

    def _connect(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self.config.timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self.config.timeout_ms)
        socket.connect(self.config.endpoint)
        self.socket = socket

    def get_state(
        self, captured_at_monotonic_s: float
    ) -> tuple[dict[str, Any], int]:
        assert self.socket is not None
        try:
            if self.config.action_protocol:
                self.socket.send_json(
                    {"protocol": "umi-arx.action.v1", "command": "state", "data": {}}
                )
                reply = _action_state_envelope(self.socket.recv_json())
            else:
                self.socket.send_pyobj({"cmd": "GET_STATE", "data": None})
                reply = self.socket.recv_pyobj()
        except zmq.ZMQError as exc:
            self._connect()
            raise TimeoutError(
                f"GET_STATE failed at {self.config.endpoint}: {exc}"
            ) from exc
        received_at_monotonic_s = time.monotonic()
        received_at_unix_ns = time.time_ns()
        if not isinstance(reply, dict) or reply.get("cmd") != "GET_STATE":
            raise RuntimeError("robot server returned an invalid GET_STATE envelope")
        raw = reply.get("data")
        if not isinstance(raw, dict):
            raise RuntimeError("robot server returned invalid GET_STATE data")
        if raw.get("emergency", False):
            raise RuntimeError("robot emergency is latched")
        state = _normalize_robot_state(
            raw,
            received_at_unix_ns=received_at_unix_ns,
            skew_ms=abs(received_at_monotonic_s - captured_at_monotonic_s) * 1000,
        )
        return state, received_at_unix_ns

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()


def _action_state_envelope(reply: Any) -> dict[str, Any]:
    if not isinstance(reply, dict) or reply.get("ok") is not True:
        raise RuntimeError("action bridge rejected the state request")
    state = reply.get("result")
    if (not isinstance(state, dict) or state.get("protocol") != "umi-arx.action.v1"
            or state.get("source") != "live" or state.get("healthy") is not True):
        raise RuntimeError("action bridge must return healthy live feedback, not mock/stale data")
    fields = {
        "timestamp": "source_timestamp_s", "ee_pose": "ee_pose_base_xyz_rpy",
        "joint_pos": "joint_position_rad", "joint_vel": "joint_velocity_rad_s",
        "joint_torque": "joint_torque_nm", "gripper_pos": "gripper_width_m",
        "gripper_vel": "gripper_velocity_m_s", "gripper_torque": "gripper_torque",
    }
    try:
        return {"cmd": "GET_STATE", "data": {old: state[new] for old, new in fields.items()}}
    except KeyError as exc:
        raise RuntimeError("action bridge state is incomplete") from exc


def _vector(raw: dict[str, Any], key: str, size: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(raw[key], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"robot state field {key} is invalid") from exc
    if result.ndim != 1 or not result.size or not np.all(np.isfinite(result)):
        raise RuntimeError(f"robot state field {key} must be a finite vector")
    if size is not None and result.shape != (size,):
        raise RuntimeError(f"robot state field {key} must have shape ({size},)")
    return result


def _scalar(raw: dict[str, Any], key: str) -> float:
    try:
        result = float(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"robot state field {key} is invalid") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"robot state field {key} must be finite")
    return result


def _ee_to_tcp(ee_pose: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = ee_pose[3:]
    rotation_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    rotation_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    rotation_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    ee_to_tcp = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    ee_rotation = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
    tcp_rotation = ee_rotation @ ee_to_tcp
    theta = np.arccos((np.trace(tcp_rotation) - 1) / 2)
    if np.isclose(theta, 0):
        rotation_vector = np.zeros(3)
    else:
        axis = np.array(
            [
                tcp_rotation[2, 1] - tcp_rotation[1, 2],
                tcp_rotation[0, 2] - tcp_rotation[2, 0],
                tcp_rotation[1, 0] - tcp_rotation[0, 1],
            ]
        )
        axis = axis / (2 * np.sin(theta))
        rotation_vector = theta * axis
    angle = np.linalg.norm(rotation_vector)
    axis = rotation_vector / angle
    rotation_vector = -axis * (2 * np.pi - angle)
    return np.concatenate([ee_pose[:3], rotation_vector])


def _normalize_robot_state(
    raw: dict[str, Any], *, received_at_unix_ns: int, skew_ms: float
) -> dict[str, Any]:
    ee_pose = _vector(raw, "ee_pose", 6)
    joint_position = _vector(raw, "joint_pos")
    joint_velocity = _vector(raw, "joint_vel", joint_position.size)
    joint_torque = _vector(raw, "joint_torque", joint_position.size)
    return {
        "host_received_at_unix_ns": received_at_unix_ns,
        "source_timestamp_s": _scalar(raw, "timestamp"),
        "host_receive_skew_ms": round(skew_ms, 3),
        "ee_pose_base_xyz_rpy": ee_pose.tolist(),
        "tcp_pose_base_xyz_rotvec": _ee_to_tcp(ee_pose).tolist(),
        "joint_position_rad": joint_position.tolist(),
        "joint_velocity_rad_s": joint_velocity.tolist(),
        "joint_torque_nm": joint_torque.tolist(),
        "gripper_width_m": _scalar(raw, "gripper_pos"),
        "gripper_velocity_m_s": _scalar(raw, "gripper_vel"),
        "gripper_torque": _scalar(raw, "gripper_torque"),
    }


def _resize_center_crop(
    frame: np.ndarray, output_width: int, output_height: int
) -> np.ndarray:
    height, width = frame.shape[:2]
    source_ratio = width / height
    target_ratio = output_width / output_height
    if source_ratio > target_ratio:
        crop_width = max(1, int(round(height * target_ratio)))
        start = (width - crop_width) // 2
        frame = frame[:, start : start + crop_width]
    elif source_ratio < target_ratio:
        crop_height = max(1, int(round(width / target_ratio)))
        start = (height - crop_height) // 2
        frame = frame[start : start + crop_height, :]
    return cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)


class LiveObservationProvider:
    source = "live"

    def __init__(self, config_path: str, action_port: int | None = None) -> None:
        self.config_path = os.path.abspath(config_path)
        self.config = LiveConfig.from_path(self.config_path)
        if action_port is not None:
            if type(action_port) is not int or not 1 <= action_port <= 65535:
                raise ValueError("action_port must be an integer in [1, 65535]")
            self.config = replace(
                self.config,
                robot=RobotConfig(
                    endpoint=f"tcp://127.0.0.1:{action_port}",
                    timeout_ms=self.config.robot.timeout_ms,
                    action_protocol=True,
                ),
            )
        capacity = max(
            2,
            math.ceil(
                self.config.sampling.frequency_hz
                * self.config.sampling.buffer_seconds
            ),
        )
        self._samples: deque[LiveObservationSample] = deque(maxlen=capacity)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._sequence = 0
        self._last_error: str | None = None
        self._last_camera_frame_unix_ns: int | None = None
        self._thread = threading.Thread(
            target=self._run, name="arx5-observation-capture", daemon=True
        )
        self._thread.start()

    def _set_error(self, error: str) -> None:
        with self._condition:
            self._last_error = error
            self._condition.notify_all()

    def _open_camera(self) -> cv2.VideoCapture:
        camera = self.config.camera
        capture = cv2.VideoCapture(camera.device)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
        capture.set(cv2.CAP_PROP_FPS, camera.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, camera.buffer_size)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open camera {camera.device}")
        return capture

    def _run(self) -> None:
        while not self._stop.is_set():
            capture: cv2.VideoCapture | None = None
            robot: RobotStateClient | None = None
            try:
                capture = self._open_camera()
                robot = RobotStateClient(self.config.robot)
                self._capture_loop(capture, robot)
            except Exception as exc:
                self._set_error(str(exc))
                self._stop.wait(1.0)
            finally:
                if capture is not None:
                    capture.release()
                if robot is not None:
                    robot.close()

    def _capture_loop(
        self, capture: cv2.VideoCapture, robot: RobotStateClient
    ) -> None:
        interval = 1.0 / self.config.sampling.frequency_hz
        next_sample_at = time.monotonic()
        camera = self.config.camera
        while not self._stop.is_set():
            ok, frame = capture.read()
            captured_at_monotonic_s = time.monotonic()
            captured_at_unix_ns = time.time_ns()
            if not ok or frame is None:
                raise RuntimeError(f"camera read failed for {camera.device}")
            self._last_camera_frame_unix_ns = captured_at_unix_ns
            if captured_at_monotonic_s < next_sample_at:
                continue
            next_sample_at = max(
                next_sample_at + interval, captured_at_monotonic_s + interval * 0.25
            )
            resized = _resize_center_crop(
                frame, camera.output_width, camera.output_height
            )
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                resized,
                [cv2.IMWRITE_JPEG_QUALITY, camera.jpeg_quality],
            )
            if not encoded_ok:
                raise RuntimeError("failed to encode camera frame as JPEG")
            try:
                robot_state, _ = robot.get_state(captured_at_monotonic_s)
            except (RuntimeError, TimeoutError) as exc:
                self._set_error(str(exc))
                continue
            sample = LiveObservationSample(
                sequence=self._sequence,
                captured_at_unix_ns=captured_at_unix_ns,
                captured_at_monotonic_s=captured_at_monotonic_s,
                image_jpeg=encoded.tobytes(),
                image_width=camera.output_width,
                image_height=camera.output_height,
                robot=robot_state,
            )
            self._sequence += 1
            with self._condition:
                self._samples.append(sample)
                self._last_error = None
                self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            count = len(self._samples)
            latest = self._samples[-1] if self._samples else None
            return {
                "protocol": "umi-arx.observation.v1",
                "source": self.source,
                "ready": latest is not None and self._last_error is None,
                "sample_count": count,
                "latest_observation_age_ms": (
                    None
                    if latest is None
                    else round(
                        (time.monotonic() - latest.captured_at_monotonic_s) * 1000,
                        3,
                    )
                ),
                "last_camera_frame_age_ms": (
                    None
                    if self._last_camera_frame_unix_ns is None
                    else round(
                        (time.time_ns() - self._last_camera_frame_unix_ns) / 1_000_000,
                        3,
                    )
                ),
                "last_error": self._last_error,
                "config": self.config_path,
            }

    def snapshot(
        self, sample_count: int, history_seconds: float, wait_timeout_seconds: float
    ) -> list[LiveObservationSample]:
        sample_interval = 1.0 / self.config.sampling.frequency_hz
        minimum_span = max(0.0, history_seconds - sample_interval * 1.5)
        deadline = time.monotonic() + wait_timeout_seconds
        with self._condition:
            while True:
                samples = list(self._samples)
                enough_count = len(samples) >= sample_count
                enough_span = (
                    sample_count == 1
                    or enough_count
                    and samples[-1].captured_at_monotonic_s
                    - samples[0].captured_at_monotonic_s
                    >= minimum_span
                )
                if enough_count and enough_span:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = self._last_error or (
                        f"only {len(samples)} samples are buffered for the requested "
                        f"{history_seconds:g}s history"
                    )
                    raise TimeoutError(f"observation source is not ready: {detail}")
                self._condition.wait(remaining)

        newest = samples[-1]
        age = time.monotonic() - newest.captured_at_monotonic_s
        if age > self.config.sampling.max_observation_age_seconds:
            raise RuntimeError(f"latest complete observation is {age:.3f}s old")
        if sample_count == 1:
            return [newest]

        start_time = newest.captured_at_monotonic_s - history_seconds
        targets = np.linspace(start_time, newest.captured_at_monotonic_s, sample_count)
        selected_indices = [
            min(
                range(len(samples)),
                key=lambda index: abs(
                    samples[index].captured_at_monotonic_s - float(target)
                ),
            )
            for target in targets
        ]
        if len(set(selected_indices)) != sample_count:
            raise ValueError(
                "requested history is too short for distinct samples; increase history_seconds"
            )
        return [samples[index] for index in selected_indices]

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=3.0)
