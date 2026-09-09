"""ARX5 evaluation client for a structured external multimodal planner.

The client is shadow-only by default. Pass --arm explicitly to allow this
script to submit teleoperation, matched-pose, hold, or agent waypoints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
import sys
import time
from multiprocessing.managers import SharedMemoryManager
from typing import Any, Callable, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import cv2
import numpy as np
import scipy.spatial.transform as st
import zmq

from modules.arx5_env import Arx5Env
from peripherals.gamepad_teleop import GamepadTeleop
from peripherals.keyboard_teleop import KeyboardTeleop
from peripherals.keystroke_counter import Key, KeyCode, KeystrokeCounter
from peripherals.spacemouse_shared_memory import Spacemouse
from utils.agent_action_safety import SafetyLimits, evaluate_plan_response
from utils.other_util import precise_wait


PROTOCOL_VERSION = 1
OBS_RESOLUTION = (224, 224)
OBS_HORIZON = 16
HISTORY_INDICES = np.array([3, 7, 11, 15], dtype=np.int64)
ACTION_HISTORY_LIMIT = 8
MAX_CONSECUTIVE_FAILURES = 3
PLANNER_POLL_MS = 50
MAX_PLANNER_WIRE_BYTES = 32 * 1024 * 1024
TASK_ALIASES = {
    "cup": (
        "pick up an espresso cup and place it onto a saucer with the cup handle "
        "oriented to the left of the robot"
    ),
    "towel": (
        "grasp the left edge of the towel and move it to the right, folding it in half"
    ),
    "mouse": "pick up the mouse and place it on the mouse pad",
}


@dataclass(frozen=True)
class PlannerExchange:
    response: Any = None
    control: Optional[str] = None
    error: Optional[str] = None
    elapsed: float = 0.0


class PlannerReqClient:
    """REQ client that remains responsive to operator stop keys."""

    def __init__(self, ip: str, port: int):
        self.endpoint = f"tcp://{ip}:{port}"
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self.socket = None
        self._rebuild_socket()

    def _rebuild_socket(self) -> None:
        if self.socket is not None:
            try:
                self.poller.unregister(self.socket)
            except KeyError:
                pass
            self.socket.close(linger=0)
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        socket.setsockopt(zmq.MAXMSGSIZE, MAX_PLANNER_WIRE_BYTES)
        socket.connect(self.endpoint)
        self.socket = socket
        self.poller.register(socket, zmq.POLLIN)

    def close(self) -> None:
        if self.socket is not None:
            try:
                self.poller.unregister(self.socket)
            except KeyError:
                pass
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()

    def exchange(
        self,
        request: dict[str, Any],
        *,
        timeout: float,
        key_counter: KeystrokeCounter,
        pump_ui: Callable[[], None],
    ) -> PlannerExchange:
        start = time.monotonic()
        try:
            self.socket.send_multipart(_encode_planner_request(request))
        except (zmq.ZMQError, TypeError, ValueError) as exc:
            self._rebuild_socket()
            return PlannerExchange(
                error=f"planner_send_failed: {exc}",
                elapsed=time.monotonic() - start,
            )

        deadline = start + timeout
        while True:
            pump_ui()
            control = _consume_agent_control(key_counter.get_press_events())
            if control is not None:
                self._rebuild_socket()
                return PlannerExchange(
                    control=control,
                    error=f"planner_request_cancelled_by_{control}",
                    elapsed=time.monotonic() - start,
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._rebuild_socket()
                return PlannerExchange(
                    error=f"planner_timeout_after_{timeout:g}s",
                    elapsed=time.monotonic() - start,
                )
            wait_ms = max(1, min(PLANNER_POLL_MS, int(remaining * 1000)))
            try:
                events = dict(self.poller.poll(wait_ms))
            except zmq.ZMQError as exc:
                self._rebuild_socket()
                return PlannerExchange(
                    error=f"planner_poll_failed: {exc}",
                    elapsed=time.monotonic() - start,
                )
            if self.socket not in events:
                continue
            try:
                response = self.socket.recv_json()
            except (zmq.ZMQError, TypeError, ValueError) as exc:
                self._rebuild_socket()
                return PlannerExchange(
                    error=f"planner_receive_failed: {exc}",
                    elapsed=time.monotonic() - start,
                )
            return PlannerExchange(
                response=response,
                elapsed=time.monotonic() - start,
            )


class _NoKeyCounter:
    def get_press_events(self) -> list[Any]:
        return []


def _consume_agent_control(events) -> Optional[str]:
    stop_requested = False
    for key_stroke in events:
        if key_stroke == KeyCode(char="q"):
            return "quit"
        if key_stroke == KeyCode(char="s"):
            stop_requested = True
    return "stop" if stop_requested else None


def _wire_json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_wire_json_safe(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("planner request keys must be strings")
        return {key: _wire_json_safe(item) for key, item in value.items()}
    raise ValueError(f"unsupported planner request value: {type(value).__name__}")


def _encode_planner_request(request: dict[str, Any]) -> list[bytes]:
    header = dict(request)
    frames = header.pop("frames", None)
    parts: list[bytes]
    if header.get("type") == "PLAN":
        rgb = _to_uint8_rgb(frames)
        header["frame_shape"] = list(rgb.shape)
        header["frame_dtype"] = "uint8"
        parts = [b"", rgb.tobytes(order="C")]
    elif frames is not None:
        raise ValueError("only PLAN requests may include frames")
    else:
        parts = [b""]
    safe_header = _wire_json_safe(header)
    parts[0] = json.dumps(
        safe_header, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if sum(len(part) for part in parts) > MAX_PLANNER_WIRE_BYTES:
        raise ValueError("planner request exceeds wire size limit")
    return parts


def _extract_plan_for_safety(
    response: Any, *, episode_id: int, step_index: int
) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError("planner response must be an object")
    if response.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("planner protocol version mismatch")
    if response.get("type") != "PLAN_RESULT":
        raise ValueError("planner response type must be PLAN_RESULT")
    if response.get("episode_id") != episode_id:
        raise ValueError("planner response episode_id mismatch")
    if response.get("step_index") != step_index:
        raise ValueError("planner response step_index mismatch")
    if response.get("error") is not None:
        raise ValueError(f"planner returned error: {response['error']!r}")
    plan = response.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("planner response plan must be an object")
    return {"type": "PLAN_RESULT", **plan}


def _parse_camera_reorder(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise click.BadParameter("camera reorder must contain at least one index")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise click.BadParameter("camera reorder must be comma-separated integers") from exc


def _to_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"camera frames must have shape (T,H,W,3), got {frames.shape}")
    if not np.all(np.isfinite(frames)):
        raise ValueError("camera frames contain NaN or Inf")
    if frames.dtype == np.uint8:
        result = frames.copy()
    elif frames.min() >= 0.0 and frames.max() <= 1.0:
        result = np.rint(np.clip(frames, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif frames.min() >= -1.0 and frames.max() <= 1.0:
        result = np.rint((np.clip(frames, -1.0, 1.0) + 1.0) * 127.5).astype(
            np.uint8
        )
    else:
        result = np.clip(frames, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(result)


def _extract_history(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    frames_all = np.asarray(obs["camera0_rgb"])
    positions_all = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    rotations_all = np.asarray(obs["robot0_eef_rot_axis_angle"], dtype=np.float64)
    grippers_all = np.asarray(obs["robot0_gripper_width"], dtype=np.float64).reshape(-1)
    timestamps_all = np.asarray(obs["timestamp"], dtype=np.float64)
    required = int(HISTORY_INDICES[-1]) + 1
    values = {
        "camera0_rgb": frames_all,
        "robot0_eef_pos": positions_all,
        "robot0_eef_rot_axis_angle": rotations_all,
        "robot0_gripper_width": grippers_all,
        "timestamp": timestamps_all,
    }
    for name, value in values.items():
        if len(value) < required:
            raise ValueError(f"{name} has {len(value)} samples; {required} required")

    frames = _to_uint8_rgb(frames_all[HISTORY_INDICES])
    if frames.shape[1:] != (OBS_RESOLUTION[1], OBS_RESOLUTION[0], 3):
        raise ValueError(
            f"expected RGB frames at 224x224, got shape {frames.shape}"
        )
    pose_history = np.concatenate(
        [positions_all[HISTORY_INDICES], rotations_all[HISTORY_INDICES]], axis=-1
    )
    gripper_history = grippers_all[HISTORY_INDICES]
    timestamp_history = timestamps_all[HISTORY_INDICES]
    if pose_history.shape != (4, 6):
        raise ValueError(f"pose history must have shape (4,6), got {pose_history.shape}")
    if gripper_history.shape != (4,):
        raise ValueError(
            f"gripper history must have shape (4,), got {gripper_history.shape}"
        )
    if not np.all(np.isfinite(pose_history)) or not np.all(
        np.isfinite(gripper_history)
    ):
        raise ValueError("robot history contains NaN or Inf")
    if not np.all(np.isfinite(timestamp_history)):
        raise ValueError("observation timestamps contain NaN or Inf")
    if np.any(np.diff(timestamp_history) < 0.0):
        raise ValueError("observation timestamps are not chronological")
    return {
        "frames": frames,
        "pose_history": pose_history,
        "gripper_history": gripper_history,
        "timestamp_history": timestamp_history,
    }


def _latest_robot_target(env: Arx5Env) -> np.ndarray:
    state = env.get_robot_state()[0]
    pose = np.asarray(state["ActualTCPPose"], dtype=np.float64)
    gripper = float(state["gripper_position"])
    target = np.concatenate([pose, [gripper]])
    if target.shape != (7,) or not np.all(np.isfinite(target)):
        raise RuntimeError("latest robot target state is malformed or non-finite")
    return target


def _submit_single_target(
    env: Arx5Env,
    target: np.ndarray,
    *,
    arm: bool,
    dt: float,
) -> float:
    if not arm:
        raise RuntimeError("robot target submission requires --arm")
    timestamp = time.time() + max(dt, 0.25)
    env.exec_actions(
        actions=np.asarray([target], dtype=np.float64),
        timestamps=np.asarray([timestamp], dtype=np.float64),
        compensate_latency=False,
        dynamic_latency=False,
    )
    return timestamp


def _submit_latest_hold(env: Arx5Env, *, arm: bool, dt: float) -> np.ndarray:
    hold_target = _latest_robot_target(env)
    if arm:
        _submit_single_target(env, hold_target, arm=arm, dt=dt)
    return hold_target


def _show_preview(
    window: str,
    frame: np.ndarray,
    *,
    episode_id: int,
    mode: str,
    detail: str = "",
) -> None:
    image = _to_uint8_rgb(np.asarray(frame)[None])[0]
    image = np.ascontiguousarray(image.copy())
    text = f"Episode: {episode_id}  {mode}"
    if detail:
        text += f"  {detail}"
    cv2.putText(
        image,
        text,
        (10, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        lineType=cv2.LINE_AA,
        thickness=3,
        color=(0, 0, 0),
    )
    cv2.putText(
        image,
        text,
        (10, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        lineType=cv2.LINE_AA,
        thickness=1,
        color=(255, 255, 255),
    )
    cv2.imshow(window, image[..., ::-1])
    cv2.pollKey()


def _next_log_episode_id(output: str, replay_episode_id: int) -> int:
    episode_id = replay_episode_id
    while os.path.exists(os.path.join(output, "obs", str(episode_id))) or os.path.exists(
        os.path.join(output, "action", str(episode_id))
    ):
        episode_id += 1
    return episode_id


def _load_match_data(path: str, episode: int) -> dict[str, Any]:
    filename = os.path.join(path, "obs", str(episode), "0.npy")
    return np.load(filename, allow_pickle=True).item()


def _match_episode_range(path: str, episode: int) -> tuple[int, int]:
    def exists(candidate: int) -> bool:
        return os.path.exists(os.path.join(path, "obs", str(candidate), "0.npy"))

    if not exists(episode):
        raise click.BadParameter(f"match episode {episode} does not exist")
    minimum = episode
    while exists(minimum - 1):
        minimum -= 1
    maximum = episode
    while exists(maximum + 1):
        maximum += 1
    return minimum, maximum


def _health_response_ok(response: Any) -> bool:
    return (
        isinstance(response, dict)
        and response.get("type") == "HEALTH_RESULT"
        and response.get("status") in {"ok", "ready"}
    )


def _verify_safe_robot_server(port: int, timeout_ms: int = 1500) -> dict[str, Any]:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.connect(f"tcp://127.0.0.1:{port}")
    try:
        socket.send_pyobj({"cmd": "GET_SERVER_INFO", "data": None})
        response = socket.recv_pyobj()
    except zmq.ZMQError as exc:
        raise click.ClickException(
            f"safe robot server verification failed on port {port}: {exc}"
        ) from exc
    finally:
        socket.close(linger=0)
        context.term()
    if not isinstance(response, dict) or response.get("cmd") != "GET_SERVER_INFO":
        raise click.ClickException("robot port is not the agent-safe ZMQ server")
    data = response.get("data")
    if (
        not isinstance(data, dict)
        or data.get("name") != "arx5_agent_safe"
        or data.get("protocol_version") != 1
    ):
        raise click.ClickException("robot port did not identify as arx5_agent_safe")
    return data


@click.command()
@click.option("--output", "-o", required=True, help="Directory to save recording")
@click.option("--planner-ip", "--policy_ip", default="localhost", show_default=True)
@click.option("--planner-port", "--policy_port", default=8766, show_default=True)
@click.option("--robot-port", default=8765, show_default=True)
@click.option("--match-dataset-path", "--match_dataset_path", default=None, type=str)
@click.option("--match-episode", "--match_episode", default=None, type=int)
@click.option("--camera-reorder", "--camera_reorder", "-cr", default="0")
@click.option("--vis-camera-idx", "--vis_camera_idx", default=0, type=int)
@click.option("--init-joints", "--init_joints", "-j", is_flag=True, default=False)
@click.option(
    "--max-duration", "--max_duration", "-md", default=2_000_000.0, type=float
)
@click.option(
    "--frequency",
    "-f",
    default=5.0,
    type=click.FloatRange(min=0.1),
    show_default=True,
)
@click.option(
    "--command-latency", "--command_latency", "-cl", default=0.01, type=float
)
@click.option(
    "-nm", "--no-mirror", "--no_mirror", "no_mirror", is_flag=True, default=False
)
@click.option(
    "--gripper-mask/--no-gripper-mask", default=False, show_default=True
)
@click.option("--mirror-swap", "--mirror_swap", is_flag=True, default=False)
@click.option("--task-name", "--task_name", type=str, required=True)
@click.option(
    "--teleop-device",
    type=click.Choice(["keyboard", "spacemouse", "gamepad"]),
    default="keyboard",
    show_default=True,
)
@click.option(
    "--planner-timeout",
    type=click.FloatRange(min=0.1),
    default=20.0,
    show_default=True,
)
@click.option("--arm", is_flag=True, default=False, help="Allow robot waypoints.")
@click.option(
    "--max-position-step",
    type=click.FloatRange(min=1e-6),
    default=0.005,
    show_default=True,
)
@click.option(
    "--max-rotation-step",
    type=click.FloatRange(min=1e-6),
    default=0.035,
    show_default=True,
)
@click.option(
    "--max-gripper-step",
    type=click.FloatRange(min=1e-6),
    default=0.002,
    show_default=True,
)
@click.option(
    "--episode-radius",
    type=click.FloatRange(min=1e-6),
    default=0.12,
    show_default=True,
)
@click.option(
    "--min-z-below-start",
    type=click.FloatRange(min=1e-6),
    default=0.08,
    show_default=True,
)
@click.option(
    "--min-confidence",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.6,
    show_default=True,
)
@click.option(
    "--max-observation-age",
    type=click.FloatRange(min=0.1),
    default=1.0,
    show_default=True,
)
@click.option(
    "--min-image-mean",
    type=click.FloatRange(min=0.0, max=255.0),
    default=3.0,
    show_default=True,
)
@click.option(
    "--max-consecutive-failures",
    type=click.IntRange(min=1),
    default=MAX_CONSECUTIVE_FAILURES,
    show_default=True,
)
def main(
    output,
    planner_ip,
    planner_port,
    robot_port,
    match_dataset_path,
    match_episode,
    camera_reorder,
    vis_camera_idx,
    init_joints,
    max_duration,
    frequency,
    command_latency,
    no_mirror,
    gripper_mask,
    mirror_swap,
    task_name,
    teleop_device,
    planner_timeout,
    arm,
    max_position_step,
    max_rotation_step,
    max_gripper_step,
    episode_radius,
    min_z_below_start,
    min_confidence,
    max_observation_age,
    min_image_mean,
    max_consecutive_failures,
):
    try:
        available_cpus = os.sched_getaffinity(0)
        if 7 in available_cpus:
            os.sched_setaffinity(0, [7])
    except (AttributeError, OSError):
        pass

    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, "obs"), exist_ok=True)
    os.makedirs(os.path.join(output, "action"), exist_ok=True)
    mode_name = "ARMED" if arm else "SHADOW"
    print(f"Agent client mode: {mode_name}")
    if not task_name.strip():
        raise click.BadParameter("--task-name must not be empty")
    requested_task = task_name.strip()
    task_instruction = TASK_ALIASES.get(requested_task.lower(), requested_task)
    if task_instruction != requested_task:
        print(f"Expanded task alias {requested_task!r}: {task_instruction}")
    if not arm:
        print(
            "SHADOW disables teleop and agent waypoint submission, but Arx5Env "
            "still starts its robot bridge and RESET_TO_HOME sequence."
        )

    limits = SafetyLimits(
        max_position_step=max_position_step,
        max_rotation_step=max_rotation_step,
        max_gripper_step=max_gripper_step,
        max_episode_radius=episode_radius,
        min_z_below_start=min_z_below_start,
        min_confidence=min_confidence,
    )
    dt = 1.0 / frequency
    max_gripper_width = limits.max_gripper_width
    gripper_speed = 0.04
    cartesian_speed = 0.4
    orientation_speed = 0.8

    match_episode_min = match_episode_max = None
    if match_dataset_path is not None:
        if match_episode is None:
            raise click.BadParameter(
                "--match-episode is required with --match-dataset-path"
            )
        match_episode_min, match_episode_max = _match_episode_range(
            match_dataset_path, match_episode
        )
        print(
            f"match_episode: {match_episode}, range: "
            f"[{match_episode_min}, {match_episode_max}]"
        )

    robots_config = [
        {
            "robot_type": "arx5",
            "robot_ip": "127.0.0.1",
            "robot_port": robot_port,
            "robot_obs_latency": 0.005,
            "robot_action_latency": 0.04,
            "height_threshold": -0.2,
            "sphere_radius": 0.1,
            "sphere_center": [0, -0.06, -0.185],
        }
    ]
    server_info = _verify_safe_robot_server(robot_port)
    print(f"Verified agent-safe robot server: {server_info['limits']}")
    planner = PlannerReqClient(planner_ip, planner_port)
    should_quit = False

    try:
        health = planner.exchange(
            {"protocol_version": PROTOCOL_VERSION, "type": "HEALTH"},
            timeout=planner_timeout,
            key_counter=_NoKeyCounter(),
            pump_ui=lambda: None,
        )
        if health.error is not None or not _health_response_ok(health.response):
            raise click.ClickException(
                health.error
                or f"invalid planner health response: {health.response!r}"
            )
        print(f"Planner health check passed at {planner.endpoint}")

        with SharedMemoryManager() as shm_manager:
            if teleop_device == "keyboard":
                teleop = KeyboardTeleop(fine_scale=0.1, fast_scale=0.5)
            elif teleop_device == "spacemouse":
                teleop = Spacemouse(shm_manager=shm_manager, deadzone=0.1)
            else:
                teleop = GamepadTeleop()

            with teleop as sm, KeystrokeCounter() as key_counter, Arx5Env(
                output_dir=output,
                robots_config=robots_config,
                frequency=frequency,
                obs_image_resolution=OBS_RESOLUTION,
                max_obs_buffer_size=1000,
                obs_float32=True,
                camera_reorder=_parse_camera_reorder(camera_reorder),
                init_joints=init_joints,
                enable_multi_cam_vis=False,
                camera_obs_latency=0.17,
                camera_obs_horizon=OBS_HORIZON,
                robot_obs_horizon=OBS_HORIZON,
                no_mirror=no_mirror,
                gripper_mask=gripper_mask,
                mirror_swap=mirror_swap,
                max_pos_speed=2.0,
                max_rot_speed=6.0,
                shm_manager=shm_manager,
            ) as env:
                cv2.setNumThreads(2)
                print("Waiting for env ready.")
                while not env.is_ready:
                    time.sleep(0.1)
                print("Env is ready")

                preview_window = "arx5-agent"
                cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(preview_window, 960, 720)
                initial_obs = env.get_obs()
                if f"camera{vis_camera_idx}_rgb" not in initial_obs:
                    raise click.BadParameter(
                        f"camera{vis_camera_idx}_rgb is unavailable for preview"
                    )
                _show_preview(
                    preview_window,
                    initial_obs[f"camera{vis_camera_idx}_rgb"][-1],
                    episode_id=env.replay_buffer.n_episodes,
                    mode=mode_name,
                    detail="health check",
                )

                log_episode_id = _next_log_episode_id(
                    output, env.replay_buffer.n_episodes
                )
                while not should_quit:
                    print(f"Human in control ({mode_name}).")
                    robot_states = env.get_robot_state()
                    target_pose = np.stack(
                        [state["ActualTCPPose"] for state in robot_states]
                    )
                    gripper_target_pos = np.asarray(
                        [state["gripper_position"] for state in robot_states]
                    )
                    control_robot_idx_list = [0]
                    t_start = time.monotonic()
                    iter_idx = 0
                    start_agent = False

                    while not should_quit and not start_agent:
                        t_cycle_end = t_start + (iter_idx + 1) * dt
                        t_sample = t_cycle_end - command_latency
                        t_command_target = t_cycle_end + dt
                        obs = env.get_obs()
                        _show_preview(
                            preview_window,
                            obs[f"camera{vis_camera_idx}_rgb"][-1],
                            episode_id=log_episode_id,
                            mode=f"{mode_name} HUMAN",
                        )

                        for key_stroke in key_counter.get_press_events():
                            if key_stroke == KeyCode(char="q"):
                                _submit_latest_hold(env, arm=arm, dt=dt)
                                env.end_episode()
                                should_quit = True
                                break
                            if key_stroke == KeyCode(char="c"):
                                start_agent = True
                            elif key_stroke == KeyCode(char="e"):
                                if match_episode is not None:
                                    match_episode = min(
                                        match_episode + 1, match_episode_max
                                    )
                                    print(f"match_episode: {match_episode}")
                            elif key_stroke == KeyCode(char="w"):
                                if match_episode is not None:
                                    match_episode = max(
                                        match_episode - 1, match_episode_min
                                    )
                                    print(f"match_episode: {match_episode}")
                            elif key_stroke == KeyCode(char="m"):
                                if (
                                    match_episode is not None
                                    and match_dataset_path is not None
                                ):
                                    match_data = _load_match_data(
                                        match_dataset_path, match_episode
                                    )
                                    start_pos = match_data["obs"][
                                        "robot0_eef_pos"
                                    ][-1]
                                    start_rot = match_data["obs"][
                                        "robot0_eef_rot_axis_angle"
                                    ][-1]
                                    target_pose[0] = np.concatenate(
                                        [start_pos, start_rot]
                                    )
                                    gripper_target_pos[0] = float(
                                        match_data["obs"][
                                            "robot0_gripper_width"
                                        ][-1]
                                    )
                                    matched_target = np.concatenate(
                                        [target_pose[0], [gripper_target_pos[0]]]
                                    )
                                    if arm:
                                        env.exec_actions(
                                            actions=np.asarray([matched_target]),
                                            timestamps=np.asarray(
                                                [
                                                    t_command_target
                                                    - time.monotonic()
                                                    + time.time()
                                                    + 5.0
                                                ]
                                            ),
                                            compensate_latency=False,
                                            dynamic_latency=False,
                                        )
                                        precise_wait(t_cycle_end + 5.0)
                                        iter_idx += int(5.0 / dt)
                                    else:
                                        print("SHADOW: matched pose not submitted")
                                    continue
                            elif key_stroke == Key.backspace:
                                if click.confirm("Are you sure to drop an episode?"):
                                    env.drop_episode()
                                    key_counter.clear()
                            elif key_stroke == KeyCode(char="a"):
                                control_robot_idx_list = list(
                                    range(target_pose.shape[0])
                                )
                            elif key_stroke == KeyCode(char="1"):
                                control_robot_idx_list = [0]
                            elif key_stroke == KeyCode(char="2"):
                                if target_pose.shape[0] > 1:
                                    control_robot_idx_list = [1]

                        if should_quit or start_agent:
                            break
                        precise_wait(t_sample)
                        sm_state = sm.get_motion_state_transformed()
                        dpos = sm_state[:3] * (0.5 / frequency) * cartesian_speed
                        drot_xyz = (
                            sm_state[3:] * (1.5 / frequency) * orientation_speed
                        )
                        drot = st.Rotation.from_euler("xyz", drot_xyz)
                        for robot_idx in control_robot_idx_list:
                            target_pose[robot_idx, :3] += dpos
                            target_pose[robot_idx, 3:] = (
                                st.Rotation.from_rotvec(target_pose[robot_idx, 3:])
                                * drot
                            ).as_rotvec()

                        reset_requested = (
                            sm.consume_reset_request()
                            if hasattr(sm, "consume_reset_request")
                            else sm.is_button_pressed(0)
                            and sm.is_button_pressed(1)
                        )
                        gripper_delta = 0.0
                        if reset_requested:
                            if arm:
                                print("Reset robot arm to home. Please wait...")
                                for robot_idx in control_robot_idx_list:
                                    env.robots[robot_idx].reset_to_home()
                                robot_states = env.get_robot_state()
                                for robot_idx in control_robot_idx_list:
                                    target_pose[robot_idx] = robot_states[robot_idx][
                                        "ActualTCPPose"
                                    ]
                                    gripper_target_pos[robot_idx] = robot_states[
                                        robot_idx
                                    ]["gripper_position"]
                            else:
                                print("SHADOW: home reset not submitted")
                        elif sm.is_button_pressed(0):
                            gripper_delta = -gripper_speed / frequency
                        elif sm.is_button_pressed(1):
                            gripper_delta = gripper_speed / frequency
                        for robot_idx in control_robot_idx_list:
                            gripper_target_pos[robot_idx] = np.clip(
                                gripper_target_pos[robot_idx] + gripper_delta,
                                0.0,
                                max_gripper_width,
                            )

                        teleop_action = np.zeros(7 * target_pose.shape[0])
                        for robot_idx in range(target_pose.shape[0]):
                            teleop_action[
                                7 * robot_idx : 7 * robot_idx + 6
                            ] = target_pose[robot_idx]
                            teleop_action[7 * robot_idx + 6] = (
                                gripper_target_pos[robot_idx]
                            )
                        if arm:
                            env.exec_actions(
                                actions=np.asarray([teleop_action]),
                                timestamps=np.asarray(
                                    [
                                        t_command_target
                                        - time.monotonic()
                                        + time.time()
                                    ]
                                ),
                                compensate_latency=False,
                                dynamic_latency=False,
                            )
                        precise_wait(t_cycle_end)
                        iter_idx += 1

                    if should_quit:
                        break

                    episode_id = log_episode_id
                    obs_dir = os.path.join(output, "obs", str(episode_id))
                    action_dir = os.path.join(output, "action", str(episode_id))
                    os.makedirs(obs_dir, exist_ok=True)
                    os.makedirs(action_dir, exist_ok=True)
                    eval_start = time.time() + 1.0
                    env.start_episode(eval_start)
                    episode_start_target = _latest_robot_target(env)
                    episode_start_pose = episode_start_target[:6].copy()
                    action_history = deque(maxlen=ACTION_HISTORY_LIMIT)
                    last_submitted_target = None
                    consecutive_failures = 0
                    step_index = 0
                    precise_wait(eval_start - 1.0 / 60.0, time_func=time.time)
                    print(f"Agent episode {episode_id} started ({mode_name}).")
                    return_to_human = False
                    episode_ended = False

                    while not should_quit and not return_to_human:
                        step_started = time.monotonic()
                        if time.time() - eval_start > max_duration:
                            print("Max duration reached; returning to human.")
                            hold_target = _submit_latest_hold(env, arm=arm, dt=dt)
                            np.save(
                                os.path.join(action_dir, f"{step_index}.npy"),
                                {
                                    "raw_response": None,
                                    "safe_target": hold_target,
                                    "submitted": bool(arm),
                                    "reason": "max_duration_reached",
                                    "mode": mode_name,
                                },
                                allow_pickle=True,
                            )
                            return_to_human = True
                            break
                        obs = env.get_obs()
                        try:
                            history = _extract_history(obs)
                            observation_age = (
                                time.time() - history["timestamp_history"][-1]
                            )
                            if observation_age < -0.5:
                                raise ValueError(
                                    f"latest observation is {-observation_age:.3f}s "
                                    "in the future"
                                )
                            if observation_age > max_observation_age:
                                raise ValueError(
                                    f"latest observation is {observation_age:.3f}s old"
                                )
                            image_mean = float(np.mean(history["frames"][-1]))
                            if image_mean < min_image_mean:
                                raise ValueError(
                                    f"latest wrist image mean {image_mean:.3f} is too dark"
                                )
                        except (KeyError, TypeError, ValueError) as exc:
                            print(f"Observation rejected: {exc}")
                            hold_target = _submit_latest_hold(env, arm=arm, dt=dt)
                            np.save(
                                os.path.join(action_dir, f"{step_index}.npy"),
                                {
                                    "raw_response": None,
                                    "transport_error": None,
                                    "safe_target": hold_target,
                                    "submitted": bool(arm),
                                    "reason": f"invalid_observation: {exc}",
                                    "mode": mode_name,
                                },
                                allow_pickle=True,
                            )
                            consecutive_failures += 1
                            if consecutive_failures >= max_consecutive_failures:
                                return_to_human = True
                            precise_wait(step_started + dt)
                            step_index += 1
                            continue

                        request = {
                            "protocol_version": PROTOCOL_VERSION,
                            "type": "PLAN",
                            "episode_id": int(episode_id),
                            "step_index": int(step_index),
                            "task": task_instruction,
                            "action_frame": "tcp_tool",
                            "frames": history["frames"],
                            "pose_history": history["pose_history"],
                            "gripper_history": history["gripper_history"],
                            "timestamp_history": history["timestamp_history"],
                            "action_history": list(action_history),
                            "safety_limits": limits.to_protocol_dict(),
                        }
                        np.save(
                            os.path.join(obs_dir, f"{step_index}.npy"),
                            {
                                "protocol_version": PROTOCOL_VERSION,
                                "frames": history["frames"],
                                "pose_history": history["pose_history"],
                                "gripper_history": history["gripper_history"],
                                "timestamp_history": history["timestamp_history"],
                                "observation_age": float(observation_age),
                                "latest_image_mean": image_mean,
                                "action_history": list(action_history),
                                "task": task_instruction,
                                "requested_task": requested_task,
                                "action_frame": "tcp_tool",
                            },
                            allow_pickle=True,
                        )
                        _show_preview(
                            preview_window,
                            history["frames"][-1],
                            episode_id=episode_id,
                            mode=f"{mode_name} AGENT",
                            detail=f"step {step_index} waiting",
                        )
                        exchange = planner.exchange(
                            request,
                            timeout=planner_timeout,
                            key_counter=key_counter,
                            pump_ui=cv2.pollKey,
                        )

                        if exchange.control is not None:
                            hold_target = _submit_latest_hold(env, arm=arm, dt=dt)
                            np.save(
                                os.path.join(action_dir, f"{step_index}.npy"),
                                {
                                    "raw_response": exchange.response,
                                    "transport_error": exchange.error,
                                    "control": exchange.control,
                                    "safe_target": hold_target,
                                    "submitted": bool(arm),
                                    "reason": f"operator_{exchange.control}",
                                },
                                allow_pickle=True,
                            )
                            env.end_episode()
                            episode_ended = True
                            should_quit = exchange.control == "quit"
                            return_to_human = not should_quit
                            break

                        post_response_control = _consume_agent_control(
                            key_counter.get_press_events()
                        )
                        if post_response_control is not None:
                            hold_target = _submit_latest_hold(env, arm=arm, dt=dt)
                            np.save(
                                os.path.join(action_dir, f"{step_index}.npy"),
                                {
                                    "raw_response": exchange.response,
                                    "transport_error": exchange.error,
                                    "control": post_response_control,
                                    "safe_target": hold_target,
                                    "submitted": bool(arm),
                                    "reason": f"operator_{post_response_control}",
                                },
                                allow_pickle=True,
                            )
                            env.end_episode()
                            episode_ended = True
                            should_quit = post_response_control == "quit"
                            return_to_human = not should_quit
                            break

                        if time.time() - eval_start > max_duration:
                            print("Max duration reached while waiting for planner.")
                            hold_target = _submit_latest_hold(env, arm=arm, dt=dt)
                            np.save(
                                os.path.join(action_dir, f"{step_index}.npy"),
                                {
                                    "raw_response": exchange.response,
                                    "transport_error": exchange.error,
                                    "safe_target": hold_target,
                                    "submitted": bool(arm),
                                    "reason": "max_duration_reached",
                                    "mode": mode_name,
                                },
                                allow_pickle=True,
                            )
                            return_to_human = True
                            break

                        measured_target = _latest_robot_target(env)
                        response_validation_error = None
                        if exchange.error is None:
                            try:
                                response_for_safety = _extract_plan_for_safety(
                                    exchange.response,
                                    episode_id=episode_id,
                                    step_index=step_index,
                                )
                            except ValueError as exc:
                                response_for_safety = None
                                response_validation_error = str(exc)
                        else:
                            response_for_safety = None
                        last_submitted_target_before = (
                            None
                            if last_submitted_target is None
                            else last_submitted_target.copy()
                        )
                        decision = evaluate_plan_response(
                            response_for_safety,
                            current_pose=measured_target[:6],
                            current_gripper=measured_target[6],
                            episode_start_pose=episode_start_pose,
                            last_submitted_target=last_submitted_target,
                            limits=limits,
                        )
                        if exchange.error is not None:
                            decision_reason = exchange.error
                        elif response_validation_error is not None:
                            decision_reason = (
                                f"invalid_planner_response: {response_validation_error}"
                            )
                        else:
                            decision_reason = decision.reason

                        submitted = False
                        submitted_timestamp = None
                        if arm:
                            submitted_timestamp = _submit_single_target(
                                env, decision.target, arm=arm, dt=dt
                            )
                            submitted = True
                            last_submitted_target = decision.target.copy()

                        action_record = {
                            "protocol_version": PROTOCOL_VERSION,
                            "step_index": int(step_index),
                            "request_timestamp": float(history["timestamp_history"][-1]),
                            "response_elapsed": float(exchange.elapsed),
                            "raw_response": exchange.response,
                            "transport_error": exchange.error,
                            "response_validation_error": response_validation_error,
                            "measured_target": measured_target,
                            "episode_start_pose": episode_start_pose,
                            "last_submitted_target_before": last_submitted_target_before,
                            "safety": decision.to_record(),
                            "effective_reason": decision_reason,
                            "submitted": submitted,
                            "submitted_timestamp": submitted_timestamp,
                            "mode": mode_name,
                            "action_frame": "tcp_tool",
                        }
                        np.save(
                            os.path.join(action_dir, f"{step_index}.npy"),
                            action_record,
                            allow_pickle=True,
                        )

                        if decision.accepted:
                            action_history.append(
                                {
                                    "step_index": int(step_index),
                                    "planner_status": decision.planner_status,
                                    "planner_action": response_for_safety,
                                    "measured_tcp_before": measured_target.tolist(),
                                    "target_tcp": decision.target.tolist(),
                                    "reason": decision.reason,
                                    "submitted": submitted,
                                    "submitted_timestamp": submitted_timestamp,
                                    "action_frame": "tcp_tool",
                                }
                            )
                        if (
                            decision.rejected
                            or exchange.error is not None
                            or response_validation_error is not None
                        ):
                            consecutive_failures += 1
                        else:
                            consecutive_failures = 0

                        print(
                            f"Agent step {step_index}: {decision_reason}; "
                            f"submitted={submitted}; failures={consecutive_failures}"
                        )
                        _show_preview(
                            preview_window,
                            history["frames"][-1],
                            episode_id=episode_id,
                            mode=f"{mode_name} AGENT",
                            detail=f"{decision.planner_status}: {decision_reason}",
                        )

                        if decision.planner_status in {"done", "human"}:
                            return_to_human = True
                        if consecutive_failures >= max_consecutive_failures:
                            print(
                                f"{max_consecutive_failures} consecutive planner "
                                "failures; returning to human."
                            )
                            return_to_human = True
                        precise_wait(step_started + max(dt, 0.25))
                        step_index += 1

                    if not should_quit and not episode_ended:
                        env.end_episode()
                        print("Agent episode stopped; human control restored.")
                    log_episode_id += 1
    finally:
        planner.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
