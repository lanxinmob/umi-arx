"""Minimal OpenVLA REST client for ARX5 real-robot evaluation.

Controls match eval_arx5.py: use teleoperation to set the start pose, press C
to hand control to OpenVLA, S to return to teleoperation, and Q to quit.
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing.managers import SharedMemoryManager
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import cv2
import numpy as np
import scipy.spatial.transform as st

from modules.arx5_env import Arx5Env
from peripherals.gamepad_teleop import GamepadTeleop
from peripherals.keyboard_teleop import KeyboardTeleop
from peripherals.keystroke_counter import KeyCode, KeystrokeCounter
from peripherals.spacemouse_shared_memory import Spacemouse
from utils.other_util import precise_wait
from utils.pose_util import apply_delta_pose


OBS_RESOLUTION = (224, 224)
MAX_GRIPPER_WIDTH = 0.085
TASK_ALIASES = {
    "cup": "pick up the cup",
    "mouse": "pick up the mouse and place it on the mouse pad",
    "towel": "fold the towel in half",
}


class OpenVLAClient:
    def __init__(self, url: str, timeout: float):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))

    def _request(self, request: Request) -> Any:
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenVLA HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenVLA request failed: {exc}") from exc

    def health(self) -> None:
        base_url = self.url.rsplit("/", 1)[0]
        result = self._request(Request(f"{base_url}/health", method="GET"))
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise RuntimeError(f"unexpected OpenVLA health response: {result!r}")
        print(f"OpenVLA ready at {self.url} ({result.get('device', 'unknown device')})")

    def predict(self, image: np.ndarray, instruction: str, unnorm_key: str) -> np.ndarray:
        image = to_uint8_rgb(image)
        payload = json.dumps(
            {
                "image": image.tolist(),
                "instruction": instruction,
                "unnorm_key": unnorm_key,
            },
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        result = self._request(
            Request(
                self.url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        )
        action = np.asarray(result, dtype=np.float64)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"invalid OpenVLA action: {result!r}")
        return action


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (H,W,3), got {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("image contains NaN or Inf")
    if image.dtype == np.uint8:
        result = image.copy()
    elif image.min() >= 0.0 and image.max() <= 1.0:
        result = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        result = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(result)


def openvla_action_to_arx5_target(
    action: np.ndarray,
    current_pose: np.ndarray,
    episode_start_pose: np.ndarray,
    *,
    max_position_delta: float,
    max_rotation_delta: float,
    episode_radius: float,
    min_z_below_start: float,
) -> np.ndarray:
    """Apply an OpenVLA base-frame delta using the UVA ARX5 convention."""
    action = np.asarray(action, dtype=np.float64)
    current_pose = np.asarray(current_pose, dtype=np.float64)
    episode_start_pose = np.asarray(episode_start_pose, dtype=np.float64)
    if action.shape != (7,) or current_pose.shape != (6,) or episode_start_pose.shape != (6,):
        raise ValueError("expected action (7,), current pose (6,), and start pose (6,)")
    if not all(np.all(np.isfinite(value)) for value in (action, current_pose, episode_start_pose)):
        raise ValueError("action and poses must be finite")

    position_delta = action[:3]
    rotation_delta = st.Rotation.from_euler("xyz", action[3:6]).as_rotvec()
    position_delta_norm = np.linalg.norm(position_delta)
    rotation_delta_norm = np.linalg.norm(rotation_delta)
    if position_delta_norm > max_position_delta:
        raise ValueError(
            f"position delta norm {position_delta_norm:g} exceeds "
            f"{max_position_delta:g} m: {position_delta}"
        )
    if rotation_delta_norm > max_rotation_delta:
        raise ValueError(
            f"rotation delta norm {rotation_delta_norm:g} exceeds "
            f"{max_rotation_delta:g} rad: {rotation_delta}"
        )
    if not 0.0 <= action[6] <= 1.0:
        raise ValueError(f"gripper action must be in [0,1], got {action[6]:g}")

    uva_delta_pose = np.concatenate([position_delta, rotation_delta])
    target_pose = apply_delta_pose(current_pose, uva_delta_pose)
    command_position_delta = target_pose[:3] - current_pose[:3]
    command_rotation_delta = (
        st.Rotation.from_rotvec(target_pose[3:])
        * st.Rotation.from_rotvec(current_pose[3:]).inv()
    ).magnitude()
    tolerance = np.finfo(np.float64).eps * 16
    if np.linalg.norm(command_position_delta) > max_position_delta + tolerance:
        raise ValueError(
            f"converted position command exceeds {max_position_delta:g} m: "
            f"{command_position_delta}"
        )
    if command_rotation_delta > max_rotation_delta + tolerance:
        raise ValueError(
            f"converted rotation command {command_rotation_delta:g} exceeds "
            f"{max_rotation_delta:g} rad"
        )
    if np.linalg.norm(target_pose[:3] - episode_start_pose[:3]) > episode_radius:
        raise ValueError(f"target leaves {episode_radius:g} m episode radius")
    if target_pose[2] < episode_start_pose[2] - min_z_below_start:
        raise ValueError(f"target is more than {min_z_below_start:g} m below episode start")

    gripper_width = float(action[6] * MAX_GRIPPER_WIDTH)
    return np.concatenate([target_pose, [gripper_width]])


def parse_camera_reorder(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise click.BadParameter("camera order must be comma-separated integers") from exc
    if not result:
        raise click.BadParameter("camera order must contain at least one index")
    return result


def show_preview(window: str, image: np.ndarray, mode: str, instruction: str) -> None:
    image = to_uint8_rgb(image).copy()
    text = f"{mode}: {instruction}"
    cv2.putText(
        image,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(window, image[..., ::-1])
    cv2.pollKey()


def submit_target(env: Arx5Env, target: np.ndarray, dt: float) -> None:
    env.exec_actions(
        actions=np.asarray([target], dtype=np.float64),
        timestamps=np.asarray([time.time() + max(dt, 0.05)], dtype=np.float64),
        compensate_latency=False,
        dynamic_latency=False,
    )


@click.command()
@click.option("--output", "-o", required=True, help="Rollout output directory")
@click.option("--policy-url", default="http://127.0.0.1:8000/act", show_default=True)
@click.option("--task-name", "--task_name", required=True, type=str)
@click.option("--unnorm-key", default="bridge_orig", show_default=True)
@click.option("--robot-port", default=8765, show_default=True)
@click.option("--camera-reorder", "--camera_reorder", "-cr", default="0")
@click.option("--init-joints", "--init_joints", is_flag=True, default=False)
@click.option("--frequency", "-f", default=5.0, type=click.FloatRange(min=0.1))
@click.option("--max-duration", default=60.0, type=click.FloatRange(min=0.1))
@click.option("--request-timeout", default=5.0, type=click.FloatRange(min=0.1))
@click.option("--max-position-delta", default=0.05, type=click.FloatRange(min=1e-6))
@click.option("--max-rotation-delta", default=0.1, type=click.FloatRange(min=1e-6))
@click.option("--episode-radius", default=0.2, type=click.FloatRange(min=1e-6))
@click.option("--min-z-below-start", default=0.1, type=click.FloatRange(min=1e-6))
@click.option("--min-image-mean", default=3.0, type=click.FloatRange(min=0.0, max=255.0))
@click.option("-nm", "--no-mirror", "--no_mirror", is_flag=True, default=False)
@click.option("--gripper-mask/--no-gripper-mask", default=False, show_default=True)
@click.option(
    "--teleop-device",
    type=click.Choice(["keyboard", "spacemouse", "gamepad"]),
    default="keyboard",
    show_default=True,
)
def main(
    output,
    policy_url,
    task_name,
    unnorm_key,
    robot_port,
    camera_reorder,
    init_joints,
    frequency,
    max_duration,
    request_timeout,
    max_position_delta,
    max_rotation_delta,
    episode_radius,
    min_z_below_start,
    min_image_mean,
    no_mirror,
    gripper_mask,
    teleop_device,
):
    instruction = TASK_ALIASES.get(task_name.strip().lower(), task_name.strip())
    if not instruction:
        raise click.BadParameter("task name must not be empty")
    os.makedirs(output, exist_ok=True)

    client = OpenVLAClient(policy_url, request_timeout)
    client.health()
    dt = 1.0 / frequency
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

    with SharedMemoryManager() as shm_manager:
        if teleop_device == "keyboard":
            teleop = KeyboardTeleop(fine_scale=0.1, fast_scale=0.5)
        elif teleop_device == "spacemouse":
            teleop = Spacemouse(shm_manager=shm_manager, deadzone=0.1)
        else:
            teleop = GamepadTeleop()

        with teleop as controller, KeystrokeCounter() as keys, Arx5Env(
            output_dir=output,
            robots_config=robots_config,
            frequency=frequency,
            obs_image_resolution=OBS_RESOLUTION,
            max_obs_buffer_size=120,
            obs_float32=True,
            camera_reorder=parse_camera_reorder(camera_reorder),
            init_joints=init_joints,
            enable_multi_cam_vis=False,
            camera_obs_latency=0.091,
            camera_obs_horizon=1,
            robot_obs_horizon=1,
            no_mirror=no_mirror,
            gripper_mask=gripper_mask,
            max_pos_speed=2.0,
            max_rot_speed=6.0,
            shm_manager=shm_manager,
        ) as env:
            cv2.setNumThreads(2)
            window = "openvla-arx5"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 960, 720)
            print("Ready. C: OpenVLA, S: teleop, Q: quit")
            should_quit = False

            while not should_quit:
                state = env.get_robot_state()[0]
                target_pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).copy()
                gripper = float(state["gripper_position"])
                human_start = time.monotonic()
                human_step = 0
                start_policy = False

                while not should_quit and not start_policy:
                    cycle_end = human_start + (human_step + 1) * dt
                    obs = env.get_obs()
                    show_preview(window, obs["camera0_rgb"][-1], "TELEOP", instruction)
                    for key in keys.get_press_events():
                        if key == KeyCode(char="q"):
                            should_quit = True
                        elif key == KeyCode(char="c"):
                            start_policy = True
                    if should_quit or start_policy:
                        break

                    motion = controller.get_motion_state_transformed()
                    target_pose[:3] += motion[:3] * (0.5 / frequency) * 0.4
                    delta_rotation = st.Rotation.from_euler(
                        "xyz", motion[3:] * (1.5 / frequency) * 0.8
                    )
                    target_pose[3:] = (
                        st.Rotation.from_rotvec(target_pose[3:]) * delta_rotation
                    ).as_rotvec()

                    reset_requested = (
                        controller.consume_reset_request()
                        if hasattr(controller, "consume_reset_request")
                        else controller.is_button_pressed(0) and controller.is_button_pressed(1)
                    )
                    if reset_requested:
                        env.robots[0].reset_to_home()
                        state = env.get_robot_state()[0]
                        target_pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).copy()
                        gripper = float(state["gripper_position"])
                    elif controller.is_button_pressed(0):
                        gripper -= 0.04 / frequency
                    elif controller.is_button_pressed(1):
                        gripper += 0.04 / frequency
                    gripper = float(np.clip(gripper, 0.0, MAX_GRIPPER_WIDTH))
                    submit_target(env, np.concatenate([target_pose, [gripper]]), dt)
                    precise_wait(cycle_end)
                    human_step += 1

                if should_quit:
                    break

                start_time = time.time()
                env.start_episode(start_time)
                state = env.get_robot_state()[0]
                episode_start_pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).copy()
                policy_step = 0
                stop_policy = False
                try:
                    while not stop_policy and time.time() - start_time < max_duration:
                        step_start = time.time()
                        obs = env.get_obs()
                        image = to_uint8_rgb(obs["camera0_rgb"][-1])
                        show_preview(window, image, "OPENVLA", instruction)
                        if float(image.mean()) < min_image_mean:
                            raise RuntimeError(f"camera image is too dark (mean={image.mean():.2f})")
                        for key in keys.get_press_events():
                            if key == KeyCode(char="q"):
                                should_quit = True
                                stop_policy = True
                            elif key == KeyCode(char="s"):
                                stop_policy = True
                        if stop_policy:
                            continue

                        action = client.predict(image, instruction, unnorm_key)
                        for key in keys.get_press_events():
                            if key == KeyCode(char="q"):
                                should_quit = True
                                stop_policy = True
                            elif key == KeyCode(char="s"):
                                stop_policy = True
                        if stop_policy:
                            continue
                        state = env.get_robot_state()[0]
                        current_pose = np.asarray(state["ActualTCPPose"], dtype=np.float64)
                        target = openvla_action_to_arx5_target(
                            action,
                            current_pose,
                            episode_start_pose,
                            max_position_delta=max_position_delta,
                            max_rotation_delta=max_rotation_delta,
                            episode_radius=episode_radius,
                            min_z_below_start=min_z_below_start,
                        )
                        submit_target(env, target, dt)
                        print(
                            f"step={policy_step} latency={time.time() - step_start:.3f}s "
                            f"delta={np.array2string(action, precision=3)}"
                        )
                        policy_step += 1
                        precise_wait(step_start + dt, time_func=time.time)
                except (RuntimeError, ValueError) as exc:
                    print(f"OpenVLA episode stopped: {exc}")
                finally:
                    env.end_episode()

            if should_quit:
                print("Returning robot arm to home. Please wait...")
                env.robots[0].reset_to_home()
                print("Robot arm returned to home.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
