import multiprocessing as mp
import enum
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, cast
import numpy as np
from utils.other_util import precise_wait
from shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

from modules.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import time
from modules.arx5_zmq_client import Arx5Client
import numpy.typing as npt

from multiprocessing import Value
import ctypes


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    RESET_TO_HOME = 3
    ADD_WAYPOINT = 4
    UPDATE_TRAJECTORY = 5
    JOINT_SCHEDULE_WAYPOINT = 6
    JOINT_DRAG = 7
    JOINT_HOLD = 8
    JOINT_SET_TRAJECTORY = 9


# A 16-step, 5 Hz policy horizon becomes roughly 300 points at 100 Hz.
MAX_JOINT_WAYPOINTS = 512


class Arx5Controller(mp.Process):
    pass

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip: str,
        robot_port: int,
        launch_timeout: float = 10,
        frequency: float = 100,
        get_max_k: Optional[int] = None,
        verbose: bool = False,
        receive_latency: float = 0.0,
        control_mode: str = "pose",
    ):
        super().__init__(name="Arx5Controller")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        if control_mode not in ("pose", "joint"):
            raise ValueError(f"Unsupported ARX5 control mode: {control_mode}")
        self.control_mode = control_mode

        example = {
            "cmd": Command.SERVOL.value,
            "target_pose": np.zeros((6,), dtype=np.float64),
            "target_joint_pos": np.zeros((6,), dtype=np.float64),
            "gripper_pos": 0.0,
            "duration": 0.0,
            "target_time": 0.0,
            "joint_traj": np.zeros((MAX_JOINT_WAYPOINTS, 6), dtype=np.float64),
            "joint_gripper_traj": np.zeros((MAX_JOINT_WAYPOINTS,), dtype=np.float64),
            "joint_target_times": np.zeros((MAX_JOINT_WAYPOINTS,), dtype=np.float64),
            "joint_traj_len": 0,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=example, buffer_size=256
        )
        self.verbose = verbose

        # build ring buffer
        receive_keys = [
            ("ActualTCPPose", "tcp_pose"),
            ("ActualQ", "joint_pos"),
            ("ActualQd", "joint_vel"),
            ("gripper_position", "gripper_pos"),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if "joint" in func_name:
                example[key] = np.zeros(6)
            elif "tcp_pose" in func_name:
                example[key] = np.zeros(6)
        example["gripper_position"] = 0.0

        example["robot_receive_timestamp"] = time.time()
        example["robot_timestamp"] = time.time()

        if get_max_k is None:
            get_max_k = int(frequency * 5)
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.5,
            put_desired_frequency=frequency,
        )
        self.launch_timeout = launch_timeout
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
        self.frequency = frequency
        self.receive_latency = receive_latency
        self.mode_state = Value(ctypes.c_int, 0)  # 0=hold, 1=drag, -1=stopped

        # Will be initialized in the subprocess
        self.robot_client: Arx5Client
        self.reset_success = Value(ctypes.c_bool, False)

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Arx5Controller] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {"cmd": Command.STOP.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()
        if self.verbose:
            print(f"[Arx5Controller] Controller process terminated at {self.pid}")

    def start_wait(self):
        print(f"[Arx5Controller] Waiting for controller process to be ready")
        print(f"{self.launch_timeout=}")
        deadline = time.monotonic() + self.launch_timeout
        while not self.is_ready:
            if not self.is_alive():
                raise RuntimeError(
                    f"ARX5 controller exited during startup (exitcode={self.exitcode}). "
                    "Check the controller traceback and robot ZMQ server."
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"ARX5 controller not ready after {self.launch_timeout}s: "
                    f"{self.ring_buffer.count} states received from "
                    f"{self.robot_ip}:{self.robot_port}. Check the robot server."
                )
            time.sleep(min(0.05, remaining))

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return (
            self.ready_event.is_set()
            and self.is_alive()
            and self.ring_buffer.count >= 2
        )

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= command methods ============
    def servoL(self, pose: npt.NDArray[np.float64], gripper_pos: float, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert duration >= (1 / self.frequency)
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.SERVOL.value,
            "target_pose": pose,
            "gripper_pos": gripper_pos,
            "duration": duration,
        }
        self.input_queue.put(message)

    def schedule_waypoint(
        self, pose: npt.NDArray[np.float64], gripper_pos: float, target_time: float
    ):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pose": pose,
            "gripper_pos": gripper_pos,
            "target_time": target_time,
        }
        self.input_queue.put(message)

    def schedule_joint_waypoint(
        self, joint_pos: npt.NDArray[np.float64], gripper_pos: float, target_time: float
    ):
        if self.control_mode != "joint":
            raise RuntimeError("Joint waypoint sent to a pose controller")
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        if joint_pos.shape != (6,):
            raise ValueError(f"Expected six joint positions, got {joint_pos.shape}")
        self.input_queue.put({
            "cmd": Command.JOINT_SCHEDULE_WAYPOINT.value,
            "target_joint_pos": joint_pos,
            "gripper_pos": float(gripper_pos),
            "target_time": float(target_time),
        })

    def set_joint_trajectory(
        self,
        joint_pos: npt.NDArray[np.float64],
        gripper_pos: npt.NDArray[np.float64],
        target_times: npt.NDArray[np.float64],
    ):
        """Queue a complete future joint trajectory for one SDK call."""
        if self.control_mode != "joint":
            raise RuntimeError("Joint trajectory sent to a pose controller")
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        gripper_pos = np.asarray(gripper_pos, dtype=np.float64)
        target_times = np.asarray(target_times, dtype=np.float64)
        if joint_pos.ndim != 2 or joint_pos.shape[1] != 6:
            raise ValueError(f"Expected a (N, 6) joint trajectory, got {joint_pos.shape}")
        if joint_pos.shape[0] == 0 or joint_pos.shape[0] > MAX_JOINT_WAYPOINTS:
            raise ValueError(
                f"Expected 1..{MAX_JOINT_WAYPOINTS} joint waypoints, got {joint_pos.shape[0]}"
            )
        if gripper_pos.shape != (joint_pos.shape[0],):
            raise ValueError(f"Expected gripper shape {(joint_pos.shape[0],)}, got {gripper_pos.shape}")
        if target_times.shape != gripper_pos.shape:
            raise ValueError(f"Expected target time shape {gripper_pos.shape}, got {target_times.shape}")
        if (
            not np.isfinite(joint_pos).all()
            or not np.isfinite(gripper_pos).all()
            or not np.isfinite(target_times).all()
            or np.any(np.diff(target_times) <= 0)
        ):
            raise ValueError("Joint trajectory values must be finite and time-ordered")
        padded_joint = np.zeros((MAX_JOINT_WAYPOINTS, 6), dtype=np.float64)
        padded_gripper = np.zeros((MAX_JOINT_WAYPOINTS,), dtype=np.float64)
        padded_times = np.zeros((MAX_JOINT_WAYPOINTS,), dtype=np.float64)
        count = joint_pos.shape[0]
        padded_joint[:count] = joint_pos
        padded_gripper[:count] = gripper_pos
        padded_times[:count] = target_times
        self.input_queue.put({
            "cmd": Command.JOINT_SET_TRAJECTORY.value,
            "joint_traj": padded_joint,
            "joint_gripper_traj": padded_gripper,
            "joint_target_times": padded_times,
            "joint_traj_len": count,
        })

    def drag(self, wait=True, timeout=2.0):
        if self.control_mode != "joint":
            raise RuntimeError("Drag mode is only available for a joint controller")
        self.input_queue.put({"cmd": Command.JOINT_DRAG.value})
        if wait:
            self._wait_for_mode(1, timeout)

    def hold(self, wait=True, timeout=2.0):
        if self.control_mode != "joint":
            raise RuntimeError("Hold mode is only available for a joint controller")
        self.input_queue.put({"cmd": Command.JOINT_HOLD.value})
        if wait:
            self._wait_for_mode(0, timeout)

    def _wait_for_mode(self, expected, timeout):
        deadline = time.monotonic() + timeout
        while self.mode_state.value != expected:
            if not self.is_alive():
                raise RuntimeError(
                    f"ARX5 controller exited while switching joint mode "
                    f"(exitcode={self.exitcode})"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for joint controller mode {expected}; "
                    f"current={self.mode_state.value}"
                )
            time.sleep(0.005)

    def add_waypoint(
        self, pose: npt.NDArray[np.float64], gripper_pos: float, target_time: float
    ):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.ADD_WAYPOINT.value,
            "target_pose": pose,
            "gripper_pos": gripper_pos,
            "target_time": target_time,
        }
        self.input_queue.put(message)

    def update_trajectory(self):
        message = {
            "cmd": Command.UPDATE_TRAJECTORY.value,
        }
        self.input_queue.put(message)

    def reset_to_home(self):
        self.reset_success.value = False
        message = {"cmd": Command.RESET_TO_HOME.value}
        self.input_queue.put(message)
        while not self.reset_success.value:
            print("waiting for reset")
            time.sleep(0.1)

    # ========= main loop in process ============
    def run(self):
        self.robot_client = Arx5Client(self.robot_ip, self.robot_port)
        if self.control_mode == "joint":
            self._run_joint()
            return
        self.robot_client.reset_to_home()
        # self.robot_client.set_gain()
        time.sleep(1)
        # self.robot_client.set_to_damping()
        # gain = self.robot_client.get_gain()
        # gain['kd'] = gain['kd'] * 0.2
        gain = self.robot_client.get_gain()
        gain["kp"] = np.array([300, 300, 400, 80, 50, 30])
        gain["kd"][3] = 0.5 # Some motors will shake if kd is high
        self.robot_client.set_gain(gain)
        np.set_printoptions(precision=3, suppress=True)

        self.waypoint_buffer = []

        try:

            dt = 1 / self.frequency
            self.robot_client.get_state()
            curr_pose = self.robot_client.tcp_command_pose
            curr_gripper_pos = self.robot_client.gripper_command_pos
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=np.array([curr_t]), poses=np.array([curr_pose])
            )
            gripper_pos_interp = PoseTrajectoryInterpolator(
                times=np.array([curr_t]),
                poses=np.array([[curr_gripper_pos, 0, 0, 0, 0, 0]]),
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                t_now = time.monotonic()
                pose_cmd = pose_interp(t_now)
                gripper_cmd = float(gripper_pos_interp(t_now)[0])

                self.robot_client.set_tcp_pose(pose_cmd, gripper_cmd)
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(self.robot_client, func_name)
                t_recv = time.time()
                state["robot_receive_timestamp"] = t_recv
                state["robot_timestamp"] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # if self.verbose:
                #     print(f"Current: {state['ActualTCPPose']} target: {pose_cmd}, gripper: {state['gripper_position']:.3f}/{gripper_cmd:.3f}")

                try:
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands["cmd"])
                except Empty:
                    commands = {}
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command["cmd"]

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command["target_pose"]
                        duration = float(command["duration"])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        gripper_pos_interp = gripper_pos_interp.drive_to_waypoint(
                            pose=[command["gripper_pos"], 0, 0, 0, 0, 0],
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print(
                                f"[Arx5Controller] New pose target: {target_pose} duration {duration:.3f}s"
                            )
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command["target_pose"]
                        target_time = float(command["target_time"])
                        target_time = time.monotonic() - time.time() + target_time
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=t_now,
                            last_waypoint_time=last_waypoint_time,
                        )
                        gripper_pos_interp = gripper_pos_interp.schedule_waypoint(
                            pose=[command["gripper_pos"], 0, 0, 0, 0, 0],
                            time=target_time,
                            curr_time=t_now,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_time

                    elif cmd == Command.RESET_TO_HOME.value:
                        self.robot_client.reset_to_home()
                        self.robot_client.get_state()

                        self.ring_buffer.clear()
                        state = dict()
                        for key, func_name in self.receive_keys:
                            state[key] = getattr(self.robot_client, func_name)
                        t_recv = time.time()
                        state["robot_receive_timestamp"] = t_recv
                        state["robot_timestamp"] = t_recv - self.receive_latency
                        self.ring_buffer.put(state)
                        if self.verbose:
                            print(f"[Arx5Controller] Reset to home")

                        curr_pose = self.robot_client.tcp_command_pose
                        curr_gripper_pos = self.robot_client.gripper_command_pos
                        curr_t = time.monotonic()
                        last_waypoint_time = curr_t
                        pose_interp = PoseTrajectoryInterpolator(
                            times=np.array([curr_t]), poses=np.array([curr_pose])
                        )
                        gripper_pos_interp = PoseTrajectoryInterpolator(
                            times=np.array([curr_t]),
                            poses=np.array([[curr_gripper_pos, 0, 0, 0, 0, 0]]),
                        )
                        self.reset_success.value = True
                        gain = self.robot_client.get_gain()
                        gain["kp"] = np.array([300, 300, 400, 80, 50, 30])
                        gain["kd"][3] = 0.5
                        self.robot_client.set_gain(gain)

                    elif cmd == Command.ADD_WAYPOINT.value:
                        if len(self.waypoint_buffer) > 0:
                            last_waypoint_time = self.waypoint_buffer[-1]["target_time"]
                            if command["target_time"] <= last_waypoint_time:
                                print(
                                    f"[Arx5Controller] Waypoint time {command['target_time']:.3f} is not in the future, skipping"
                                )
                                continue
                        self.waypoint_buffer.append(command)
                    elif cmd == Command.UPDATE_TRAJECTORY.value:
                        if len(self.waypoint_buffer) == 0:
                            print(
                                f"[Arx5Controller] No new waypoints to update trajectory"
                            )
                            continue
                        start_time = time.monotonic()
                        # Dynamic latency matching
                        matching_dt = 0.01
                        pose_samples = np.zeros((3, 6))
                        pose_samples[0, :] = pose_interp(t_now - matching_dt)
                        pose_samples[1, :] = pose_interp(t_now)
                        pose_samples[2, :] = pose_interp(t_now + matching_dt)

                        input_poses = np.array(
                            [cmd["target_pose"] for cmd in self.waypoint_buffer]
                        )  # (N, 6)
                        input_times = np.array(
                            [cmd["target_time"] for cmd in self.waypoint_buffer]
                        )
                        input_gripper_pos = np.array(
                            [cmd["gripper_pos"] for cmd in self.waypoint_buffer]
                        )
                        input_pose_interp = PoseTrajectoryInterpolator(
                            times=input_times - input_times[0], poses=input_poses
                        )

                        latency_precision = 0.02
                        max_latency = 1.2
                        smoothing_time = 0.4
                        errors = []
                        error_weights = np.array(
                            [1, 1, 1, 0.1, 0.1, 0.1]
                        )  # x, y, z, rx, ry, rz
                        for latency in np.arange(
                            matching_dt, max_latency, latency_precision
                        ):
                            input_pose_samples = np.zeros((3, 6))
                            input_pose_samples[0, :] = input_pose_interp(
                                latency - matching_dt
                            )
                            input_pose_samples[1, :] = input_pose_interp(latency)
                            input_pose_samples[2, :] = input_pose_interp(
                                latency + matching_dt
                            )
                            error = np.sum(
                                np.abs(input_pose_samples - pose_samples)
                                * error_weights
                            )
                            errors.append(error)
                        errors = np.array(errors)
                        best_latency = np.arange(
                            matching_dt, max_latency, latency_precision
                        )[np.argmin(errors)]

                        smoothened_input_poses = input_poses
                        new_times = input_times - input_times[0] + t_now - best_latency
                        for i in range((smoothened_input_poses.shape[0])):
                            if new_times[i] < t_now:
                                smoothened_input_poses[i] = pose_interp(new_times[i])
                            elif t_now <= new_times[i] < t_now + smoothing_time:
                                alpha = (new_times[i] - t_now) / smoothing_time
                                smoothened_input_poses[i] = (1 - alpha) * pose_interp(
                                    new_times[i]
                                ) + alpha * input_poses[i]
                            else:
                                smoothened_input_poses[i] = input_poses[i]

                        pose_interp = PoseTrajectoryInterpolator(
                            times=new_times, poses=smoothened_input_poses
                        )
                        extended_gripper_pos = np.zeros_like(input_poses)
                        extended_gripper_pos[:, 0] = input_gripper_pos
                        gripper_pos_interp = PoseTrajectoryInterpolator(
                            times=new_times, poses=extended_gripper_pos
                        )
                        if self.verbose:
                            print(
                                f"[Arx5Controller] latency: {best_latency:.3f}s, error: {errors.min():.3f}, time: {time.monotonic() - start_time:.3f}s"
                            )
                        # clear buffer
                        self.waypoint_buffer = []
                    else:
                        keep_running = False
                        print(f"[Arx5Controller] Unknown command {cmd}")
                        break
                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)
                # Pose interpolation requires at least two received states.
                if not self.ready_event.is_set() and self.ring_buffer.count >= 2:
                    self.ready_event.set()
                    if self.verbose:
                        print("[Arx5Controller] Controller process is ready")
                iter_idx += 1
                # if self.verbose:
                #     print(f"[Arx5Controller] Actual frequency {1/(time.monotonic() - t_now)} Hz")

        finally:
            self.ready_event.clear()
            print("[Arx5Controller] Setting robot to damping")
            self.robot_client.set_to_damping()
            del self.robot_client
            if self.verbose:
                print("[Arx5Controller] Controller process terminated")

    def _run_joint(self):
        """Run joint control while the SDK owns the continuous interpolation.

        Policy chunks are sent as one timed trajectory.  Streaming a single
        endpoint every cycle would repeatedly clear the SDK trajectory and can
        leave the motors waiting for the next ZMQ round trip.
        """
        np.set_printoptions(precision=3, suppress=True)
        dt = 1 / self.frequency
        try:
            self.robot_client.get_state()
            missing = {"armed", "mode"}.difference(self.robot_client.latest_state)
            if missing:
                raise RuntimeError(
                    "Joint control requires communication/zmq_joint_server.py; "
                    f"GET_STATE is missing {sorted(missing)}. Stop the Cartesian "
                    "zmq_server.py and restart the joint server on the same port."
                )
            self.robot_client.reset_to_home()
            self.robot_client.get_state()
        except Exception:
            try:
                self.robot_client.set_to_damping()
            except Exception:
                pass
            raise
        t_start = time.monotonic()
        iter_idx = 0
        manual_drag = False
        try:
            keep_running = True
            while keep_running:
                # Refresh feedback without replacing the SDK's future command.
                self.robot_client.get_state()
                state = {
                    key: getattr(self.robot_client, func_name)
                    for key, func_name in self.receive_keys
                }
                t_recv = time.time()
                state["robot_receive_timestamp"] = t_recv
                state["robot_timestamp"] = t_recv - self.receive_latency
                self.ring_buffer.put(state)
                try:
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands["cmd"])
                except Empty:
                    commands, n_cmd = {}, 0
                for i in range(n_cmd):
                    command = {key: value[i] for key, value in commands.items()}
                    cmd = command["cmd"]
                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    if cmd == Command.JOINT_DRAG.value:
                        self.robot_client.drag()
                        manual_drag = True
                        self.mode_state.value = 1
                    elif cmd == Command.JOINT_HOLD.value:
                        self.robot_client.hold()
                        manual_drag = False
                        self.mode_state.value = 0
                    elif cmd == Command.JOINT_SET_TRAJECTORY.value:
                        if manual_drag:
                            continue
                        count = int(command["joint_traj_len"])
                        if count <= 0 or count > MAX_JOINT_WAYPOINTS:
                            raise ValueError(f"Invalid queued joint trajectory length: {count}")
                        joint_pos = np.asarray(command["joint_traj"][:count], dtype=np.float64)
                        gripper_pos = np.asarray(
                            command["joint_gripper_traj"][:count], dtype=np.float64
                        )
                        target_times = np.asarray(
                            command["joint_target_times"][:count], dtype=np.float64
                        )
                        # The server drops points that are already due.  Avoid
                        # making an empty request when inference was over budget.
                        future = target_times > time.time()
                        if np.any(future):
                            self.robot_client.set_joint_traj(
                                joint_pos=joint_pos,
                                gripper_pos=gripper_pos,
                                target_times=target_times,
                            )
                            if self.verbose:
                                print(
                                    f"[Arx5Controller] Submitted {count} joint waypoints "
                                    "to the SDK trajectory interpolator"
                                )
                    elif cmd == Command.JOINT_SCHEDULE_WAYPOINT.value:
                        if manual_drag or float(command["target_time"]) <= time.time():
                            continue
                        self.robot_client.set_joint_traj(
                            joint_pos=np.asarray(command["target_joint_pos"], dtype=np.float64)[None],
                            gripper_pos=np.asarray([command["gripper_pos"]], dtype=np.float64),
                            target_times=np.asarray([command["target_time"]], dtype=np.float64),
                        )
                    elif cmd == Command.RESET_TO_HOME.value:
                        self.robot_client.reset_to_home()
                        self.robot_client.get_state()
                        self.ring_buffer.clear()
                        self.reset_success.value = True
                        manual_drag = False
                        self.mode_state.value = 0
                    else:
                        print(f"[Arx5Controller] Unsupported joint command {cmd}")
                        keep_running = False
                        break
                precise_wait(t_start + (iter_idx + 1) * dt, time_func=time.monotonic)
                if not self.ready_event.is_set() and self.ring_buffer.count >= 2:
                    self.ready_event.set()
                    if self.verbose:
                        print("[Arx5Controller] Joint controller process is ready")
                iter_idx += 1
        finally:
            self.ready_event.clear()
            self.mode_state.value = -1
            try:
                self.robot_client.set_to_damping()
            except Exception:
                pass
            del self.robot_client
            if self.verbose:
                print("[Arx5Controller] Controller process terminated")
