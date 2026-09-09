import copy
from typing import Optional, Callable, Dict
import enum
import time
import cv2
import numpy as np
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from modules.timestamp_accumulator import get_accumulate_timestamp_idxs
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from peripherals.video_recorder import VideoRecorder
from utils.usb_util import reset_usb_device
import os


class Command(enum.Enum):
    RESTART_PUT = 0
    START_RECORDING = 1
    STOP_RECORDING = 2


class UvcCamera(mp.Process):
    """
    Call umi.common.usb_util.reset_all_elgato_devices
    if you are using Elgato capture cards.
    Required to workaround firmware bugs.
    """

    MAX_PATH_LENGTH = 4096  # linux path has a limit of 4096 bytes

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        # v4l2 device file path
        # e.g. /dev/video0
        # or /dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB320216MTR-video-index0
        dev_video_path,
        resolution=(1280, 720),
        capture_fps=60,
        capture_fourcc=None,
        put_fps=None,
        put_downsample=True,
        get_max_k=30,
        receive_latency=0.0,
        cap_buffer_size=1,
        num_threads=2,
        transform: Optional[Callable[[Dict], Dict]] = None,
        vis_transform: Optional[Callable[[Dict], Dict]] = None,
        recording_transform: Optional[Callable[[Dict], Dict]] = None,
        video_recorder: Optional[VideoRecorder] = None,
        verbose=False,
    ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = {"color": np.empty(shape=shape + (3,), dtype=np.uint8)}
        examples["camera_capture_timestamp"] = 0.0
        examples["camera_receive_timestamp"] = 0.0
        examples["timestamp"] = 0.0
        examples["step_idx"] = 0
        print(f"{examples['color'].shape=}")

        vis_examples = copy.deepcopy(examples)
        ### WTF why this doesn't work?
        # tf_example = {'color': np.empty(shape=shape+(3,), dtype=np.uint8)}
        # vis_examples = examples.copy()
        # vis_shape = vis_transform(tf_example)["color"].shape
        # print(f"{vis_shape=}")
        if vis_transform is not None:
            vis_shape = (720, 960, 3)
        else:
            vis_shape = shape + (3,)
        vis_examples["color"] = np.empty(shape=vis_shape, dtype=np.uint8)
        # print(f"{vis_examples['color'].shape=}, {examples['color'].shape=}")
        # print(f"{vis_examples['color'].flags=}, {examples['color'].flags=}")

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=vis_examples,
            get_max_k=1,
            get_time_budget=0.5,
            put_desired_frequency=capture_fps,
        )

   
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.5,
            put_desired_frequency=put_fps,
        )

        # create command queue
        examples = {
            "cmd": Command.RESTART_PUT.value,
            "put_start_time": 0.0,
            "video_path": np.array("a" * self.MAX_PATH_LENGTH),
            "recording_start_time": 0.0,
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=examples, buffer_size=128
        )

        # create video recorder
        if video_recorder is None:
            # default to nvenc GPU encoder
            video_recorder = VideoRecorder.create_hevc_nvenc(
                shm_manager=shm_manager,
                fps=capture_fps,
                input_pix_fmt="bgr24",
                bit_rate=6000 * 1000,
            )
        assert video_recorder.fps == capture_fps

        # copied variables
        self.shm_manager = shm_manager
        self.dev_video_path = dev_video_path
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.capture_fourcc = capture_fourcc
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.receive_latency = receive_latency
        self.cap_buffer_size = cap_buffer_size
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None
        self.num_threads = num_threads

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        shape = self.resolution[::-1]
        data_example = np.empty(shape=shape + (3,), dtype=np.uint8)
        self.video_recorder.start(
            shm_manager=self.shm_manager, data_example=data_example
        )
        # must start video recorder first to create share memories
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.video_recorder.stop()
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self, timeout=75.0):
        if not self.ready_event.wait(timeout=timeout):
            if not self.is_alive():
                raise RuntimeError(
                    f"UvcCamera process for {self.dev_video_path} exited before "
                    "the first frame was received. Check the child-process "
                    "traceback above."
                )
            raise TimeoutError(
                f"Timed out after {timeout:g}s waiting for the first frame from "
                f"{self.dev_video_path}"
            )
        self.video_recorder.start_wait()

    def end_wait(self):
        self.join()
        self.video_recorder.end_wait()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    @property
    def frame_count(self):
        return self.ring_buffer.count

    def wait_for_frames(self, count, timeout=10.0):
        if count > self.ring_buffer.get_max_k:
            raise ValueError(
                f"Camera {self.dev_video_path} needs {count} history frames, "
                f"but its ring buffer supports at most "
                f"{self.ring_buffer.get_max_k}"
            )

        deadline = time.monotonic() + timeout
        while self.frame_count < count:
            if self.exitcode is not None:
                raise RuntimeError(
                    f"Camera process for {self.dev_video_path} exited after "
                    f"producing {self.frame_count} of {count} required frames"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out after {timeout:g}s waiting for "
                    f"{self.dev_video_path} to produce {count} frames; "
                    f"received {self.frame_count}"
                )
            time.sleep(min(0.01, remaining))

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    def start_recording(self, video_path: str, start_time: float = -1):
        path_len = len(video_path.encode("utf-8"))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError("video_path too long.")
        self.command_queue.put(
            {
                "cmd": Command.START_RECORDING.value,
                "video_path": video_path,
                "recording_start_time": start_time,
            }
        )

    def stop_recording(self):
        self.command_queue.put({"cmd": Command.STOP_RECORDING.value})

    def restart_put(self, start_time):
        self.command_queue.put(
            {"cmd": Command.RESTART_PUT.value, "put_start_time": start_time}
        )

    # ========= interval API ===========
    def run(self):

        pid = os.getpid()
        os.sched_setaffinity(pid, [5])
        # limit threads
        threadpool_limits(self.num_threads)
        cv2.setNumThreads(self.num_threads)

        # open VideoCapture
        cap = cv2.VideoCapture(self.dev_video_path, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.dev_video_path}")
            # set resolution and fps
            w, h = self.resolution
            fps = self.capture_fps
            if self.capture_fourcc is not None:
                if len(self.capture_fourcc) != 4:
                    raise ValueError("capture_fourcc must contain exactly 4 characters")
                fourcc = cv2.VideoWriter_fourcc(*self.capture_fourcc)
                if not cap.set(cv2.CAP_PROP_FOURCC, fourcc):
                    raise RuntimeError(
                        f"Camera rejected pixel format {self.capture_fourcc}"
                    )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            # set fps
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cap_buffer_size)
            cap.set(cv2.CAP_PROP_FPS, fps)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            actual_fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
            actual_fourcc = "".join(
                chr((actual_fourcc_value >> (8 * i)) & 0xFF) for i in range(4)
            )
            print(
                f"UvcCamera {self.dev_video_path} requested {w}x{h}@{fps}, "
                f"format {self.capture_fourcc or 'driver-default'}; driver reports "
                f"{actual_w}x{actual_h}@{actual_fps:g} {actual_fourcc}, "
                f"buffer size {self.cap_buffer_size}"
            )

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            consecutive_frame_failures = 0
            # Some inexpensive MJPEG capture cards emit a burst of empty or
            # truncated buffers while the HDMI link is synchronizing.  ffplay
            # drops these and keeps reading, so do the same instead of killing
            # the camera worker after only a handful of bad buffers.
            max_consecutive_frame_failures = 60
            t_start = time.time()
            while not self.stop_event.is_set():
                ts = time.time()
                ret = cap.grab()
                if not ret:
                    consecutive_frame_failures += 1
                    if consecutive_frame_failures <= max_consecutive_frame_failures:
                        print(
                            f"Warning: failed to grab a V4L2 frame from "
                            f"{self.dev_video_path}; retrying "
                            f"({consecutive_frame_failures}/"
                            f"{max_consecutive_frame_failures})"
                        )
                        time.sleep(0.2)
                        continue
                    raise RuntimeError(
                        f"V4L2 failed {consecutive_frame_failures} consecutive "
                        f"times while waiting for frames from "
                        f"{self.dev_video_path} using "
                        f"{actual_w}x{actual_h}@{actual_fps:g} {actual_fourcc}. "
                        "The capture device is delivering no usable buffers. "
                        "Check for zero-byte/corrupted MJPEG frames with ffplay "
                        "or v4l2-ctl, then inspect the HDMI source and USB link."
                    )

                # OpenCV 5 no longer reliably accepts a preallocated ndarray in
                # VideoCapture.retrieve(). Retrieve normally, then copy into the
                # recorder's shared buffer only while recording.
                ret, frame = cap.retrieve()
                t_recv = time.time()
                if not ret or frame is None:
                    consecutive_frame_failures += 1
                    if consecutive_frame_failures <= max_consecutive_frame_failures:
                        print(
                            f"Warning: received an undecodable "
                            f"{actual_fourcc} frame from {self.dev_video_path}; "
                            f"retrying ({consecutive_frame_failures}/"
                            f"{max_consecutive_frame_failures})"
                        )
                        continue
                    raise RuntimeError(
                        f"Failed to decode {consecutive_frame_failures} "
                        f"consecutive frames from {self.dev_video_path} in "
                        f"{actual_fourcc} mode"
                    )
                consecutive_frame_failures = 0
                expected_shape = (h, w, 3)
                if frame.shape != expected_shape:
                    raise RuntimeError(
                        f"Camera returned frame shape {frame.shape}, but "
                        f"{expected_shape} was requested. The capture device does "
                        "not support the configured resolution/FPS/pixel-format "
                        "combination. Inspect it with `v4l2-ctl "
                        f"--list-formats-ext -d {self.dev_video_path}` and choose "
                        "a supported mode in modules/arx5_env.py."
                    )
                mt_cap = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                t_cap = mt_cap - time.monotonic() + time.time()
                t_cal = t_recv - self.receive_latency  # calibrated latency

                # record frame
                if self.video_recorder.is_ready():
                    record_buffer = self.video_recorder.get_img_buffer()
                    if record_buffer.shape != frame.shape:
                        raise RuntimeError(
                            "Camera frame shape changed from "
                            f"{record_buffer.shape} to {frame.shape}"
                        )
                    np.copyto(record_buffer, frame)
                    self.video_recorder.write_img_buffer(
                        record_buffer, frame_time=t_cal
                    )

                data = dict()
                data["camera_receive_timestamp"] = t_cap
                data["camera_capture_timestamp"] = t_recv
                data["color"] = frame

                # apply transform
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[t_cal],
                        start_time=put_start_time,
                        dt=1 / self.put_fps,
                        # this is non in first iteration
                        # and then replaced with a concrete number
                        next_global_idx=put_idx,
                        # continue to pump frames even if not started.
                        # start_time is simply used to align timestamps.
                        allow_negative=True,
                    )

                    for step_idx in global_idxs:
                        put_data["step_idx"] = step_idx
                        put_data["timestamp"] = t_cal
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((t_cal - put_start_time) * self.put_fps)
                    put_data["step_idx"] = step_idx
                    put_data["timestamp"] = t_cal
                    self.ring_buffer.put(put_data, wait=False)

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()

                # put to vis
                vis_data = data
                if self.vis_transform == self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                self.vis_ring_buffer.put(vis_data, wait=False)

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f"[UvcCamera {self.dev_video_path}] FPS {frequency}")

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command["cmd"]
                    if cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command["put_start_time"]
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command["video_path"])
                        start_time = command["recording_start_time"]
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start_recording(
                            video_path, start_time=start_time
                        )
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop_recording()

                iter_idx += 1
        finally:
            self.video_recorder.stop()
            # When everything done, release the capture
            cap.release()
