"""Mock and opt-in can0 backends. Importing this module never enables hardware."""

import fcntl
from pathlib import Path
import socket
import struct
import sys
import time

import numpy as np


class MockBackend:
    source = "mock"

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.q = np.array([.32, 0., .16, 0., 0., 0.])
        self.width = .02
        self.stopped = False

    def read(self):
        return dict(pose=self.fk(self.q), q=self.q.copy(), dq=np.zeros(6),
                    torque=np.zeros(6), width=self.width, gripper_speed=0., gripper_load=0.,
                    motor_received=np.full(7, self.clock()), emergency=False, motor_fault=False)

    def solve(self, pose, seed):
        return pose.copy()

    def fk(self, q):
        return q.copy()

    def check_joints(self, q):
        if np.asarray(q).shape != (6,) or not np.isfinite(q).all():
            raise ValueError("invalid joints")

    def command(self, q, width):
        self.q, self.width = q.copy(), width

    def gains(self, fraction):
        pass

    def stop(self):
        self.stopped = True

    def close(self):
        self.stop()


def load_sdk(path):
    root = Path(path).resolve()
    sys.path.insert(0, str(root / "python"))
    import arx5_interface as arx
    if Path(arx.__file__).resolve().parent != root / "python":
        raise RuntimeError("refusing installed wheel; use the patched local SDK")
    if not hasattr(arx.Arx5JointController, "is_emergency"):
        raise RuntimeError("patched SDK is required")
    config = arx.RobotConfigFactory.get_instance().get_config("X5_umi")
    solver = arx.Arx5Solver(config.urdf_path, 6, config.joint_pos_min, config.joint_pos_max,
                          config.base_link_name, config.eef_link_name, config.gravity_vector)
    return arx, config, solver


class CanFeedback:
    """Kernel-timestamped incoming feedback; local TX is not motor feedback."""

    ids = (1, 2, 4, 5, 6, 7, 8)

    def __init__(self):
        self.socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, 35, 1)  # Linux SO_TIMESTAMPNS
            self.socket.setsockopt(socket.SOL_SOCKET, 40, 1)  # Linux SO_RXQ_OVFL
            self.socket.bind(("can0",))
            self.socket.setblocking(False)
        except BaseException:
            self.socket.close()
            raise
        self.received = np.full(7, -1.)
        self.fault = False
        self.local_tx_seen = False
        self.initializing = True
        self.dm_status = {}

    def ingest(self, packet, ancillary, flags, now, wall):
        if flags & socket.MSG_DONTROUTE:
            self.local_tx_seen = True
            return
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            self.fault = True
            return
        if len(packet) != 16:
            return
        cid, dlc, data = struct.unpack("=IB3x8s", packet)
        if cid == 0 and dlc == 8 and data[0] & 15 in (5, 6, 7, 8):
            mid = data[0] & 15
            status = data[0] >> 4
            self.dm_status[mid] = status
            # SDK startup briefly disables DM motors before enabling them.
            # Real faults always latch; disabling after startup also latches.
            self.fault |= status not in (0, 1) or (status == 0 and not self.initializing)
        elif cid in (1, 2, 4) and dlc == 8 and data[0] >> 5 == 1:
            mid = cid
            self.fault |= (data[0] & 31) != 0
        else:
            return
        received = None
        for level, kind, value in ancillary:
            if level == socket.SOL_SOCKET and kind == 35:
                sec, nsec = struct.unpack("@ll", value)
                age = wall-(sec+nsec/1e9)
                if age < -.01:
                    self.fault = True
                received = now-max(0., age)
            if level == socket.SOL_SOCKET and kind == 40 and struct.unpack("=I", value)[0]:
                self.fault = True
        if received is None:
            self.fault = True
        else:
            self.received[self.ids.index(mid)] = received

    def poll(self):
        for _ in range(4096):
            try:
                packet, ancillary, flags, _ = self.socket.recvmsg(16, 128)
            except BlockingIOError:
                return
            self.ingest(packet, ancillary, flags, time.monotonic(), time.time())
        self.fault = True

    def close(self):
        self.socket.close()

    def finish_initialization(self):
        self.poll()
        if self.fault:
            raise RuntimeError("CAN fault during initialization")
        if any(self.dm_status.get(mid) != 1 for mid in (5, 6, 7, 8)):
            raise RuntimeError("not all DM motors are enabled after initialization")
        if np.max(time.monotonic()-self.received) > .1:
            raise RuntimeError("stale motor feedback after initialization")
        self.initializing = False


class HardwareBackend:
    source = "live"

    def __init__(self, sdk_path):
        self.controller = self.can = self.lock = None
        self.arx, self.config, self.solver = load_sdk(sdk_path)
        try:
            self.lock = open("/tmp/umi-arx-action-can0.lock", "a")
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.can = CanFeedback()
            until = time.monotonic()+.25
            while time.monotonic() < until:
                self.can.poll()
                if self.can.local_tx_seen:
                    raise RuntimeError("can0 already has a sender; stop the other controller first")
                time.sleep(.01)
            self.config.gripper_open_readout = -3.4
            self.config.joint_vel_max = np.full(6, .15)
            self.config.gripper_vel_max = .004
            self.config.joint_torque_max = np.array([10., 12., 10., 3., 3., 3.])
            self.config.gripper_torque_max = .25
            self.cc = self.arx.ControllerConfigFactory.get_instance().get_config("cartesian_controller", 6)
            # set_to_damping() does not disable gravity feedforward in this SDK.
            # Keep it off for the entire service, including every stop path.
            self.cc.gravity_compensation = False
            self.cc.shutdown_to_passive = True
            self.cc.default_preview_time = .02
            self.cc.over_current_cnt_max = 0
            self.controller = self.arx.Arx5JointController(self.config, self.cc, "can0")
            until = time.monotonic()+.15
            while time.monotonic() < until:
                self.can.poll()
                time.sleep(.005)
            self.can.finish_initialization()
        except BaseException:
            self.close()
            raise

    def read(self):
        self.can.poll()
        state = self.controller.get_joint_state()
        q = state.pos().copy()
        return dict(pose=self.fk(q), q=q, dq=state.vel().copy(), torque=state.torque().copy(),
                    width=state.gripper_pos, gripper_speed=state.gripper_vel,
                    gripper_load=state.gripper_torque, motor_received=self.can.received.copy(),
                    emergency=self.controller.is_emergency(), motor_fault=self.can.fault)

    def solve(self, pose, seed):
        status, q = self.solver.inverse_kinematics(pose, seed)
        if status != 0:
            raise ValueError("IK rejected: " + self.solver.get_ik_status_name(status))
        return q.copy()

    def fk(self, q):
        result = self.solver.forward_kinematics(q).copy()
        if result.shape != (6,) or not np.isfinite(result).all():
            raise RuntimeError("invalid FK result")
        return result

    def check_joints(self, q):
        if (np.asarray(q).shape != (6,) or not np.isfinite(q).all() or
                np.any(q < self.config.joint_pos_min) or np.any(q > self.config.joint_pos_max)):
            raise RuntimeError("joint limits exceeded")

    def command(self, q, width):
        self.check_joints(q)
        cmd = self.arx.JointState(6)
        cmd.pos()[:] = q
        cmd.gripper_pos = width
        self.controller.set_joint_cmd(cmd)

    def gains(self, fraction):
        self.controller.set_gain(self.arx.Gain(self.cc.default_kp*fraction, self.cc.default_kd,
                                             fraction, .2*fraction))

    def stop(self):
        if self.controller is not None:
            self.controller.set_to_damping()

    def close(self):
        try:
            self.stop()
        finally:
            self.controller = None
            if self.can is not None:
                self.can.close()
                self.can = None
            if self.lock is not None:
                self.lock.close()
                self.lock = None
