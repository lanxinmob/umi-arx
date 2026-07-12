import unittest
from types import SimpleNamespace

import numpy as np

from peripherals.gamepad_teleop import GamepadTeleop


class FakeEcodes:
    EV_KEY = 1
    EV_ABS = 3
    ABS_X = 0
    ABS_Y = 1
    ABS_Z = 2
    ABS_RX = 3
    ABS_RY = 4
    ABS_RZ = 5
    BTN_TL = 310
    BTN_TR = 311
    BTN_SELECT = 314
    BTN_START = 315


class FakeEvdev:
    ecodes = FakeEcodes


class FakeInputDevice:
    last_absinfo = None

    def __init__(self, path):
        self.path = path
        self.name = "Microsoft X-Box 360 pad"

    def capabilities(self, absinfo=True):
        FakeInputDevice.last_absinfo = absinfo
        axes = [
            FakeEcodes.ABS_X,
            FakeEcodes.ABS_Y,
            FakeEcodes.ABS_Z,
            FakeEcodes.ABS_RX,
            FakeEcodes.ABS_RY,
            FakeEcodes.ABS_RZ,
        ]
        if absinfo:
            axes = [(code, SimpleNamespace()) for code in axes]
        return {
            FakeEcodes.EV_ABS: axes,
            FakeEcodes.EV_KEY: [
                FakeEcodes.BTN_TL,
                FakeEcodes.BTN_TR,
                FakeEcodes.BTN_SELECT,
                FakeEcodes.BTN_START,
            ],
        }

    def close(self):
        pass


class FakeDiscoveryEvdev:
    ecodes = FakeEcodes
    InputDevice = FakeInputDevice

    @staticmethod
    def list_devices():
        return ["/dev/input/event19"]


class GamepadTeleopTest(unittest.TestCase):
    def setUp(self):
        self.teleop = GamepadTeleop(deadzone=0.1, scale=0.5)
        self.teleop._load_evdev = lambda: FakeEvdev
        self.teleop._connected = True
        stick = SimpleNamespace(min=-32768, max=32767, value=0)
        trigger = SimpleNamespace(min=0, max=255, value=0)
        self.teleop._axis_info = {
            FakeEcodes.ABS_X: stick,
            FakeEcodes.ABS_Y: stick,
            FakeEcodes.ABS_RX: stick,
            FakeEcodes.ABS_RY: stick,
            FakeEcodes.ABS_Z: trigger,
            FakeEcodes.ABS_RZ: trigger,
        }
        self.teleop._axis_values = {
            code: info.value for code, info in self.teleop._axis_info.items()
        }

    def test_default_motion_mapping(self):
        self.teleop._axis_values.update(
            {
                FakeEcodes.ABS_X: -32768,
                FakeEcodes.ABS_Y: -32768,
                FakeEcodes.ABS_RX: 32767,
                FakeEcodes.ABS_RY: -32768,
                FakeEcodes.ABS_Z: 0,
                FakeEcodes.ABS_RZ: 255,
            }
        )
        self.teleop._pressed_buttons.add(FakeEcodes.BTN_TL)

        np.testing.assert_allclose(
            self.teleop.get_motion_state_transformed(),
            [0.5, 0.5, 0.5, 0.5, -0.5, 0.5],
            atol=1e-4,
        )

    def test_deadzone_and_disconnect_output_zero(self):
        self.teleop._axis_values[FakeEcodes.ABS_X] = 1000
        np.testing.assert_array_equal(
            self.teleop.get_motion_state_transformed(), np.zeros(6)
        )

        self.teleop._connected = False
        self.teleop._axis_values[FakeEcodes.ABS_X] = 32767
        np.testing.assert_array_equal(
            self.teleop.get_motion_state_transformed(), np.zeros(6)
        )

    def test_home_combo_is_one_shot_and_gripper_is_disabled(self):
        self.teleop._pressed_buttons.update(
            {FakeEcodes.BTN_SELECT, FakeEcodes.BTN_START}
        )
        self.teleop._update_reset_request(FakeEcodes)

        self.assertTrue(self.teleop.consume_reset_request())
        self.assertFalse(self.teleop.consume_reset_request())
        self.assertFalse(self.teleop.is_button_pressed(0))
        self.assertFalse(self.teleop.is_button_pressed(1))

    def test_detection_requests_plain_axis_codes_from_evdev(self):
        teleop = GamepadTeleop(device_path="/dev/input/event19")

        device = teleop._find_device(FakeDiscoveryEvdev)

        self.assertEqual(device.name, "Microsoft X-Box 360 pad")
        self.assertFalse(FakeInputDevice.last_absinfo)


if __name__ == "__main__":
    unittest.main()
