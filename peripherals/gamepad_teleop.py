import threading
import time

import numpy as np


class GamepadTeleop:
    """Linux USB gamepad adapter with the same 6D motion output as Spacemouse.

    Default mapping (Xbox-style Linux event codes):
      left stick   -> end-effector X/Y translation
      LT / RT      -> Z down/up
      right stick  -> rotation around X/Y
      LB / RB      -> rotation around Z
      Back + Start -> one-shot home reset request

    Gripper buttons are intentionally disabled.
    """

    def __init__(self, device_path=None, deadzone=0.12, scale=0.5):
        if not 0 <= deadzone < 1:
            raise ValueError("deadzone must be in [0, 1)")
        if not 0 < scale <= 1:
            raise ValueError("scale must be in (0, 1]")

        self.device_path = device_path
        self.deadzone = float(deadzone)
        self.scale = float(scale)

        self._device = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._axis_values = {}
        self._axis_info = {}
        self._pressed_buttons = set()
        self._reset_combo_down = False
        self._reset_requested = False
        self._connected = False

    @staticmethod
    def _load_evdev():
        try:
            import evdev
        except ImportError as exc:
            raise RuntimeError(
                "Gamepad support requires evdev. Install it with: pip install evdev"
            ) from exc
        return evdev

    @classmethod
    def list_devices(cls):
        """Return Linux input device paths and names for setup diagnostics."""
        evdev = cls._load_evdev()
        devices = []
        for path in evdev.list_devices():
            try:
                device = evdev.InputDevice(path)
                devices.append((path, device.name))
                device.close()
            except OSError:
                continue
        return devices

    def _find_device(self, evdev):
        required_axes = {
            evdev.ecodes.ABS_X,
            evdev.ecodes.ABS_Y,
            evdev.ecodes.ABS_Z,
            evdev.ecodes.ABS_RX,
            evdev.ecodes.ABS_RY,
            evdev.ecodes.ABS_RZ,
        }
        required_buttons = {
            evdev.ecodes.BTN_TL,
            evdev.ecodes.BTN_TR,
            evdev.ecodes.BTN_SELECT,
            evdev.ecodes.BTN_START,
        }

        if self.device_path is not None:
            candidates = [self.device_path]
        else:
            candidates = evdev.list_devices()

        inspected = []
        for path in candidates:
            try:
                device = evdev.InputDevice(path)
                inspected.append(f"{path}: {device.name}")
                device_caps = device.capabilities()
                axes = set(device_caps.get(evdev.ecodes.EV_ABS, []))
                buttons = set(device_caps.get(evdev.ecodes.EV_KEY, []))
                if required_axes.issubset(axes) and required_buttons.issubset(buttons):
                    return device
                device.close()
            except OSError as exc:
                inspected.append(f"{path}: unavailable ({exc})")

        detail = "\n  ".join(inspected) if inspected else "no readable input devices"
        raise RuntimeError(
            "No compatible Xbox-style gamepad was found. Inspected:\n  " + detail
        )

    def __enter__(self):
        evdev = self._load_evdev()
        self._device = self._find_device(evdev)
        capabilities = self._device.capabilities(absinfo=True)
        for code, info in capabilities.get(evdev.ecodes.EV_ABS, []):
            if info is not None:
                self._axis_info[code] = info
                self._axis_values[code] = info.value

        self._stop_event.clear()
        self._connected = True
        self._thread = threading.Thread(
            target=self._read_events,
            args=(evdev,),
            name="gamepad-teleop",
            daemon=True,
        )
        self._thread.start()
        print(f"Using gamepad: {self._device.path} ({self._device.name})")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_event.set()
        if self._device is not None:
            self._device.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._set_disconnected()

    def _set_disconnected(self):
        with self._lock:
            self._connected = False
            self._axis_values.clear()
            self._pressed_buttons.clear()
            self._reset_combo_down = False
            self._reset_requested = False

    def _read_events(self, evdev):
        try:
            for event in self._device.read_loop():
                if self._stop_event.is_set():
                    break
                with self._lock:
                    if event.type == evdev.ecodes.EV_ABS:
                        self._axis_values[event.code] = event.value
                    elif event.type == evdev.ecodes.EV_KEY:
                        if event.value:
                            self._pressed_buttons.add(event.code)
                        else:
                            self._pressed_buttons.discard(event.code)
                        self._update_reset_request(evdev.ecodes)
        except OSError as exc:
            if not self._stop_event.is_set():
                print(f"Gamepad disconnected; motion output forced to zero: {exc}")
        finally:
            self._set_disconnected()

    def _update_reset_request(self, ecodes):
        combo_down = (
            ecodes.BTN_SELECT in self._pressed_buttons
            and ecodes.BTN_START in self._pressed_buttons
        )
        if combo_down and not self._reset_combo_down:
            self._reset_requested = True
        self._reset_combo_down = combo_down

    def _normalize_stick(self, code):
        info = self._axis_info[code]
        value = self._axis_values.get(code, info.value)
        center = (info.minimum + info.maximum) / 2.0
        span = (info.maximum - info.minimum) / 2.0
        normalized = 0.0 if span <= 0 else (value - center) / span
        normalized = float(np.clip(normalized, -1.0, 1.0))
        magnitude = abs(normalized)
        if magnitude <= self.deadzone:
            return 0.0
        return np.sign(normalized) * (
            (magnitude - self.deadzone) / (1.0 - self.deadzone)
        )

    def _normalize_trigger(self, code):
        info = self._axis_info[code]
        value = self._axis_values.get(code, info.minimum)
        span = info.maximum - info.minimum
        if span <= 0:
            return 0.0
        return float(np.clip((value - info.minimum) / span, 0.0, 1.0))

    def get_motion_state_transformed(self):
        evdev = self._load_evdev()
        e = evdev.ecodes
        with self._lock:
            if not self._connected:
                return np.zeros(6, dtype=np.float64)

            left_x = self._normalize_stick(e.ABS_X)
            left_y = self._normalize_stick(e.ABS_Y)
            right_x = self._normalize_stick(e.ABS_RX)
            right_y = self._normalize_stick(e.ABS_RY)
            left_trigger = self._normalize_trigger(e.ABS_Z)
            right_trigger = self._normalize_trigger(e.ABS_RZ)
            roll_z = (
                int(e.BTN_TL in self._pressed_buttons)
                - int(e.BTN_TR in self._pressed_buttons)
            )

        state = np.array(
            [
                -left_y,
                -left_x,
                right_trigger - left_trigger,
                -right_y,
                -right_x,
                roll_z,
            ],
            dtype=np.float64,
        )
        return state * self.scale

    def is_button_pressed(self, button_id):
        """Gamepad gripper control is intentionally disabled."""
        if button_id not in (0, 1):
            raise ValueError(f"Unknown button id: {button_id}")
        return False

    def consume_reset_request(self):
        """Consume the rising edge of the Back + Start reset combination."""
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
            return requested


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Inspect/test a Linux gamepad")
    parser.add_argument("--list", action="store_true", help="list input devices")
    parser.add_argument("--device", help="input event path, e.g. /dev/input/event12")
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()

    if args.list:
        devices = GamepadTeleop.list_devices()
        if not devices:
            print("No readable Linux input devices found.")
        for path, name in devices:
            print(f"{path}: {name}")
        return

    try:
        with GamepadTeleop(
            device_path=args.device,
            deadzone=args.deadzone,
            scale=args.scale,
        ) as gamepad:
            print("Move the controls; press Ctrl+C to stop.")
            while True:
                state = gamepad.get_motion_state_transformed()
                reset = gamepad.consume_reset_request()
                print(
                    "\r6D: "
                    + np.array2string(state, precision=3, suppress_small=True)
                    + ("  HOME" if reset else "      "),
                    end="",
                    flush=True,
                )
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    _main()
