from threading import Lock

import numpy as np
from pynput.keyboard import Key, KeyCode, Listener


class KeyboardTeleop:
    """
    输出格式与 Spacemouse.get_motion_state_transformed() 相同：
    [x, y, z, rx, ry, rz]
    """

    def __init__(self, fine_scale=0.1, fast_scale=0.5):
        self.fine_scale = fine_scale
        self.fast_scale = fast_scale
        self._pressed = set()
        self._lock = Lock()
        self._listener = Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )

    def _on_press(self, key):
        with self._lock:
            self._pressed.add(key)

    def _on_release(self, key):
        with self._lock:
            self._pressed.discard(key)

    def __enter__(self):
        self._listener.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._listener.stop()

    def get_motion_state_transformed(self):
        with self._lock:
            pressed = set(self._pressed)

        state = np.zeros(6, dtype=np.float64)

        # 位置
        state[0] = (
            int(Key.up in pressed or KeyCode.from_char("w") in pressed)
            - int(Key.down in pressed or KeyCode.from_char("s") in pressed)
        )
        state[1] = (
            int(Key.left in pressed or KeyCode.from_char("a") in pressed)
            - int(Key.right in pressed or KeyCode.from_char("d") in pressed)
        )
        state[2] = (
            int(Key.page_up in pressed or KeyCode.from_char("z") in pressed)
            - int(Key.page_down in pressed or KeyCode.from_char("x") in pressed)
        )

        # 避开 eval_arx5.py 已使用的 Q/C/S/W/E/A
        state[3] = (
            int(KeyCode.from_char("i") in pressed)
            - int(KeyCode.from_char("k") in pressed)
        )
        state[4] = (
            int(KeyCode.from_char("j") in pressed)
            - int(KeyCode.from_char("l") in pressed)
        )
        state[5] = (
            int(KeyCode.from_char("u") in pressed)
            - int(KeyCode.from_char("o") in pressed)
        )

        fast = Key.shift in pressed or Key.shift_r in pressed
        scale = self.fast_scale if fast else self.fine_scale
        return state * scale

    def is_button_pressed(self, button_id):
        with self._lock:
            pressed = set(self._pressed)

        # 保持与原 SpaceMouse 按钮语义一致
        if button_id == 0:
            return KeyCode.from_char("f") in pressed  # 关闭夹爪
        if button_id == 1:
            return KeyCode.from_char("r") in pressed  # 打开夹爪
        raise ValueError(f"Unknown button id: {button_id}")
