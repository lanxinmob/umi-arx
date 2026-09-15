"""Unified MCP: python -m robot_api.server (mock by default)."""

import argparse
import json
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import sys

from action_mcp.server import VECTOR
from observation_mcp.server import ObservationMcpServer
from .api import CONFIG, ROOT, RobotAPI


def schema(properties=None, required=()):
    return {"type": "object", "additionalProperties": False,
            "properties": properties or {}, "required": list(required)}


IDS = {"request_id": {"type": "string", "minLength": 1, "maxLength": 128},
       "observation_id": {"type": "string", "minLength": 1},
       "duration_s": {"type": "number", "minimum": 1, "maximum": 5, "default": 5}}
TOOLS = {
    "status": ("Read source, actual limits, controller state and physical readiness blockers.", schema()),
    "observe": ("Return 1-8 chronological images, proprioception and local image paths. No motion.",
                ObservationMcpServer._tools()[0]["inputSchema"]),
    "arm": ("Explicitly engage measured-pose holding, once per bridge. Returns new observation. May move hardware; never resets.", schema()),
    "move": ("One bounded relative SDK eef_link move, metres/radians. Consume latest observation_id; return measured outcome and new images. executed is progress, not exact arrival.",
             schema({**IDS, "delta_xyz_m": VECTOR, "delta_rotvec_rad": VECTOR,
                     "frame": {"enum": ["base", "tool"], "default": "base"}},
                    ("request_id", "observation_id", "delta_xyz_m"))),
    "gripper": ("One bounded width increment in metres: positive opens, negative closes. Never asserts grasp success.",
                schema({**IDS, "delta_m": {"type": "number", "minimum": -.002, "maximum": .002}},
                       ("request_id", "observation_id", "delta_m"))),
    "stop": ("Latch damping; NOT power removal. No automatic restart.", schema()),
}


class RobotMcpServer(ObservationMcpServer):
    def __init__(self, api):
        super().__init__(api.provider)
        self.api = api

    @staticmethod
    def _tools():
        return [{"name": name, "description": description, "inputSchema": inputs,
                 "annotations": {"readOnlyHint": name in ("status", "observe"),
                                 "destructiveHint": name not in ("status", "observe"),
                                 "openWorldHint": False}}
                for name, (description, inputs) in TOOLS.items()]

    def dispatch(self, method, params):
        result = super().dispatch(method, params)
        if method == "initialize":
            result["serverInfo"] = {"name": "umi-arx-robot", "version": "0.1.0"}
            result["instructions"] = (
                "Call status, then arm explicitly, then use returned observation_id for move/gripper. "
                "Always inspect returned images; do not assume target arrival or grasp. "
                "Use a unique request_id; retry with identical arguments only. "
                "Read actual limits from status. Live commissioning failures block actuation. "
                "stop is damping, not power removal. Mock images are fixtures, not camera observations."
            )
        return result

    def _call_tool(self, params):
        try:
            name, args = params.get("name"), params.get("arguments", {})
            if name not in TOOLS or not isinstance(args, dict):
                raise ValueError("unknown tool or invalid arguments")
            inputs = TOOLS[name][1]
            if set(args) - set(inputs["properties"]) or set(inputs.get("required", ())) - set(args):
                raise ValueError("arguments must match the tool schema exactly")
            return getattr(self.api, name)(**args)
        except Exception as exc:
            return self.api._result({"error": str(exc)}, failed=True)


def start_mock(port=None):
    if port is None:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "action_mcp/bridge.py"), "--port", str(port), "--visual-steps"],
        stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(5):
                raise RuntimeError("mock bridge startup timed out")
            ready = json.loads(process.stdout.readline())
        if ready.get("source") != "mock" or ready.get("port") != port or process.poll() is not None:
            raise RuntimeError("mock bridge failed identity check")
        return process, port
    except Exception:
        stop_mock(process)
        raise


def stop_mock(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()  # Only this owned, synthetic backend process.
            process.wait(timeout=3)
    process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="attach to an existing live bridge; never enables hardware")
    parser.add_argument("--port", type=int, help="live bridge port (default 8766) or new mock bridge port")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    process, api = None, None

    def shutdown(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        port = args.port or 8766
        if not args.live:
            process, port = start_mock(args.port)
        api = RobotAPI(port, source="live" if args.live else "mock", config=args.config, output=args.output)
        RobotMcpServer(api).serve_stdio()
    finally:
        if api is not None:
            api.close()
        if process is not None:
            stop_mock(process)


if __name__ == "__main__":
    main()
