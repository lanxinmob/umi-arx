#!/usr/bin/env python3
"""MCP stdio proxy for the independent, loopback-only action bridge."""

import argparse
import json
import sys
import time

import zmq

from control import PROTOCOL


EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}
VECTOR = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
ACTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["session", "sequence", "state_token", "frame", "delta_xyz_m",
                 "delta_rotvec_rad", "gripper_delta_m", "duration_s"],
    "properties": {"session": {"type": "string"}, "sequence": {"type": "integer", "minimum": 1},
                   "state_token": {"type": "string"}, "frame": {"enum": ["base", "tool"]},
                   "delta_xyz_m": VECTOR, "delta_rotvec_rad": VECTOR,
                   "gripper_delta_m": {"type": "number", "minimum": -.002, "maximum": .002},
                   "duration_s": {"type": "number", "minimum": 1, "maximum": 5}}}
TOOLS = [
    ("get_action_contract", "info", "Read units, frames, limits and backend source; never moves hardware."),
    ("get_action_state", "state", "Read measured state and a short-lived state_token; does not renew control lease."),
    ("arm_action_session", "arm", "Ramp holding gains at current pose; no reset. May move hardware while engaging."),
    ("execute_action", "action", "Execute one bounded relative action and wait for measured completion or failure."),
    ("stop_action_session", "stop", "Latch damping and reject further motion until bridge restart; NOT power removal."),
]


class Client:
    def __init__(self, port):
        self.endpoint = "tcp://127.0.0.1:" + str(port)
        self.context = zmq.Context()
        self.owns_session = False

    def request(self, command, data=None):
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 500)
        socket.setsockopt(zmq.SNDTIMEO, 500)
        socket.connect(self.endpoint)
        try:
            socket.send_json({"protocol": PROTOCOL, "command": command, "data": {} if data is None else data})
            reply = socket.recv_json()
            if not isinstance(reply, dict) or type(reply.get("ok")) is not bool:
                raise RuntimeError("invalid bridge response")
            if not reply["ok"]:
                raise ValueError(reply.get("error", "bridge rejected request"))
            return reply["result"]
        finally:
            socket.close()

    def call(self, command, args):
        if command != "action" and args != {}:
            raise ValueError("this tool takes no arguments")
        if command == "arm":
            # Also stop on EOF when the arm reply was lost after acceptance.
            self.owns_session = True
        result = self.request(command, args)
        if command == "arm":
            until = time.monotonic()+3
            while time.monotonic() < until:
                state = self.request("state")
                if state["mode"] != "arming":
                    return state
                time.sleep(.05)
            raise RuntimeError("arming did not finish")
        if command != "action":
            return result
        until = time.monotonic()+7
        while result["status"] == "running" and time.monotonic() < until:
            time.sleep(.05)
            state = self.request("state")
            result = state["last_action"]
            if result is None or result["sequence"] != args["sequence"]:
                raise RuntimeError("action result lost or replaced")
        if result["status"] == "running":
            self.request("stop")
            raise RuntimeError("action result deadline exceeded; stop requested")
        return result

    def close(self):
        if self.owns_session:
            try:
                self.request("stop")
            except Exception as exc:
                print("Could not confirm stop: " + str(exc), file=sys.stderr)
        self.context.term()


def dispatch(client, request):
    method = request.get("method")
    params = request.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    if method == "initialize":
        version = params.get("protocolVersion")
        supported = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25", "2026-07-28")
        return {"protocolVersion": version if version in supported else "2025-11-25",
                "capabilities": {"tools": {}}, "serverInfo": {"name": "umi-arx-action", "version": "0.1.0"}}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [{"name": name, "description": description,
                           "inputSchema": ACTION_SCHEMA if cmd == "action" else EMPTY,
                           "annotations": {"readOnlyHint": cmd in ("info", "state"),
                                           "destructiveHint": cmd not in ("info", "state"),
                                           "openWorldHint": False}}
                          for name, cmd, description in TOOLS]}
    if method != "tools/call":
        raise LookupError("method not found")
    name = params.get("name")
    command = next((cmd for n, cmd, _ in TOOLS if n == name), None)
    if command is None:
        raise ValueError("unknown tool")
    try:
        result = client.call(command, params.get("arguments", {}))
        failed = result.get("status") == "failed" or (command == "arm" and result.get("mode") != "ready")
        return {"content": [{"type": "text", "text": json.dumps(result, allow_nan=False)}],
                "structuredContent": result, "isError": failed}
    except Exception as exc:
        # Transport failure after a write means outcome unknown, never success.
        stop = None
        if command in ("action", "arm") and not isinstance(exc, ValueError):
            try:
                stop = client.request("stop")
            except Exception:
                stop = "unconfirmed; bridge deadline remains the fallback"
        result = {"error": str(exc), "stop": stop}
        return {"content": [{"type": "text", "text": json.dumps(result)}],
                "structuredContent": result, "isError": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("invalid port")
    client = Client(args.port)
    try:
        for line in sys.stdin:
            request_id = None
            try:
                if len(line) > 65536:
                    raise ValueError("request too large")
                request = json.loads(line)
                if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                    raise ValueError("expected JSON-RPC 2.0 object")
                request_id = request.get("id")
                if "id" not in request:
                    continue
                result = dispatch(client, request)
                response = {"jsonrpc": "2.0", "id": request_id, "result": result}
            except Exception as exc:
                code = -32700 if isinstance(exc, json.JSONDecodeError) else -32601 if isinstance(exc, LookupError) else -32602
                response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": str(exc)}}
            print(json.dumps(response, allow_nan=False), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
