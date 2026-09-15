#!/usr/bin/env python3
"""Local JSON action bridge. Default: mock. Live startup explicitly enables can0."""

import argparse
import json
import signal
import time

import zmq

from backend import HardwareBackend, MockBackend
from control import Executor, PROTOCOL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--allow-hardware-enable", action="store_true")
    parser.add_argument("--sdk", default="/home/hjadmin/projects/arx5-sdk")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--workspace-min", nargs=3, type=float)
    parser.add_argument("--workspace-max", nargs=3, type=float)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("invalid port")
    if args.live and (not args.allow_hardware_enable or args.workspace_min is None or args.workspace_max is None):
        parser.error("live mode requires --allow-hardware-enable and explicit workspace bounds")
    lower = args.workspace_min if args.workspace_min is not None else [.29, -.03, .13]
    upper = args.workspace_max if args.workspace_max is not None else [.35, .03, .19]
    # Validate configuration and bind the service before touching hardware.
    Executor(MockBackend(), lower, upper)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, 50)
    socket.setsockopt(zmq.MAXMSGSIZE, 16384)
    backend, executor = None, None
    running = True

    def shutdown(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        socket.bind("tcp://127.0.0.1:" + str(args.port))
        backend = HardwareBackend(args.sdk) if args.live else MockBackend()
        executor = Executor(backend, lower, upper)
        print(json.dumps({"bridge": PROTOCOL, "source": backend.source, "port": args.port}), flush=True)
        while running:
            executor.tick()
            if not socket.poll(5):
                continue
            try:
                message = socket.recv_json()
                if not isinstance(message, dict) or set(message) != {"protocol", "command", "data"}:
                    raise ValueError("expected protocol, command, data")
                if message["protocol"] != PROTOCOL:
                    raise ValueError("unsupported protocol")
                command, data = message["command"], message["data"]
                if command != "action" and data != {}:
                    raise ValueError("non-action data must be an empty object")
                if command == "info":
                    result = executor.info()
                elif command == "state":
                    result = executor.snapshot()
                elif command == "arm":
                    result = executor.arm()
                elif command == "action":
                    result = executor.submit(data)
                elif command == "stop":
                    executor.stop()
                    result = {"mode": executor.mode, "reason": executor.reason}
                else:
                    raise ValueError("unknown command")
                reply = {"ok": True, "result": result}
            except ValueError as exc:
                reply = {"ok": False, "error": str(exc)}
            except Exception as exc:
                executor.stop(str(exc))
                reply = {"ok": False, "error": str(exc)}
            executor.tick()
            socket.send_json(reply)
    finally:
        try:
            if executor is not None:
                executor.stop("bridge shutdown")
        finally:
            if backend is not None:
                backend.close()
            socket.close()
            context.term()


if __name__ == "__main__":
    main()
