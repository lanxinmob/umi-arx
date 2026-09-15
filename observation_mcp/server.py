#!/usr/bin/env python3
"""Read-only MCP server for timestamped ARX5 observations."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
import os
import sys
import time
from typing import Any, Protocol

import cv2
import numpy as np


MCP_PROTOCOL_VERSION = "2026-07-28"
OBSERVATION_PROTOCOL_VERSION = "umi-arx.observation.v1"
SERVER_VERSION = "0.1.0"
SCHEMA_URI = "umi-arx://observation/schema/v1"
SUPPORTED_MCP_VERSIONS = {
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
    MCP_PROTOCOL_VERSION,
}


OBSERVATION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": SCHEMA_URI,
    "title": "UMI ARX Observation v1",
    "type": "object",
    "additionalProperties": False,
    "required": ["protocol", "source", "generated_at_unix_ns", "samples"],
    "properties": {
        "protocol": {"const": OBSERVATION_PROTOCOL_VERSION},
        "source": {"enum": ["mock", "live"]},
        "generated_at_unix_ns": {"type": "integer"},
        "samples": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "sequence",
                    "captured_at_unix_ns",
                    "relative_time_ms",
                    "image",
                    "robot",
                ],
                "properties": {
                    "sequence": {"type": "integer", "minimum": 0},
                    "captured_at_unix_ns": {"type": "integer"},
                    "relative_time_ms": {"type": "number"},
                    "image": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "camera_id",
                            "content_index",
                            "mime_type",
                            "width",
                            "height",
                        ],
                        "properties": {
                            "camera_id": {"type": "string"},
                            "content_index": {"type": "integer", "minimum": 1},
                            "mime_type": {"const": "image/jpeg"},
                            "width": {"type": "integer", "minimum": 1},
                            "height": {"type": "integer", "minimum": 1},
                        },
                    },
                    "robot": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "host_received_at_unix_ns",
                            "source_timestamp_s",
                            "host_receive_skew_ms",
                            "ee_pose_base_xyz_rpy",
                            "tcp_pose_base_xyz_rotvec",
                            "joint_position_rad",
                            "joint_velocity_rad_s",
                            "joint_torque_nm",
                            "gripper_width_m",
                            "gripper_velocity_m_s",
                            "gripper_torque",
                        ],
                        "properties": {
                            "host_received_at_unix_ns": {"type": "integer"},
                            "source_timestamp_s": {"type": "number"},
                            "host_receive_skew_ms": {"type": "number", "minimum": 0},
                            "ee_pose_base_xyz_rpy": {"$ref": "#/$defs/vector6"},
                            "tcp_pose_base_xyz_rotvec": {"$ref": "#/$defs/vector6"},
                            "joint_position_rad": {"$ref": "#/$defs/vector"},
                            "joint_velocity_rad_s": {"$ref": "#/$defs/vector"},
                            "joint_torque_nm": {"$ref": "#/$defs/vector"},
                            "gripper_width_m": {"type": "number"},
                            "gripper_velocity_m_s": {"type": "number"},
                            "gripper_torque": {"type": "number"},
                        },
                    },
                },
            },
        },
    },
    "$defs": {
        "vector": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "number"},
        },
        "vector6": {
            "type": "array",
            "minItems": 6,
            "maxItems": 6,
            "items": {"type": "number"},
        },
    },
}


@dataclass(frozen=True)
class ObservationSample:
    sequence: int
    captured_at_unix_ns: int
    image_jpeg: bytes
    image_width: int
    image_height: int
    robot: dict[str, Any]


class ObservationProvider(Protocol):
    source: str

    def status(self) -> dict[str, Any]: ...

    def snapshot(
        self, sample_count: int, history_seconds: float, wait_timeout_seconds: float
    ) -> list[ObservationSample]: ...

    def close(self) -> None: ...


class MockObservationProvider:
    source = "mock"

    def __init__(self, width: int = 224, height: int = 224) -> None:
        self.width = width
        self.height = height
        self._sequence = 0

    def status(self) -> dict[str, Any]:
        return {
            "protocol": OBSERVATION_PROTOCOL_VERSION,
            "source": self.source,
            "ready": True,
            "detail": "deterministic mock observations",
        }

    def snapshot(
        self, sample_count: int, history_seconds: float, wait_timeout_seconds: float
    ) -> list[ObservationSample]:
        del wait_timeout_seconds
        now_ns = time.time_ns()
        offsets = np.linspace(-history_seconds, 0.0, sample_count)
        samples: list[ObservationSample] = []
        for index, offset in enumerate(offsets):
            sequence = self._sequence + index
            captured_at_ns = now_ns + int(float(offset) * 1_000_000_000)
            position = 0.001 * sequence
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            image[:, :, 0] = np.uint8((40 + sequence * 7) % 255)
            image[:, :, 1] = np.uint8((100 + sequence * 11) % 255)
            cv2.putText(
                image,
                f"mock {sequence}",
                (12, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            ok, encoded = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if not ok:
                raise RuntimeError("failed to encode mock image")
            samples.append(
                ObservationSample(
                    sequence=sequence,
                    captured_at_unix_ns=captured_at_ns,
                    image_jpeg=encoded.tobytes(),
                    image_width=self.width,
                    image_height=self.height,
                    robot={
                        "host_received_at_unix_ns": captured_at_ns,
                        "source_timestamp_s": captured_at_ns / 1_000_000_000,
                        "host_receive_skew_ms": 0.0,
                        "ee_pose_base_xyz_rpy": [position, 0.0, 0.3, 0.0, 0.0, 0.0],
                        "tcp_pose_base_xyz_rotvec": [
                            position,
                            0.0,
                            0.3,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        "joint_position_rad": [0.0] * 6,
                        "joint_velocity_rad_s": [0.0] * 6,
                        "joint_torque_nm": [0.0] * 6,
                        "gripper_width_m": 0.04,
                        "gripper_velocity_m_s": 0.0,
                        "gripper_torque": 0.0,
                    },
                )
            )
        self._sequence += sample_count
        return samples

    def close(self) -> None:
        return None


class McpProtocolError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _observation_content(
    provider: ObservationProvider,
    sample_count: int,
    history_seconds: float,
    wait_timeout_seconds: float,
) -> list[dict[str, Any]]:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError("sample_count must be an integer")
    if not 1 <= sample_count <= 8:
        raise ValueError("sample_count must be between 1 and 8")
    if isinstance(history_seconds, bool) or not isinstance(history_seconds, (int, float)):
        raise ValueError("history_seconds must be a number")
    if not 0.0 <= float(history_seconds) <= 8.0:
        raise ValueError("history_seconds must be between 0 and 8")
    if sample_count > 1 and history_seconds <= 0:
        raise ValueError("history_seconds must be positive when requesting history")
    if not isinstance(wait_timeout_seconds, (int, float)) or isinstance(
        wait_timeout_seconds, bool
    ):
        raise ValueError("wait_timeout_seconds must be a number")
    if not 0.0 <= float(wait_timeout_seconds) <= 15.0:
        raise ValueError("wait_timeout_seconds must be between 0 and 15")

    samples = provider.snapshot(
        sample_count, float(history_seconds), float(wait_timeout_seconds)
    )
    newest_ns = samples[-1].captured_at_unix_ns
    metadata_samples: list[dict[str, Any]] = []
    content: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        metadata_samples.append(
            {
                "sequence": sample.sequence,
                "captured_at_unix_ns": sample.captured_at_unix_ns,
                "relative_time_ms": round(
                    (sample.captured_at_unix_ns - newest_ns) / 1_000_000, 3
                ),
                "image": {
                    "camera_id": "wrist",
                    "content_index": index,
                    "mime_type": "image/jpeg",
                    "width": sample.image_width,
                    "height": sample.image_height,
                },
                "robot": sample.robot,
            }
        )
    packet = {
        "protocol": OBSERVATION_PROTOCOL_VERSION,
        "source": provider.source,
        "generated_at_unix_ns": time.time_ns(),
        "samples": metadata_samples,
    }
    content.append(
        {
            "type": "text",
            "text": json.dumps(packet, ensure_ascii=False, separators=(",", ":")),
        }
    )
    for sample in samples:
        content.append(
            {
                "type": "image",
                "data": base64.b64encode(sample.image_jpeg).decode("ascii"),
                "mimeType": "image/jpeg",
            }
        )
    return content


class ObservationMcpServer:
    def __init__(self, provider: ObservationProvider) -> None:
        self.provider = provider
        self.initialized = False

    @staticmethod
    def _tools() -> list[dict[str, Any]]:
        read_only = {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
        return [
            {
                "name": "get_observation",
                "title": "Get synchronized ARX5 observation history",
                "description": (
                    "Return chronological wrist-camera images with synchronized "
                    "ARX5 end-effector, TCP, joints, and gripper state. The first "
                    "content block is an umi-arx.observation.v1 JSON packet; each "
                    "following image block is addressed by image.content_index. "
                    "This tool is strictly read-only."
                ),
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "sample_count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 8,
                            "default": 4,
                        },
                        "history_seconds": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 8,
                            "default": 2.4,
                        },
                        "wait_timeout_seconds": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 15,
                            "default": 5,
                        },
                    },
                },
                "annotations": read_only,
            },
            {
                "name": "get_observation_status",
                "title": "Get ARX5 observation source status",
                "description": "Return readiness and freshness without reading image content.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "annotations": read_only,
            },
        ]

    def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if method == "initialize":
            requested = params.get("protocolVersion")
            negotiated = (
                requested if requested in SUPPORTED_MCP_VERSIONS else MCP_PROTOCOL_VERSION
            )
            self.initialized = True
            return {
                "protocolVersion": negotiated,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                },
                "serverInfo": {
                    "name": "umi-arx-observation",
                    "version": SERVER_VERSION,
                },
                "instructions": (
                    "Use get_observation for synchronized read-only camera and robot "
                    "state. The server exposes no motion commands."
                ),
            }
        if method in {"notifications/initialized", "notifications/cancelled"}:
            return None
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self._tools()}
        if method == "tools/call":
            return self._call_tool(params)
        if method == "resources/list":
            return {
                "resources": [
                    {
                        "uri": SCHEMA_URI,
                        "name": "UMI ARX observation protocol v1",
                        "description": "JSON Schema for get_observation metadata.",
                        "mimeType": "application/schema+json",
                    }
                ]
            }
        if method == "resources/read":
            if params.get("uri") != SCHEMA_URI:
                raise McpProtocolError(-32002, "resource not found")
            return {
                "contents": [
                    {
                        "uri": SCHEMA_URI,
                        "mimeType": "application/schema+json",
                        "text": json.dumps(OBSERVATION_SCHEMA, indent=2),
                    }
                ]
            }
        raise McpProtocolError(-32601, f"method not found: {method}")

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise McpProtocolError(-32602, "tool arguments must be an object")
        try:
            if name == "get_observation":
                content = _observation_content(
                    self.provider,
                    arguments.get("sample_count", 4),
                    arguments.get("history_seconds", 2.4),
                    arguments.get("wait_timeout_seconds", 5.0),
                )
                return {"content": content, "isError": False}
            if name == "get_observation_status":
                if arguments:
                    raise ValueError("get_observation_status takes no arguments")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                self.provider.status(),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        }
                    ],
                    "isError": False,
                }
            raise McpProtocolError(-32602, f"unknown tool: {name}")
        except McpProtocolError:
            raise
        except (RuntimeError, TimeoutError, ValueError) as exc:
            return {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            }

    def serve_stdio(self) -> None:
        for raw_line in sys.stdin.buffer:
            has_id = False
            request_id: Any = None
            try:
                request = json.loads(raw_line)
                if not isinstance(request, dict):
                    raise McpProtocolError(-32600, "request must be an object")
                has_id = "id" in request
                request_id = request.get("id")
                if request.get("jsonrpc") != "2.0":
                    raise McpProtocolError(-32600, "jsonrpc must be '2.0'")
                method = request.get("method")
                params = request.get("params", {})
                if not isinstance(method, str) or not isinstance(params, dict):
                    raise McpProtocolError(-32600, "invalid method or params")
                result = self.dispatch(method, params)
                if not has_id or result is None:
                    continue
                response = {"jsonrpc": "2.0", "id": request_id, "result": result}
            except json.JSONDecodeError as exc:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32700, "message": f"parse error: {exc.msg}"},
                }
            except McpProtocolError as exc:
                if not has_id:
                    continue
                error = {"code": exc.code, "message": exc.message}
                if exc.data is not None:
                    error["data"] = exc.data
                response = {"jsonrpc": "2.0", "id": request_id, "error": error}
            except Exception as exc:
                print(f"unexpected MCP server error: {exc}", file=sys.stderr)
                if not has_id:
                    continue
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": "internal error"},
                }
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="serve deterministic observations without camera or robot hardware",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.json"),
        help="live camera, robot, and sampling JSON configuration",
    )
    parser.add_argument(
        "--action-port", type=int,
        help="read robot state from the JSON action bridge instead of the legacy controller",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mock:
        provider: ObservationProvider = MockObservationProvider()
    else:
        from live import LiveObservationProvider

        provider = LiveObservationProvider(args.config, action_port=args.action_port)
    try:
        ObservationMcpServer(provider).serve_stdio()
    finally:
        provider.close()


if __name__ == "__main__":
    main()
