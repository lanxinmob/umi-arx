from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path


CSV_FIELDS = [
    "episode_id",
    "iter",
    "elapsed_s",
    "current_x_m",
    "current_y_m",
    "current_z_m",
    "current_rx_rad",
    "current_ry_rad",
    "current_rz_rad",
    "raw_x",
    "raw_y",
    "raw_z",
    "raw_rx_rad",
    "raw_ry_rad",
    "raw_rz_rad",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "target_rx_rad",
    "target_ry_rad",
    "target_rz_rad",
    "target_gripper_m",
    "position_command_delta_m",
    "rotation_command_delta_deg",
    "inference_latency_ms",
]


class _RunningStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.maximum = None

    def add(self, value):
        value = float(value)
        self.count += 1
        self.total += value
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def as_dict(self):
        return {
            "mean": self.total / self.count if self.count else None,
            "max": self.maximum,
        }


class PoseEpisodeLogger:
    """Write one directly inspectable pose report per policy episode."""

    def __init__(self, root_dir, episode_id, metadata=None, max_plot_samples=4000):
        self.episode_id = int(episode_id)
        self.root_dir = Path(root_dir)
        self.episode_dir = self.root_dir / str(self.episode_id)
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        self.trace_path = self.episode_dir / "trace.csv"
        self._trace_file = self.trace_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._trace_file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._trace_file.flush()

        self.metadata = dict(metadata or {})
        self.started_at = datetime.now(timezone.utc).astimezone().isoformat()
        self._started_monotonic = time.monotonic()
        self._last_elapsed_s = 0.0
        self._row_count = 0
        self._finished = False
        self._plot_rows = []
        self._plot_stride = 1
        self._max_plot_samples = max(100, int(max_plot_samples))
        self._stats = {
            "position_command_delta_m": _RunningStats(),
            "rotation_command_delta_deg": _RunningStats(),
            "inference_latency_ms": _RunningStats(),
        }

    @staticmethod
    def _three(values, name):
        values = list(values)
        if len(values) != 3:
            raise ValueError(f"{name} must contain exactly three values")
        return [float(value) for value in values]

    def log_policy_sample(
        self,
        *,
        iter_idx,
        elapsed_s,
        current_pos,
        current_rot,
        raw_pos,
        raw_rot,
        target_pos,
        target_rot,
        target_gripper_m,
        position_command_delta_m,
        rotation_command_delta_deg,
        inference_latency_ms,
    ):
        if self._finished:
            raise RuntimeError("Cannot write to a finished pose episode log")

        current_pos = self._three(current_pos, "current_pos")
        current_rot = self._three(current_rot, "current_rot")
        raw_pos = self._three(raw_pos, "raw_pos")
        raw_rot = self._three(raw_rot, "raw_rot")
        target_pos = self._three(target_pos, "target_pos")
        target_rot = self._three(target_rot, "target_rot")

        row = dict(
            episode_id=self.episode_id,
            iter=int(iter_idx),
            elapsed_s=float(elapsed_s),
            current_x_m=current_pos[0],
            current_y_m=current_pos[1],
            current_z_m=current_pos[2],
            current_rx_rad=current_rot[0],
            current_ry_rad=current_rot[1],
            current_rz_rad=current_rot[2],
            raw_x=raw_pos[0],
            raw_y=raw_pos[1],
            raw_z=raw_pos[2],
            raw_rx_rad=raw_rot[0],
            raw_ry_rad=raw_rot[1],
            raw_rz_rad=raw_rot[2],
            target_x_m=target_pos[0],
            target_y_m=target_pos[1],
            target_z_m=target_pos[2],
            target_rx_rad=target_rot[0],
            target_ry_rad=target_rot[1],
            target_rz_rad=target_rot[2],
            target_gripper_m=float(target_gripper_m),
            position_command_delta_m=float(position_command_delta_m),
            rotation_command_delta_deg=float(rotation_command_delta_deg),
            inference_latency_ms=float(inference_latency_ms),
        )
        self._writer.writerow(row)
        self._trace_file.flush()

        self._row_count += 1
        self._last_elapsed_s = row["elapsed_s"]
        for key, stats in self._stats.items():
            stats.add(row[key])

        if (self._row_count - 1) % self._plot_stride == 0:
            self._plot_rows.append(row)
        if len(self._plot_rows) > self._max_plot_samples:
            self._plot_rows = self._plot_rows[::2]
            self._plot_stride *= 2

    def finish(self, status, error=None):
        if self._finished:
            return
        self._finished = True
        self._trace_file.close()

        duration_s = max(
            self._last_elapsed_s,
            time.monotonic() - self._started_monotonic,
        )
        summary = {
            "episode_id": self.episode_id,
            "status": str(status),
            "error": None if error is None else str(error),
            "started_at": self.started_at,
            "duration_s": duration_s,
            "samples": self._row_count,
            "plot_stride": self._plot_stride,
            "metadata": self.metadata,
            "metrics": {key: value.as_dict() for key, value in self._stats.items()},
            "metric_note": (
                "Position and rotation deltas are commanded changes from the "
                "current observation to the first decoded target; they are not "
                "time-aligned execution tracking errors."
            ),
        }
        (self.episode_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (self.episode_dir / "summary.txt").write_text(
            self._summary_text(summary), encoding="utf-8"
        )
        (self.episode_dir / "summary.svg").write_text(
            self._summary_svg(summary), encoding="utf-8"
        )
        self._write_index()

    @staticmethod
    def _format_metric(value, digits=4):
        return "n/a" if value is None else f"{value:.{digits}f}"

    def _summary_text(self, summary):
        metrics = summary["metrics"]
        lines = [
            f"Episode: {self.episode_id}",
            f"Status: {summary['status']}",
            f"Started: {summary['started_at']}",
            f"Duration: {summary['duration_s']:.2f} s",
            f"Samples: {summary['samples']}",
            "",
            "Command metrics (first decoded target):",
            "  position delta mean/max: "
            f"{self._format_metric(metrics['position_command_delta_m']['mean'])} / "
            f"{self._format_metric(metrics['position_command_delta_m']['max'])} m",
            "  rotation delta mean/max: "
            f"{self._format_metric(metrics['rotation_command_delta_deg']['mean'], 2)} / "
            f"{self._format_metric(metrics['rotation_command_delta_deg']['max'], 2)} deg",
            "  inference latency mean/max: "
            f"{self._format_metric(metrics['inference_latency_ms']['mean'], 1)} / "
            f"{self._format_metric(metrics['inference_latency_ms']['max'], 1)} ms",
            "",
            "Note: these deltas describe command size, not execution tracking error.",
            "Open summary.svg for plots and trace.csv for every recorded sample.",
        ]
        if summary["error"]:
            lines.insert(2, f"Error: {summary['error']}")
        if self.metadata:
            lines.extend(["", "Run metadata:"])
            lines.extend(f"  {key}: {value}" for key, value in self.metadata.items())
        return "\n".join(lines) + "\n"

    @staticmethod
    def _polyline(rows, x_key, y_key, left, top, width, height, y_min, y_max):
        if not rows:
            return ""
        x_values = [float(row[x_key]) for row in rows]
        x_min, x_max = min(x_values), max(x_values)
        if x_max == x_min:
            x_max = x_min + 1.0
        if y_max == y_min:
            y_min -= 0.5
            y_max += 0.5
        points = []
        for row in rows:
            x = left + (float(row[x_key]) - x_min) / (x_max - x_min) * width
            y = top + height - (float(row[y_key]) - y_min) / (y_max - y_min) * height
            points.append(f"{x:.1f},{y:.1f}")
        return " ".join(points)

    def _summary_svg(self, summary):
        width = 1200
        panel_height = 135
        panel_gap = 48
        panel_left = 95
        panel_width = 1040
        panel_top = 170
        panels = [
            ("X position", "m", [("current_x_m", "current", "#2563eb"), ("target_x_m", "target", "#dc2626")]),
            ("Y position", "m", [("current_y_m", "current", "#2563eb"), ("target_y_m", "target", "#dc2626")]),
            ("Z position", "m", [("current_z_m", "current", "#2563eb"), ("target_z_m", "target", "#dc2626")]),
            ("Position command delta", "m", [("position_command_delta_m", "delta", "#7c3aed")]),
            ("Rotation command delta", "deg", [("rotation_command_delta_deg", "delta", "#ea580c")]),
            ("Inference latency", "ms", [("inference_latency_ms", "latency", "#059669")]),
            ("Target gripper width", "m", [("target_gripper_m", "gripper", "#0891b2")]),
        ]
        height = panel_top + len(panels) * (panel_height + panel_gap) + 55
        metrics = summary["metrics"]
        metric_line = (
            "pos delta mean/max "
            f"{self._format_metric(metrics['position_command_delta_m']['mean'])}/"
            f"{self._format_metric(metrics['position_command_delta_m']['max'])} m    |    "
            "rot delta mean/max "
            f"{self._format_metric(metrics['rotation_command_delta_deg']['mean'], 2)}/"
            f"{self._format_metric(metrics['rotation_command_delta_deg']['max'], 2)} deg    |    "
            "latency mean/max "
            f"{self._format_metric(metrics['inference_latency_ms']['mean'], 1)}/"
            f"{self._format_metric(metrics['inference_latency_ms']['max'], 1)} ms"
        )
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f8fafc"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:28px;font-weight:700}.sub{font-size:15px;fill:#475569}.label{font-size:16px;font-weight:700}.axis{font-size:12px;fill:#64748b}</style>',
            f'<text x="48" y="48" class="title">Episode {self.episode_id} pose report</text>',
            f'<text x="48" y="78" class="sub">status: {escape(summary["status"])} | duration: {summary["duration_s"]:.2f}s | samples: {summary["samples"]}</text>',
            f'<text x="48" y="105" class="sub">{escape(metric_line)}</text>',
            '<text x="48" y="132" class="sub">Command deltas are not time-aligned execution tracking errors.</text>',
        ]

        for panel_index, (title, unit, series) in enumerate(panels):
            top = panel_top + panel_index * (panel_height + panel_gap)
            all_values = [
                float(row[key])
                for row in self._plot_rows
                for key, _, _ in series
            ]
            y_min = min(all_values) if all_values else 0.0
            y_max = max(all_values) if all_values else 1.0
            if y_max == y_min:
                padding = max(abs(y_max) * 0.05, 0.001)
                y_min -= padding
                y_max += padding
            else:
                padding = (y_max - y_min) * 0.08
                y_min -= padding
                y_max += padding

            parts.extend(
                [
                    f'<text x="{panel_left}" y="{top - 14}" class="label">{escape(title)} ({escape(unit)})</text>',
                    f'<rect x="{panel_left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="white" stroke="#cbd5e1"/>',
                    f'<text x="{panel_left - 8}" y="{top + 10}" text-anchor="end" class="axis">{y_max:.4g}</text>',
                    f'<text x="{panel_left - 8}" y="{top + panel_height}" text-anchor="end" class="axis">{y_min:.4g}</text>',
                ]
            )
            legend_x = panel_left + 12
            for key, label, color in series:
                points = self._polyline(
                    self._plot_rows,
                    "elapsed_s",
                    key,
                    panel_left,
                    top,
                    panel_width,
                    panel_height,
                    y_min,
                    y_max,
                )
                if points:
                    parts.append(
                        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
                    )
                parts.append(
                    f'<line x1="{legend_x}" y1="{top + 20}" x2="{legend_x + 24}" y2="{top + 20}" stroke="{color}" stroke-width="3"/>'
                )
                parts.append(
                    f'<text x="{legend_x + 30}" y="{top + 25}" class="axis">{escape(label)}</text>'
                )
                legend_x += 100
            parts.append(
                f'<text x="{panel_left + panel_width}" y="{top + panel_height + 18}" text-anchor="end" class="axis">elapsed time (s)</text>'
            )

        parts.append("</svg>")
        return "\n".join(parts)

    def _write_index(self):
        summaries = []
        for path in self.root_dir.glob("*/summary.json"):
            try:
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        summaries.sort(key=lambda item: int(item.get("episode_id", -1)))

        cards = []
        for item in summaries:
            episode_id = int(item["episode_id"])
            metrics = item.get("metrics", {})
            pos = metrics.get("position_command_delta_m", {})
            rot = metrics.get("rotation_command_delta_deg", {})
            latency = metrics.get("inference_latency_ms", {})
            cards.append(
                f"""
                <article>
                  <h2>Episode {episode_id}</h2>
                  <p><b>Status:</b> {escape(str(item.get('status', 'unknown')))} &nbsp; <b>Duration:</b> {float(item.get('duration_s', 0.0)):.2f}s</p>
                  <p>Position delta mean/max: {self._format_metric(pos.get('mean'))}/{self._format_metric(pos.get('max'))} m<br>
                     Rotation delta mean/max: {self._format_metric(rot.get('mean'), 2)}/{self._format_metric(rot.get('max'), 2)} deg<br>
                     Latency mean/max: {self._format_metric(latency.get('mean'), 1)}/{self._format_metric(latency.get('max'), 1)} ms</p>
                  <p><a href="{episode_id}/summary.svg">full plot</a> · <a href="{episode_id}/summary.txt">text summary</a> · <a href="{episode_id}/trace.csv">CSV data</a></p>
                  <video controls preload="metadata" src="../videos/{episode_id}/0.mp4"></video>
                  <a href="{episode_id}/summary.svg"><img src="{episode_id}/summary.svg" alt="Episode {episode_id} pose report"></a>
                </article>
                """
            )
        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pose episode comparison</title>
  <style>
    body {{ margin: 0; padding: 32px; background: #eef2f7; color: #172033; font-family: Arial, sans-serif; }}
    h1 {{ margin-top: 0; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 24px; }}
    article {{ background: white; border: 1px solid #cbd5e1; border-radius: 12px; padding: 20px; box-shadow: 0 4px 14px rgba(15, 23, 42, .08); }}
    article img, article video {{ width: 100%; margin-top: 12px; border: 1px solid #e2e8f0; background: #0f172a; }}
    article img {{ max-height: 620px; object-fit: cover; object-position: top; }}
    a {{ color: #2563eb; }}
  </style>
</head>
<body>
  <h1>Pose episode comparison</h1>
  <p>Each card uses the same units. Command deltas are not execution tracking errors.</p>
  <main>{''.join(cards)}</main>
</body>
</html>
"""
        (self.root_dir / "index.html").write_text(document, encoding="utf-8")
