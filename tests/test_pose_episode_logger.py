import csv
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from utils.pose_episode_logger import PoseEpisodeLogger


class PoseEpisodeLoggerTest(unittest.TestCase):
    def test_writes_per_episode_trace_summary_plot_and_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "debug_pose"
            logger = PoseEpisodeLogger(
                root,
                episode_id=3,
                metadata={"checkpoint": "model.ckpt", "task_name": "cup"},
            )
            logger.log_policy_sample(
                iter_idx=0,
                elapsed_s=0.2,
                current_pos=[0.1, 0.2, 0.3],
                current_rot=[0.0, 0.1, 0.2],
                raw_pos=[0.2, 0.3, 0.4],
                raw_rot=[0.1, 0.2, 0.3],
                target_pos=[0.11, 0.22, 0.33],
                target_rot=[0.01, 0.12, 0.23],
                target_gripper_m=0.04,
                position_command_delta_m=0.0374,
                rotation_command_delta_deg=2.5,
                inference_latency_ms=45.0,
            )
            logger.log_policy_sample(
                iter_idx=4,
                elapsed_s=1.0,
                current_pos=[0.11, 0.22, 0.33],
                current_rot=[0.01, 0.12, 0.23],
                raw_pos=[0.3, 0.4, 0.5],
                raw_rot=[0.2, 0.3, 0.4],
                target_pos=[0.12, 0.23, 0.34],
                target_rot=[0.02, 0.13, 0.24],
                target_gripper_m=0.05,
                position_command_delta_m=0.0173,
                rotation_command_delta_deg=1.2,
                inference_latency_ms=55.0,
            )
            logger.finish("manual_stop")
            logger.finish("ignored_second_finish")

            episode_dir = root / "3"
            for name in ["trace.csv", "summary.txt", "summary.json", "summary.svg"]:
                self.assertTrue((episode_dir / name).is_file(), name)
            self.assertTrue((root / "index.html").is_file())

            with (episode_dir / "trace.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["episode_id"], "3")
            self.assertEqual(rows[1]["iter"], "4")

            summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "manual_stop")
            self.assertEqual(summary["samples"], 2)
            self.assertAlmostEqual(
                summary["metrics"]["inference_latency_ms"]["mean"], 50.0
            )
            ET.parse(episode_dir / "summary.svg")
            index_html = (root / "index.html").read_text(encoding="utf-8")
            self.assertIn("Episode 3", index_html)
            self.assertIn("../videos/3/0.mp4", index_html)


if __name__ == "__main__":
    unittest.main()
