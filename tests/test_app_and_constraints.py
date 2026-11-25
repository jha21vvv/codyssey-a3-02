"""Unit tests for VideoAnalyzerApp, CLI, visualizer, and strict mission constraints."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.app import VideoAnalyzerApp
from src.config import AppConfig
from src.main import build_parser
from src.visualizer import FrameVisualizer


class TestAppAndConstraints(unittest.TestCase):
    """Mocked VideoCapture tests + static verification of forbidden AI/DNN imports."""

    def test_headless_app_run_with_mock_capture(self) -> None:
        frames = [
            np.full((120, 160, 3), 30, dtype=np.uint8),
            np.full((120, 160, 3), 30, dtype=np.uint8),
            np.full((120, 160, 3), 30, dtype=np.uint8),
        ]
        # Draw a moving white square on frames 2 and 3.
        frames[1][30:60, 30:60] = 255
        frames[2][35:65, 40:70] = 255

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]

        cfg = AppConfig(
            source="mock_video.avi",
            bg_method="absdiff",
            min_area=100,
            target_image=None,
            headless=True,
        )
        with patch("src.app.cv2.VideoCapture", return_value=mock_cap):
            app = VideoAnalyzerApp(cfg)
            summary = app.run()

        self.assertEqual(summary.frames_processed, 3)
        self.assertGreaterEqual(summary.total_unique_ids, 1)
        mock_cap.release.assert_called_once()

    def test_keyboard_controls_pause_snapshot_quit(self) -> None:
        frame = np.full((120, 160, 3), 50, dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = AppConfig(
                source="0",
                target_image=None,
                snapshot_dir=tmpdir,
                headless=False,
            )
            # Key sequence: 'p' (pause), 's' (snapshot), 'q' (quit)
            with (
                patch("src.app.cv2.VideoCapture", return_value=mock_cap),
                patch("src.app.cv2.imshow"),
                patch("src.app.cv2.destroyAllWindows"),
                patch("src.app.cv2.waitKey", side_effect=[ord("p"), ord("s"), ord("q")]),
            ):
                app = VideoAnalyzerApp(cfg)
                summary = app.run()

            self.assertEqual(summary.frames_processed, 1)
            self.assertEqual(summary.snapshots_saved, 1)
            self.assertEqual(len(list(Path(tmpdir).glob("*.png"))), 1)

    def test_snapshot_unicode_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            korean_dir = Path(tmpdir) / "스냅샷 저장소"
            frame = np.zeros((60, 80, 3), dtype=np.uint8)
            saved = FrameVisualizer.save_snapshot(frame, korean_dir, frame_idx=7)
            self.assertTrue(saved.exists())
            self.assertGreater(saved.stat().st_size, 0)

    def test_cli_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--source", "0", "--headless", "--velocity-pred"])
        self.assertEqual(args.source, "0")
        self.assertTrue(args.headless)
        self.assertTrue(args.velocity_pred)

    def test_forbidden_libraries_not_used_in_src(self) -> None:
        """Mission constraint: no YOLO, SSD, R-CNN, cv2.dnn, MediaPipe, torch, tensorflow."""
        forbidden = (
            "cv2.dnn",
            "mediapipe",
            "ultralytics",
            "import torch",
            "import tensorflow",
            "from torch",
            "from tensorflow",
        )
        src_dir = Path(__file__).resolve().parent.parent / "src"
        self.assertTrue((src_dir / "tracker.py").exists(), "tracker.py must be separated.")
        for py_file in src_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(
                    token,
                    content,
                    f"Forbidden token '{token}' found in {py_file.name}",
                )


if __name__ == "__main__":
    unittest.main()
