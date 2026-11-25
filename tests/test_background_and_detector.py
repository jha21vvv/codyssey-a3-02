"""Unit tests for BackgroundSubtractor and MotionDetector."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.background import BackgroundSubtractor
from src.detector import MotionDetector


class TestBackgroundAndDetector(unittest.TestCase):
    """Fast synthetic-frame tests for background subtraction and contour detection."""

    @staticmethod
    def _make_frame(box: tuple[int, int, int, int] | None = None) -> np.ndarray:
        frame = np.full((180, 240, 3), 40, dtype=np.uint8)
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (240, 240, 240), -1)
        return frame

    def test_mog2_detects_moving_object(self) -> None:
        bg = BackgroundSubtractor(method="mog2", learning_rate=0.05, morph_kernel_size=3)
        detector = MotionDetector(min_area=200, max_area=20000)

        # Warm up static background.
        for _ in range(10):
            bg.apply(self._make_frame())

        mask = bg.apply(self._make_frame((60, 50, 40, 40)))
        detections = detector.detect(mask)
        self.assertEqual(len(detections), 1)
        cx, cy = detections[0].centroid
        self.assertAlmostEqual(cx, 80, delta=6)
        self.assertAlmostEqual(cy, 70, delta=6)

    def test_absdiff_detects_moving_object(self) -> None:
        bg = BackgroundSubtractor(method="absdiff", learning_rate=0.01, morph_kernel_size=3)
        detector = MotionDetector(min_area=200, max_area=20000)

        first_mask = bg.apply(self._make_frame())
        self.assertEqual(int(first_mask.sum()), 0)

        mask = bg.apply(self._make_frame((100, 60, 35, 35)))
        detections = detector.detect(mask)
        self.assertEqual(len(detections), 1)
        self.assertGreaterEqual(detections[0].area, 900)

    def test_area_filtering_ignores_small_noise(self) -> None:
        mask = np.zeros((180, 240), dtype=np.uint8)
        # Small 5x5 noise (area ~16) and valid 30x30 object (area ~841).
        cv2.rectangle(mask, (10, 10), (15, 15), 255, -1)
        cv2.rectangle(mask, (100, 80), (130, 110), 255, -1)

        detector = MotionDetector(min_area=200, max_area=5000)
        detections = detector.detect(mask)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox[:2], (100, 80))

    def test_invalid_inputs_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BackgroundSubtractor(method="invalid")
        with self.assertRaises(ValueError):
            BackgroundSubtractor(morph_kernel_size=4)
        with self.assertRaises(ValueError):
            BackgroundSubtractor(learning_rate=1.5)
        with self.assertRaises(ValueError):
            MotionDetector(min_area=0)

        bg = BackgroundSubtractor()
        with self.assertRaises(ValueError):
            bg.apply(np.array([], dtype=np.uint8))

        detector = MotionDetector()
        with self.assertRaises(ValueError):
            detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
