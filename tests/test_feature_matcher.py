"""Unit tests for FeatureMatcher (ORB/SIFT) including Unicode paths and transformations."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.feature_matcher import FeatureMatcher


def _make_textured_image(seed: int = 42, size: int = 160) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 240, dtype=np.uint8)
    for _ in range(35):
        x1, y1 = int(rng.integers(10, size - 25)), int(rng.integers(10, size - 25))
        w, h = int(rng.integers(8, 25)), int(rng.integers(8, 25))
        color = tuple(int(c) for c in rng.integers(0, 180, size=3))
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, -1)
    cv2.putText(img, "TARGET", (18, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 3)
    return img


class TestFeatureMatcher(unittest.TestCase):
    """Tests for ORB/SIFT matching, blank frames, transformations, and Korean file paths."""

    def test_orb_matches_identical_and_rejects_blank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Verify Korean + space path works safely on Windows.
            target_path = Path(tmpdir) / "타겟 이미지.png"
            target_img = _make_textured_image()
            ok, encoded = cv2.imencode(".png", target_img)
            self.assertTrue(ok)
            target_path.write_bytes(encoded.tobytes())

            matcher = FeatureMatcher(target_path, method="orb", min_match_count=10)
            res_same = matcher.match(target_img)
            self.assertTrue(res_same.detected)
            self.assertGreaterEqual(res_same.good_matches, 10)
            self.assertGreater(res_same.match_rate, 0.0)

            blank = np.zeros_like(target_img)
            res_blank = matcher.match(blank)
            self.assertFalse(res_blank.detected)
            self.assertEqual(res_blank.good_matches, 0)

    def test_sift_matches_and_occlusion_reduces_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.png"
            target_img = _make_textured_image(seed=7)
            ok, encoded = cv2.imencode(".png", target_img)
            self.assertTrue(ok)
            target_path.write_bytes(encoded.tobytes())

            matcher = FeatureMatcher(target_path, method="sift", min_match_count=10)
            res_front = matcher.match(target_img)
            occluded = FeatureMatcher.transform_image(target_img, occlusion_ratio=0.5)
            res_occ = matcher.match(occluded)

            self.assertTrue(res_front.detected)
            self.assertLess(res_occ.good_matches, res_front.good_matches)

    def test_missing_file_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            FeatureMatcher("non_existent_target_file_12345.png")


if __name__ == "__main__":
    unittest.main()
