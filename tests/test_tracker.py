"""Unit tests for Tracker class (src/tracker.py)."""

from __future__ import annotations

import unittest

from src.detector import Detection
from src.tracker import Tracker


def _det(cx: int, cy: int, size: int = 20) -> Detection:
    half = size // 2
    return Detection(
        bbox=(cx - half, cy - half, size, size),
        centroid=(cx, cy),
        area=float(size * size),
    )


class TestTracker(unittest.TestCase):
    """Tests for ID assignment, distance matching, tie-breaking, disappearance, and trajectories."""

    def test_register_and_maintain_id_with_trajectory(self) -> None:
        tracker = Tracker(max_disappeared=3, max_distance=50.0, max_trajectory_len=10)
        objs1 = tracker.update([_det(50, 50)])
        self.assertEqual(list(objs1.keys()), [1])

        objs2 = tracker.update([_det(60, 55)])
        self.assertEqual(list(objs2.keys()), [1])
        self.assertEqual(list(objs2[1].trajectory), [(50, 50), (60, 55)])

    def test_deregister_after_max_disappeared(self) -> None:
        tracker = Tracker(max_disappeared=2, max_distance=50.0)
        tracker.update([_det(40, 40)])
        self.assertIn(1, tracker.update([]))  # disappeared = 1
        self.assertIn(1, tracker.update([]))  # disappeared = 2
        self.assertNotIn(1, tracker.update([]))  # disappeared = 3 -> removed

    def test_new_id_when_distance_exceeds_threshold(self) -> None:
        tracker = Tracker(max_disappeared=5, max_distance=30.0)
        tracker.update([_det(10, 10)])
        objs = tracker.update([_det(200, 200)])
        self.assertIn(1, objs)
        self.assertIn(2, objs)
        self.assertEqual(objs[1].disappeared, 1)
        self.assertEqual(objs[2].disappeared, 0)

    def test_deterministic_tie_breaking_on_equal_distances(self) -> None:
        tracker = Tracker(max_disappeared=5, max_distance=100.0)
        tracker.update([_det(0, 0), _det(20, 0)])  # ID 1 at (0,0), ID 2 at (20,0)
        # Single detection equidistant (10 px) from both ID 1 and ID 2.
        objs = tracker.update([_det(10, 0)])
        # Lower ID (1) must deterministically win the tie.
        self.assertEqual(objs[1].centroid, (10, 0))
        self.assertEqual(objs[1].disappeared, 0)
        self.assertEqual(objs[2].disappeared, 1)

    def test_velocity_prediction_reduces_crossing_switch(self) -> None:
        # Two objects moving toward each other and crossing at x=50.
        tracker_vel = Tracker(
            max_disappeared=5, max_distance=50.0, use_velocity_prediction=True
        )
        tracker_vel.update([_det(30, 50), _det(70, 50)])  # ID1 -> right, ID2 -> left
        tracker_vel.update([_det(45, 50), _det(55, 50)])  # velocities: +15, -15
        objs = tracker_vel.update([_det(60, 50), _det(40, 50)])  # crossed!
        self.assertEqual(objs[1].centroid, (60, 50))
        self.assertEqual(objs[2].centroid, (40, 50))

    def test_invalid_constructor_args(self) -> None:
        with self.assertRaises(ValueError):
            Tracker(max_disappeared=-1)
        with self.assertRaises(ValueError):
            Tracker(max_distance=0)
        with self.assertRaises(ValueError):
            Tracker(max_trajectory_len=0)


if __name__ == "__main__":
    unittest.main()
