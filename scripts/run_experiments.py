"""Reproducible quantitative experiments for Codyssey A3-02 report."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.background import BackgroundSubtractor
from src.detector import MotionDetector
from src.feature_matcher import FeatureMatcher, _imread_unicode
from src.tracker import Tracker


def _read_video_frames(path: Path, start: int = 0, count: int = 300) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: List[np.ndarray] = []
    idx = 0
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def measure_feature_matching() -> Dict[str, Any]:
    """Measure ORB and SIFT matching across 5 required conditions + real box_in_scene."""
    target_path = ROOT / "data" / "targets" / "box.png"
    scene_path = ROOT / "data" / "targets" / "box_in_scene.png"

    target_gray = _imread_unicode(target_path, cv2.IMREAD_GRAYSCALE)
    scene_gray = _imread_unicode(scene_path, cv2.IMREAD_GRAYSCALE)
    assert target_gray is not None and scene_gray is not None

    conditions: List[Tuple[str, np.ndarray]] = [
        ("정면", target_gray),
        ("30도 회전", FeatureMatcher.transform_image(target_gray, angle_deg=30.0)),
        ("60도 회전", FeatureMatcher.transform_image(target_gray, angle_deg=60.0)),
        ("30% 가림", FeatureMatcher.transform_image(target_gray, occlusion_ratio=0.30)),
        ("50% 가림", FeatureMatcher.transform_image(target_gray, occlusion_ratio=0.50)),
        ("실제 장면(box_in_scene)", scene_gray),
    ]

    results: Dict[str, List[Dict[str, Any]]] = {"orb": [], "sift": []}
    for method in ("orb", "sift"):
        matcher = FeatureMatcher(
            target_image_path=target_path,
            method=method,
            ratio_thresh=0.75,
            min_match_count=15,
            nfeatures=1000,
        )
        for label, img in conditions:
            res = matcher.match(img)
            results[method].append(
                {
                    "condition": label,
                    "target_keypoints": res.target_keypoints,
                    "frame_keypoints": res.frame_keypoints,
                    "good_matches": res.good_matches,
                    "match_rate": res.match_rate,
                    "detected": res.detected,
                }
            )
    return results


def measure_learning_rates() -> List[Dict[str, Any]]:
    """Compare 3 MOG2 learning rates (0.001, 0.01, 0.1) on vtest.avi + lighting jump + static pause."""
    vtest_frames = _read_video_frames(ROOT / "data" / "videos" / "vtest.avi", start=0, count=250)
    base_bg = vtest_frames[0].copy()
    h, w = base_bg.shape[:2]

    rates = [0.001, 0.01, 0.1]
    summary: List[Dict[str, Any]] = []

    for lr in rates:
        # 1) Average foreground pixel ratio & detection count on real vtest.avi
        bg = BackgroundSubtractor(method="mog2", learning_rate=lr, morph_kernel_size=5)
        det = MotionDetector(min_area=500, max_area=100000)
        fg_ratios: List[float] = []
        det_counts: List[int] = []
        for idx, frame in enumerate(vtest_frames):
            mask = bg.apply(frame)
            if idx >= 30:
                fg_ratios.append(float(np.count_nonzero(mask)) / mask.size * 100.0)
                det_counts.append(len(det.detect(mask)))

        # 2) Frames until a stopped object is absorbed into background
        bg_stop = BackgroundSubtractor(method="mog2", learning_rate=lr, morph_kernel_size=5)
        for _ in range(40):
            bg_stop.apply(base_bg)
        absorbed_frame = None
        stopped_frame = base_bg.copy()
        cv2.rectangle(stopped_frame, (w // 2 - 25, h // 2 - 40), (w // 2 + 25, h // 2 + 40), (30, 200, 240), -1)
        for f_idx in range(1, 201):
            mask = bg_stop.apply(stopped_frame)
            if len(det.detect(mask)) == 0:
                absorbed_frame = f_idx
                break

        # 3) Frames to recover after sudden +60 brightness jump
        bg_light = BackgroundSubtractor(method="mog2", learning_rate=lr, morph_kernel_size=5)
        for _ in range(40):
            bg_light.apply(base_bg)
        bright_bg = cv2.convertScaleAbs(base_bg, alpha=1.0, beta=60)
        peak_fg_pct = 0.0
        recovery_frame = None
        for f_idx in range(1, 201):
            mask = bg_light.apply(bright_bg)
            fg_pct = float(np.count_nonzero(mask)) / mask.size * 100.0
            if f_idx == 1:
                peak_fg_pct = round(fg_pct, 2)
            if fg_pct < 2.0 and recovery_frame is None:
                recovery_frame = f_idx
                break

        summary.append(
            {
                "learning_rate": lr,
                "vtest_avg_fg_pct": round(float(np.mean(fg_ratios)), 2),
                "vtest_avg_detections": round(float(np.mean(det_counts)), 2),
                "stopped_object_absorbed_frames": absorbed_frame if absorbed_frame is not None else ">200",
                "lighting_jump_peak_fg_pct": peak_fg_pct,
                "lighting_recovery_frames": recovery_frame if recovery_frame is not None else ">200",
            }
        )
    return summary


def _measure_single_object_trials() -> Dict[str, Any]:
    """10 trials of single-object movement on real one-by-one-person-detection.mp4 episodes."""
    video_path = ROOT / "data" / "videos" / "one-by-one-person-detection.mp4"
    frames = _read_video_frames(video_path, start=0, count=1390)
    bg = BackgroundSubtractor(method="mog2", learning_rate=0.005, morph_kernel_size=5)
    det = MotionDetector(min_area=800, max_area=120000)
    tracker = Tracker(max_disappeared=15, max_distance=100.0)

    # Record active IDs per frame after 30-frame background warmup.
    per_frame_ids: List[List[int]] = []
    for frame in frames:
        mask = bg.apply(frame)
        detections = det.detect(mask)
        tracked = tracker.update(detections)
        active = [oid for oid, obj in tracked.items() if obj.disappeared == 0]
        per_frame_ids.append(active)

    # Segment contiguous episodes where a single person enters and walks through the corridor
    # (ignoring empty corridor gaps > 15 frames where no person is present).
    episodes: List[Tuple[int, int]] = []
    in_ep = False
    ep_start = 0
    gap = 0
    for idx in range(30, len(per_frame_ids)):
        if len(per_frame_ids[idx]) > 0:
            if not in_ep:
                in_ep = True
                ep_start = idx
            gap = 0
        elif in_ep:
            gap += 1
            if gap > 15:
                ep_end = idx - gap
                if ep_end - ep_start >= 15:
                    episodes.append((ep_start, ep_end))
                in_ep = False
    if in_ep and (len(per_frame_ids) - 1 - ep_start) >= 15:
        episodes.append((ep_start, len(per_frame_ids) - 1))

    # Select up to 10 episodes (or split longer episodes if fewer than 10).
    selected = episodes[:10]
    while len(selected) < 10 and selected:
        longest_idx = max(range(len(selected)), key=lambda i: selected[i][1] - selected[i][0])
        s, e = selected.pop(longest_idx)
        mid = (s + e) // 2
        selected.insert(longest_idx, (s, mid))
        selected.insert(longest_idx + 1, (mid + 1, e))

    trials: List[Dict[str, Any]] = []
    total_switches = 0
    total_failures = 0

    for start_f, end_f in selected[:10]:
        window = per_frame_ids[start_f : end_f + 1]
        primary_id = None
        switches = 0
        consecutive_miss = 0
        max_miss = 0
        new_id_streak = 0
        candidate_id = None

        for active in window:
            if len(active) == 0:
                consecutive_miss += 1
                max_miss = max(max_miss, consecutive_miss)
                continue
            consecutive_miss = 0
            if primary_id is None:
                primary_id = active[0]
            elif primary_id in active:
                new_id_streak = 0
                candidate_id = None
            else:
                if candidate_id == active[0]:
                    new_id_streak += 1
                else:
                    candidate_id = active[0]
                    new_id_streak = 1
                if new_id_streak >= 5:
                    switches += 1
                    primary_id = candidate_id
                    new_id_streak = 0

        failed = 1 if max_miss >= 10 else 0
        total_switches += switches
        total_failures += failed
        trials.append(
            {
                "video": "one-by-one-person-detection.mp4",
                "condition": "단일 객체 이동",
                "frame_range": f"{start_f}-{end_f}",
                "id_switches": switches,
                "failures": failed,
                "note": f"단일 보행자 구간 (최대 연속 미검출 {max_miss}프레임)",
            }
        )

    return {
        "condition": "단일 객체 이동",
        "tests": len(trials),
        "id_switches": total_switches,
        "failures": total_failures,
        "trials": trials,
    }


def _measure_crossing_trials(use_velocity_prediction: bool) -> Dict[str, Any]:
    """10 controlled crossing trials on real vtest.avi background to measure exact ID switches."""
    vtest_frames = _read_video_frames(ROOT / "data" / "videos" / "vtest.avi", start=0, count=20)
    real_bg = vtest_frames[0]
    h, w = real_bg.shape[:2]

    # 10 realistic crossing variations: head-on, diagonal, slight pause at center, unequal speeds,
    # sharp turn after crossing (where even velocity prediction fails!), close parallel pass.
    configs = [
        ("정면 수평 교차 (등속)", (120, 280), (8, 0), (640, 280), (-8, 0), False, False),
        ("대각선 X자 교차", (140, 180), (7, 3), (620, 380), (-7, -3), False, False),
        ("비대칭 속도 교차", (100, 260), (10, 0), (640, 260), (-6, 0), False, False),
        ("교차점 3프레임 겹침 정체", (140, 290), (7, 0), (620, 290), (-7, 0), True, False),
        ("교차 직후 방향 전환 A", (140, 280), (7, 0), (620, 280), (-7, 0), False, True),
        ("교차 직후 방향 전환 B", (140, 220), (6, 2), (620, 340), (-6, -2), False, True),
        ("완만한 대각선 교차", (120, 240), (7, 1), (640, 320), (-7, -1), False, False),
        ("교차점 정체 + 방향 전환", (150, 280), (6, 0), (610, 280), (-6, 0), True, True),
        ("고속 수평 교차", (80, 300), (11, 0), (680, 300), (-11, 0), False, False),
        ("저속 밀착 교차", (180, 280), (5, 0), (580, 280), (-5, 0), True, False),
    ]

    total_switches = 0
    total_failures = 0
    trials: List[Dict[str, Any]] = []

    for idx, (label, p1_start, v1, p2_start, v2, pause_mid, turn_after) in enumerate(configs, 1):
        bg = BackgroundSubtractor(method="mog2", learning_rate=0.002, morph_kernel_size=5)
        det = MotionDetector(min_area=400, max_area=120000)
        tracker = Tracker(
            max_disappeared=15,
            max_distance=85.0,
            use_velocity_prediction=use_velocity_prediction,
        )
        for _ in range(25):
            bg.apply(real_bg)

        x1, y1 = float(p1_start[0]), float(p1_start[1])
        x2, y2 = float(p2_start[0]), float(p2_start[1])
        vx1, vy1 = float(v1[0]), float(v1[1])
        vx2, vy2 = float(v2[0]), float(v2[1])

        id_a = None
        id_b = None
        crossed = False
        post_cross_frames = 0
        wrong_id_frames = 0
        miss_streak = 0
        max_miss = 0

        for f in range(1, 75):
            if pause_mid and abs(x1 - x2) < 18 and f % 2 == 0:
                pass
            else:
                x1 += vx1
                y1 += vy1
                x2 += vx2
                y2 += vy2

            if not crossed and x1 > x2:
                crossed = True
                if turn_after:
                    # Both people bounce/turn back after meeting — defeats constant velocity!
                    vx1, vx2 = -vx1, -vx2

            frame = real_bg.copy()
            cv2.rectangle(
                frame,
                (int(x1) - 18, int(y1) - 34),
                (int(x1) + 18, int(y1) + 34),
                (30, 180, 255),
                -1,
            )
            cv2.rectangle(
                frame,
                (int(x2) - 18, int(y2) - 34),
                (int(x2) + 18, int(y2) + 34),
                (255, 80, 80),
                -1,
            )

            mask = bg.apply(frame)
            detections = det.detect(mask)
            tracked = tracker.update(detections)

            # Identify initial IDs for A (left) and B (right) before crossing.
            if id_a is None and len(tracked) >= 2:
                sorted_objs = sorted(tracked.values(), key=lambda o: o.centroid[0])
                id_a = sorted_objs[0].object_id
                id_b = sorted_objs[-1].object_id

            # Check if A or B is detected.
            if not crossed:
                continue

            post_cross_frames += 1
            # After separation (distance > 45px), verify which tracked ID is closest to true (x1, y1).
            if abs(x1 - x2) > 45 and id_a is not None:
                best_oid = None
                best_dist = 9999.0
                for oid, obj in tracked.items():
                    if obj.disappeared > 0:
                        continue
                    d = math.hypot(obj.centroid[0] - x1, obj.centroid[1] - y1)
                    if d < best_dist:
                        best_dist = d
                        best_oid = oid
                if best_oid is None or best_dist > 50.0:
                    miss_streak += 1
                    max_miss = max(max_miss, miss_streak)
                else:
                    miss_streak = 0
                    if best_oid != id_a:
                        wrong_id_frames += 1

        switched = 1 if wrong_id_frames >= 5 else 0
        failed = 1 if max_miss >= 10 else 0
        total_switches += switched
        total_failures += failed
        trials.append(
            {
                "video": f"vtest_crossing_case_{idx:02d}",
                "condition": "두 객체 겹침",
                "frame_range": "1-74",
                "id_switches": switched,
                "failures": failed,
                "note": f"{label} (겹침 후 오매칭 {wrong_id_frames}프레임)",
            }
        )

    return {
        "condition": "두 객체 겹침",
        "use_velocity_prediction": use_velocity_prediction,
        "tests": 10,
        "id_switches": total_switches,
        "failures": total_failures,
        "trials": trials,
    }


def _measure_real_vtest_and_people_summary() -> List[Dict[str, Any]]:
    """Run tracker directly on real vtest.avi and people-detection.mp4 for observational logs."""
    logs: List[Dict[str, Any]] = []
    for fname, max_f in (("vtest.avi", 300), ("people-detection.mp4", 300)):
        frames = _read_video_frames(ROOT / "data" / "videos" / fname, start=0, count=max_f)
        bg = BackgroundSubtractor(method="mog2", learning_rate=0.005, morph_kernel_size=5)
        det = MotionDetector(min_area=500, max_area=100000)
        tracker = Tracker(max_disappeared=15, max_distance=80.0)
        max_concurrent = 0
        for frame in frames:
            tracked = tracker.update(det.detect(bg.apply(frame)))
            active = sum(1 for o in tracked.values() if o.disappeared == 0)
            max_concurrent = max(max_concurrent, active)
        logs.append(
            {
                "video": fname,
                "frames": len(frames),
                "total_unique_ids": tracker.total_registered,
                "max_concurrent_objects": max_concurrent,
            }
        )
    return logs


def _measure_pause_trials() -> Dict[str, Any]:
    """5 trials of temporary object stop (30~60 frames) on real background with MOG2."""
    base_bg = _read_video_frames(
        ROOT / "data" / "videos" / "one-by-one-person-detection.mp4", start=0, count=1
    )[0]
    pause_durations = [15, 25, 40, 55, 70]
    total_switches = 0
    total_failures = 0
    trials: List[Dict[str, Any]] = []

    for idx, pause_len in enumerate(pause_durations, 1):
        bg = BackgroundSubtractor(method="mog2", learning_rate=0.03, morph_kernel_size=5)
        det = MotionDetector(min_area=400, max_area=120000)
        tracker = Tracker(max_disappeared=15, max_distance=80.0)
        for _ in range(25):
            bg.apply(base_bg)

        x, y = 120, 220
        initial_id = None
        miss_streak = 0
        max_miss = 0
        id_changed_frames = 0

        for f in range(1, 40 + pause_len + 40):
            if f <= 30 or f > 30 + pause_len:
                x += 6
            frame = base_bg.copy()
            cv2.rectangle(frame, (x - 20, y - 38), (x + 20, y + 38), (40, 220, 120), -1)
            tracked = tracker.update(det.detect(bg.apply(frame)))
            active = [oid for oid, o in tracked.items() if o.disappeared == 0]
            if not active:
                miss_streak += 1
                max_miss = max(max_miss, miss_streak)
            else:
                miss_streak = 0
                if initial_id is None:
                    initial_id = active[0]
                elif initial_id not in active:
                    id_changed_frames += 1

        failed = 1 if max_miss >= 10 else 0
        switched = 1 if id_changed_frames >= 5 else 0
        total_failures += failed
        total_switches += switched
        trials.append(
            {
                "video": f"one-by-one_pause_case_{idx:02d}",
                "condition": "객체 일시 정지",
                "frame_range": f"1-{80 + pause_len}",
                "id_switches": switched,
                "failures": failed,
                "note": f"{pause_len}프레임 정지 (최대 연속 미검출 {max_miss}프레임)",
            }
        )

    return {
        "condition": "객체 일시 정지",
        "tests": 5,
        "id_switches": total_switches,
        "failures": total_failures,
        "trials": trials,
    }


def _measure_lighting_trials() -> Dict[str, Any]:
    """5 trials of lighting changes (+65, -55, +40, flicker, gradual) on real vtest.avi."""
    vtest_frames = _read_video_frames(ROOT / "data" / "videos" / "vtest.avi", start=0, count=220)
    scenarios = [
        ("급격한 조명 증가 (+65)", "step_up", 65),
        ("급격한 조명 감소 (-55)", "step_down", -55),
        ("중간 조명 증가 (+40)", "step_up", 40),
        ("조명 깜빡임 (Flicker)", "flicker", 50),
        ("점진적 일조 변화 (+1/f)", "gradual", 50),
    ]

    total_switches = 0
    total_failures = 0
    trials: List[Dict[str, Any]] = []

    for idx, (label, mode, delta) in enumerate(scenarios, 1):
        bg = BackgroundSubtractor(method="mog2", learning_rate=0.005, morph_kernel_size=5)
        det = MotionDetector(min_area=500, max_area=100000)
        tracker = Tracker(max_disappeared=15, max_distance=80.0)

        collapse_streak = 0
        max_collapse = 0
        id_burst = 0
        pre_ids = 0

        for f_idx, orig in enumerate(vtest_frames):
            frame = orig.copy()
            if f_idx >= 80:
                if mode in ("step_up", "step_down"):
                    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=delta)
                elif mode == "flicker" and (f_idx % 6 < 3):
                    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=delta)
                elif mode == "gradual":
                    shift = min(delta, (f_idx - 80) * 1.0)
                    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=shift)

            mask = bg.apply(frame)
            fg_pct = float(np.count_nonzero(mask)) / mask.size * 100.0
            detections = det.detect(mask)
            tracker.update(detections)

            if f_idx == 79:
                pre_ids = tracker.total_registered
            if f_idx >= 80:
                # Foreground mask collapse: >30% of entire frame falsely classified as moving foreground
                # or all valid contours merged into a single giant blob (>max_area -> 0 detections).
                if fg_pct > 30.0 or len(detections) == 0:
                    collapse_streak += 1
                    max_collapse = max(max_collapse, collapse_streak)
                else:
                    collapse_streak = 0
                if f_idx == 110:
                    id_burst = tracker.total_registered - pre_ids

        failed = 1 if max_collapse >= 10 else 0
        switched = 1 if id_burst >= 8 else 0
        total_failures += failed
        total_switches += switched
        trials.append(
            {
                "video": f"vtest_lighting_case_{idx:02d}",
                "condition": "조명 변화",
                "frame_range": "1-220 (변화: 80~)",
                "id_switches": switched,
                "failures": failed,
                "note": f"{label} (붕괴 지속 {max_collapse}프레임, 30프레임 내 신규ID +{id_burst})",
            }
        )

    return {
        "condition": "조명 변화",
        "tests": 5,
        "id_switches": total_switches,
        "failures": total_failures,
        "trials": trials,
    }


def main() -> int:
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "tracking": {
            "single_object": _measure_single_object_trials(),
            "two_objects_crossing_baseline": _measure_crossing_trials(use_velocity_prediction=False),
            "two_objects_crossing_velocity_rule": _measure_crossing_trials(use_velocity_prediction=True),
            "object_pause": _measure_pause_trials(),
            "lighting_change": _measure_lighting_trials(),
            "real_video_observations": _measure_real_vtest_and_people_summary(),
        },
        "learning_rates": measure_learning_rates(),
        "feature_matching": measure_feature_matching(),
    }

    out_path = out_dir / "experiment_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved experiment results to {out_path}")
    print("\n--- Tracking Summary ---")
    for key in ("single_object", "two_objects_crossing_baseline", "two_objects_crossing_velocity_rule", "object_pause", "lighting_change"):
        row = results["tracking"][key]
        print(f"{key}: tests={row['tests']}, id_switches={row['id_switches']}, failures={row['failures']}")
    print("\n--- Learning Rates ---")
    for row in results["learning_rates"]:
        print(row)
    print("\n--- Feature Matching (ORB) ---")
    for row in results["feature_matching"]["orb"]:
        print(row)
    print("\n--- Feature Matching (SIFT) ---")
    for row in results["feature_matching"]["sift"]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
