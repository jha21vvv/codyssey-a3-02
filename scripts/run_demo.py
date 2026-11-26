"""Generate and optionally play an evaluator-ready demo showcasing tracking + TARGET DETECTED."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app import VideoAnalyzerApp
from src.config import AppConfig
from src.feature_matcher import _imread_unicode


def _ensure_utf8() -> None:
    if sys.platform == "win32":
        for name in ("stdout", "stderr"):
            stream = getattr(sys, name, None)
            if stream is not None and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except (OSError, io.UnsupportedOperation):
                    pass


def _save_png_unicode(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    path.write_bytes(encoded.tobytes())


def run_demo(gui: bool = False, max_frames: int = 120) -> None:
    vtest_path = ROOT / "data" / "videos" / "vtest.avi"
    box_path = ROOT / "data" / "targets" / "box.png"
    out_video_path = ROOT / "output" / "demo_combined.mp4"
    snap_tracking = ROOT / "diagrams" / "demo_tracking_vtest.png"
    snap_target = ROOT / "diagrams" / "demo_target_detected.png"

    box_bgr = _imread_unicode(box_path, cv2.IMREAD_COLOR)
    if box_bgr is None:
        raise FileNotFoundError(f"Target image not found: {box_path}")
    # Scale target box cleanly for natural placement in 768x576 scene.
    box_small = cv2.resize(box_bgr, (180, 125), interpolation=cv2.INTER_AREA)
    bh, bw = box_small.shape[:2]

    # 1) Pure vtest.avi multi-pedestrian tracking snapshot
    cap = cv2.VideoCapture(str(vtest_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {vtest_path}")

    app_track = VideoAnalyzerApp(
        AppConfig(
            source=str(vtest_path),
            bg_method="mog2",
            learning_rate=0.005,
            min_area=500,
            target_image=None,
            headless=True,
        )
    )

    frames = []
    for idx in range(max_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
        rendered, _ = app_track.process_frame(frame, fps=24.5, paused=False)
        if idx == 85:
            _save_png_unicode(snap_tracking, rendered)
    cap.release()

    # 2) Combined demo: real vtest.avi + moving target box -> Tracking + TARGET DETECTED simultaneously!
    app_combo = VideoAnalyzerApp(
        AppConfig(
            source=str(vtest_path),
            bg_method="mog2",
            learning_rate=0.005,
            min_area=500,
            target_image=str(box_path),
            feature_method="orb",
            min_match_count=15,
            headless=not gui,
        )
    )

    h, w = frames[0].shape[:2]
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h)
    )

    saved_target_snap = False
    prev_t = time.perf_counter()
    try:
        for idx, base in enumerate(frames):
            frame = base.copy()
            # Slide target box smoothly across frames 45..110 so it triggers BOTH tracker & ORB Re-ID.
            if 45 <= idx <= 110:
                progress = (idx - 45) / 65.0
                x0 = int(60 + progress * (w - bw - 140))
                y0 = int(320 - 40 * np.sin(progress * np.pi))
                frame[y0 : y0 + bh, x0 : x0 + bw] = box_small

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            rendered, match_res = app_combo.process_frame(frame, fps=min(fps, 60.0), paused=False)
            writer.write(rendered)

            if not saved_target_snap and idx >= 80 and match_res is not None and match_res.detected:
                _save_png_unicode(snap_target, rendered)
                saved_target_snap = True

            if gui:
                cv2.imshow("Codyssey A3-02 | Evaluator Live Demo (q: Quit)", rendered)
                if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        writer.release()
        if gui:
            cv2.destroyAllWindows()

    print(f"[OK] 순수 다중 추적 시연 스냅샷: {snap_tracking.relative_to(ROOT)}")
    print(f"[OK] 추적 + TARGET DETECTED 통합 스냅샷: {snap_target.relative_to(ROOT)}")
    print(f"[OK] 통합 시연 영상(MP4): {out_video_path.relative_to(ROOT)}")


def main() -> int:
    _ensure_utf8()
    parser = argparse.ArgumentParser(description="Run evaluator demo for Codyssey A3-02.")
    parser.add_argument("--gui", action="store_true", help="Open live OpenCV demo window.")
    parser.add_argument("--max-frames", type=int, default=120, help="Frames to render.")
    args = parser.parse_args()
    run_demo(gui=args.gui, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
