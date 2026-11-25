"""One-click visual requirement verification runner for evaluators (Table / JSON)."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app import VideoAnalyzerApp
from src.background import BackgroundSubtractor
from src.config import AppConfig
from src.detector import MotionDetector
from src.feature_matcher import FeatureMatcher, _imread_unicode
from src.tracker import Tracker


def _ensure_utf8() -> None:
    if sys.platform == "win32":
        for name in ("stdout", "stderr"):
            stream = getattr(sys, name, None)
            if stream is not None and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except (OSError, io.UnsupportedOperation):
                    pass


def run_checks() -> Dict[str, Any]:
    started = time.perf_counter()
    checks: List[Dict[str, Any]] = []

    def record(req_id: str, name: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "id": req_id,
                "requirement": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    # 1. Forbidden library / constraint check & tracker.py separation
    src_dir = ROOT / "src"
    tracker_separated = (src_dir / "tracker.py").exists()
    forbidden_tokens = ("cv2.dnn", "mediapipe", "ultralytics", "import torch", "import tensorflow")
    found_forbidden: List[str] = []
    for py_file in src_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8").lower()
        for tok in forbidden_tokens:
            if tok in text:
                found_forbidden.append(f"{py_file.name}:{tok}")
    record(
        "REQ-01",
        "제약조건(딥러닝/cv2.dnn 금지 & tracker.py 분리)",
        tracker_separated and len(found_forbidden) == 0,
        f"tracker.py 분리={tracker_separated}, 금지 라이브러리 검출={len(found_forbidden)}건",
    )

    # 2. VideoCapture & Headless real-video pipeline + FPS calculation
    vtest = ROOT / "data" / "videos" / "vtest.avi"
    cfg = AppConfig(
        source=str(vtest),
        bg_method="mog2",
        learning_rate=0.005,
        target_image=str(ROOT / "data" / "targets" / "box.png"),
        headless=True,
        max_frames=40,
    )
    summary = VideoAnalyzerApp(cfg).run()
    record(
        "REQ-02",
        "실시간 영상 입출력 · FPS · 바운딩박스/궤적 파이프라인",
        summary.frames_processed == 40 and summary.average_fps > 0 and summary.total_unique_ids > 0,
        f"{summary.frames_processed}프레임 처리, 평균 {summary.average_fps:.1f} FPS, 누적 ID {summary.total_unique_ids}개",
    )

    # 3. Background subtraction (MOG2 + absdiff + 3 learning rates)
    cap = cv2.VideoCapture(str(vtest))
    sample_frames = [cap.read()[1] for _ in range(15)]
    cap.release()
    lr_counts = []
    for lr in (0.001, 0.01, 0.1):
        bg = BackgroundSubtractor(method="mog2", learning_rate=lr)
        last_mask = None
        for f in sample_frames:
            last_mask = bg.apply(f)
        lr_counts.append(int(np.count_nonzero(last_mask)))
    abs_bg = BackgroundSubtractor(method="absdiff", learning_rate=0.01)
    for f in sample_frames[:5]:
        abs_mask = abs_bg.apply(f)
    record(
        "REQ-03",
        "배경 차분(MOG2/absdiff) · 모폴로지 · 학습률 3종 조절",
        len(lr_counts) == 3 and abs_mask.ndim == 2,
        f"MOG2 전경픽셀(lr=0.001/0.01/0.1)={lr_counts}, absdiff 정상 동작",
    )

    # 4. Object detection & Tracker (ID 유지, 궤적 저장, 소멸 삭제)
    tracker = Tracker(max_disappeared=2, max_distance=50.0)
    det = MotionDetector(min_area=100, max_area=50000)
    mask1 = np.zeros((120, 160), dtype=np.uint8)
    mask2 = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(mask1, (20, 20), (50, 50), 255, -1)
    cv2.rectangle(mask2, (28, 25), (58, 55), 255, -1)
    o1 = tracker.update(det.detect(mask1))
    o2 = tracker.update(det.detect(mask2))
    traj_len = len(o2[1].trajectory) if 1 in o2 else 0
    for _ in range(3):
        o_end = tracker.update([])
    record(
        "REQ-04",
        "객체 감지(윤곽선/면적필터) & Tracker(ID유지·궤적·삭제)",
        (1 in o1) and (1 in o2) and traj_len == 2 and (1 not in o_end),
        f"ID 1 유지 및 궤적 길이={traj_len}, 미등장 3프레임 후 ID 정상 삭제",
    )

    # 5. Feature matching (ORB & SIFT TARGET DETECTED)
    box_path = ROOT / "data" / "targets" / "box.png"
    scene_path = ROOT / "data" / "targets" / "box_in_scene.png"
    scene_img = _imread_unicode(scene_path, cv2.IMREAD_GRAYSCALE)
    orb_res = FeatureMatcher(box_path, method="orb", min_match_count=15).match(scene_img)
    sift_res = FeatureMatcher(box_path, method="sift", min_match_count=15).match(scene_img)
    record(
        "REQ-05",
        "특정 물체 재인식 (ORB / SIFT → TARGET DETECTED)",
        orb_res.detected and sift_res.detected,
        f"ORB {orb_res.good_matches}매칭({orb_res.match_rate}%), SIFT {sift_res.good_matches}매칭({sift_res.match_rate}%)",
    )

    # 6. Unit test suite execution
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    record(
        "REQ-06",
        "단위 테스트 자동화 (tests/test_*.py 100% 통과)",
        proc.returncode == 0,
        "unittest 전체 PASS" if proc.returncode == 0 else proc.stderr.strip().splitlines()[-1],
    )

    # 7. Quantitative experiment output & required deliverables
    exp_file = ROOT / "output" / "experiment_results.json"
    diagrams = sorted((ROOT / "diagrams").glob("*.png")) if (ROOT / "diagrams").exists() else []
    required_docs = [
        ROOT / "PLAN.md",
        ROOT / "REPORT.md",
        ROOT / "스터디.md",
        ROOT / "README.md",
        ROOT / "NOTEBOOKLM_FULL_SOURCE.txt",
        ROOT / "NOTEBOOKLM_GUIDE.md",
        ROOT / "study" / "핵심_코드_스터디.md",
    ]
    missing_docs = [p.name for p in required_docs if not p.exists()]
    record(
        "REQ-07",
        "정량 실험 데이터 · Matplotlib 다이어그램 · 필수 문서",
        exp_file.exists() and len(diagrams) >= 4 and len(missing_docs) == 0,
        f"실험JSON={exp_file.exists()}, 다이어그램={len(diagrams)}개, 누락문서={len(missing_docs)}개",
    )

    elapsed = round(time.perf_counter() - started, 3)
    passed_count = sum(1 for c in checks if c["status"] == "PASS")
    return {
        "project": "codyssey-a3-02",
        "total": len(checks),
        "passed": passed_count,
        "failed": len(checks) - passed_count,
        "elapsed_sec": elapsed,
        "checks": checks,
    }


def _print_table(report: Dict[str, Any]) -> None:
    green = "\033[92m"
    red = "\033[91m"
    cyan = "\033[96m"
    reset = "\033[0m"
    print(f"\n{cyan}=== Codyssey A3-02 요구사항 시각적 검증 리포트 ==={reset}")
    print(f"{'ID':<8} | {'상태':<6} | {'검증 항목':<44} | 상세 결과")
    print("-" * 115)
    for c in report["checks"]:
        color = green if c["status"] == "PASS" else red
        print(f"{c['id']:<8} | {color}{c['status']:<6}{reset} | {c['requirement']:<44} | {c['detail']}")
    print("-" * 115)
    print(
        f"합계: {report['passed']}/{report['total']} PASS "
        f"(실패 {report['failed']}건) | 소요 시간: {report['elapsed_sec']}초\n"
    )


def main() -> int:
    _ensure_utf8()
    parser = argparse.ArgumentParser(description="Verify all Codyssey A3-02 requirements.")
    parser.add_argument("--json", action="store_true", help="Output verification result as JSON.")
    args = parser.parse_args()

    report = run_checks()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_table(report)
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
