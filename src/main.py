"""Command-line entry point for the real-time motion tracking and re-ID application."""

# [1차]: 타입 힌트 지연 평가를 활성화합니다.
# [2차]: 타입 이름표를 깔끔하게 처리합니다.
from __future__ import annotations

# [1차]: CLI 명령행 인자 파싱을 위해 표준 라이브러리 argparse를 임포트합니다.
# [2차]: 터미널에서 사용자가 입력한 옵션(--source, --learning-rate 등)을 읽어주는 주문서 해석기입니다.
import argparse

# [1차]: 스트림 재구성 예외(UnsupportedOperation) 처리를 위해 io 모듈을 임포트합니다.
# [2차]: 특수 터미널 환경에서 인코딩 변경이 안 될 때를 대비하는 안전장치입니다.
import io

# [1차]: 플랫폼 확인(sys.platform)과 표준 입출력 스트림 접근을 위해 sys 모듈을 임포트합니다.
# [2차]: 지금 실행 중인 컴퓨터가 윈도우인지 확인하는 시스템 정보창입니다.
import sys

# [1차]: Optional과 Sequence 타입 힌트를 임포트합니다.
# [2차]: 테스트 시 인자 목록을 직접 넣거나 비워둘 수 있음을 표시합니다.
from typing import Optional, Sequence

# [1차]: 애플리케이션 본체와 설정 데이터클래스를 임포트합니다.
# [2차]: 주문서를 넘겨줄 감독관(VideoAnalyzerApp)과 설정표(AppConfig)를 불러옵니다.
from src.app import VideoAnalyzerApp
from src.config import AppConfig


# [1차]: Windows 콘솔(cp949)에서 한글 출력 시 UnicodeEncodeError가 발생하지 않도록 UTF-8로 재설정합니다.
# [2차]: 윈도우 터미널이 한글을 깨뜨리지 않도록 입구에서 통역기(UTF-8)를 켜주는 방어 코드입니다.
def _ensure_utf8_console() -> None:
    """Protect Windows terminals from cp949 UnicodeEncodeError when printing Korean text."""
    if sys.platform == "win32":
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name, None)
            if stream is not None and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except (OSError, io.UnsupportedOperation):
                    pass


# [1차]: 미션의 모든 기능 옵션(영상 소스, 배경 차분, 학습률, 추적기, 특징점 매칭, Headless)을 정의하는 파서를 반환합니다.
# [2차]: 사용자가 터미널에서 고를 수 있는 모든 스위치와 다이얼을 메뉴판에 등록합니다.
def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser covering all mission requirements."""
    parser = argparse.ArgumentParser(
        description="Codyssey A3-02: Classic CV Real-Time Motion Detection, Tracking & Re-ID"
    )
    parser.add_argument(
        "--source",
        default="data/videos/vtest.avi",
        help="Video file path or webcam index (e.g. '0').",
    )
    parser.add_argument(
        "--bg-method",
        choices=("mog2", "absdiff"),
        default="mog2",
        help="Background subtraction algorithm.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.005,
        help="Background model learning rate (-1 for auto, or 0.0~1.0).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum contour area in pixels to filter noise.",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=100000,
        help="Maximum contour area in pixels.",
    )
    parser.add_argument(
        "--max-disappeared",
        type=int,
        default=15,
        help="Frames to keep an object ID before deregistration.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=80.0,
        help="Maximum Euclidean distance (px) for ID association.",
    )
    parser.add_argument(
        "--velocity-pred",
        action="store_true",
        help="Enable constant-velocity prediction rule for centroid matching.",
    )
    parser.add_argument(
        "--target",
        default="data/targets/box.png",
        help="Path to target object image for ORB/SIFT re-identification (empty to disable).",
    )
    parser.add_argument(
        "--feature-method",
        choices=("orb", "sift"),
        default="orb",
        help="Feature descriptor algorithm for target re-ID.",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=15,
        help="Minimum good feature matches required to trigger TARGET DETECTED.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="output/snapshots",
        help="Directory to store snapshots when pressing 's'.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window (useful for batch processing and verification).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing N frames.",
    )
    parser.add_argument(
        "--save-video",
        default=None,
        help="Optional output video path (.mp4) to save rendered tracking results.",
    )
    return parser


# [1차]: CLI 인자를 파싱해 AppConfig를 생성하고 VideoAnalyzerApp을 실행한 뒤 종료 코드 0을 반환합니다.
# [2차]: 터미널 주문서를 해석해 감독관에게 전달하고, 작업이 끝나면 최종 요약 영수증을 출력합니다.
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse CLI arguments, execute VideoAnalyzerApp, and print summary."""
    _ensure_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)

    config = AppConfig(
        source=args.source,
        bg_method=args.bg_method,
        learning_rate=args.learning_rate,
        min_area=args.min_area,
        max_area=args.max_area,
        max_disappeared=args.max_disappeared,
        max_distance=args.max_distance,
        use_velocity_prediction=args.velocity_pred,
        target_image=args.target if args.target else None,
        feature_method=args.feature_method,
        min_match_count=args.min_matches,
        snapshot_dir=args.snapshot_dir,
        headless=args.headless,
        max_frames=args.max_frames,
        save_video=args.save_video,
    )

    app = VideoAnalyzerApp(config)
    summary = app.run()
    print(
        f"[완료] 처리 프레임: {summary.frames_processed} | "
        f"평균 FPS: {summary.average_fps:.2f} | "
        f"누적 부여 ID: {summary.total_unique_ids} | "
        f"타겟 감지 프레임: {summary.target_detected_frames} | "
        f"스냅샷 저장: {summary.snapshots_saved}"
    )
    return 0


# [1차]: 스크립트가 직접 실행(`python -m src.main`)될 때만 main()을 호출하고 반환값으로 프로세스를 종료합니다.
# [2차]: 다른 파일에서 부품으로 불러올 때는 조용히 있고, 직접 실행 버튼을 눌렀을 때만 시동을 겁니다.
if __name__ == "__main__":
    raise SystemExit(main())
