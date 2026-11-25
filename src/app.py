"""Real-time video analysis application loop with keyboard controls and headless mode."""

# [1차]: 타입 힌트 지연 평가를 활성화합니다.
# [2차]: 타입 이름표를 유연하게 참조할 수 있게 합니다.
from __future__ import annotations

# [1차]: 실행 요약 통계를 담는 dataclass를 임포트합니다.
# [2차]: 영상 분석이 끝난 뒤 성적표를 담아둘 결과 양식입니다.
from dataclasses import dataclass

# [1차]: 타겟 이미지 및 출력 비디오 경로 확인을 위해 Path를 임포트합니다.
# [2차]: 파일이 존재하는지, 저장할 폴더가 있는지 관리하는 도구입니다.
from pathlib import Path

# [1차]: 고정밀 단조 시계(time.perf_counter)로 실시간 FPS를 계산하기 위해 time 모듈을 임포트합니다.
# [2차]: 1프레임 처리에 몇 밀리초가 걸렸는지 재는 초정밀 스톱워치입니다.
import time

# [1차]: Optional 타입 힌트를 임포트합니다.
# [2차]: 비어 있을 수 있는 변수(예: FeatureMatcher, VideoWriter)를 표시합니다.
from typing import Optional

# [1차]: VideoCapture, VideoWriter, GUI 창 표시 및 키보드 이벤트 처리를 위해 OpenCV를 임포트합니다.
# [2차]: 영상을 읽고 화면에 띄우고 키보드 입력을 받는 재생기 본체입니다.
import cv2

# [1차]: 프레임 타입 힌트를 위해 NumPy를 임포트합니다.
# [2차]: 영상 1장이 숫자 배열임을 표시합니다.
import numpy as np

# [1차]: SRP로 분리된 각 하위 파이프라인 모듈을 임포트합니다.
# [2차]: 배경 담당, 감지 담당, 추적 담당, 지문 감식 담당, 화가 팀원들을 한자리에 모읍니다.
from src.background import BackgroundSubtractor
from src.config import AppConfig
from src.detector import MotionDetector
from src.feature_matcher import FeatureMatcher, MatchResult
from src.tracker import Tracker
from src.visualizer import FrameVisualizer


# [1차]: 영상 분석 실행 완료 후 핵심 집계 지표를 반환하는 RunSummary 데이터클래스입니다.
# [2차]: 상영이 끝난 뒤 "총 몇 프레임, 평균 FPS, 누적 ID, 타겟 감지 횟수, 스냅샷 수"를 정리한 영수증입니다.
@dataclass
class RunSummary:
    """Summary metrics collected during a video analysis run."""

    frames_processed: int
    average_fps: float
    total_unique_ids: int
    target_detected_frames: int
    snapshots_saved: int


# [1차]: 전체 실시간 영상 분석 루프를 총괄하는 오케스트레이터 클래스 VideoAnalyzerApp입니다.
# [2차]: 모든 팀원(배경·감지·추적·감식·화가)에게 순서대로 일을 지휘하는 감독관입니다.
class VideoAnalyzerApp:
    """Orchestrates VideoCapture, background subtraction, detection, tracking, and feature matching."""

    # [1차]: OpenCV GUI 창에 표시할 고정 타이틀 문자열입니다.
    # [2차]: 모니터에 뜨는 재생 창 맨 위 제목표시줄 글자입니다.
    WINDOW_NAME = "Codyssey A3-02 | Motion Tracker & Re-ID"

    # [1차]: AppConfig 설정을 받아 5개 하위 컴포넌트(배경·감지·추적·시각화·특징점매칭)를 조립합니다.
    # [2차]: 설정표를 보고 각 담당 팀원들에게 알맞은 장비를 나눠주며 출격 준비를 마칩니다.
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.bg_subtractor = BackgroundSubtractor(
            method=config.bg_method,
            learning_rate=config.learning_rate,
            history=config.history,
            var_threshold=config.var_threshold,
            morph_kernel_size=config.morph_kernel_size,
        )
        self.detector = MotionDetector(
            min_area=config.min_area,
            max_area=config.max_area,
        )
        self.tracker = Tracker(
            max_disappeared=config.max_disappeared,
            max_distance=config.max_distance,
            max_trajectory_len=config.max_trajectory_len,
            use_velocity_prediction=config.use_velocity_prediction,
        )
        self.visualizer = FrameVisualizer()
        # [1차]: target_image가 설정되어 있고 파일이 실제로 존재할 때만 FeatureMatcher를 활성화합니다.
        # [2차]: 수배 사진이 주어졌을 때만 감식반을 현장에 투입하고, 없으면 추적 업무에만 집중합니다.
        self.feature_matcher: Optional[FeatureMatcher] = None
        if config.target_image:
            target_path = Path(config.target_image)
            if target_path.exists():
                self.feature_matcher = FeatureMatcher(
                    target_image_path=target_path,
                    method=config.feature_method,
                    ratio_thresh=config.ratio_thresh,
                    min_match_count=config.min_match_count,
                )

    # [1차]: 문자열 '0' 같은 숫자는 웹캠 장치 인덱스 int로 변환하고, 파일 경로는 문자열 그대로 유지합니다.
    # [2차]: "0"이라고 적으면 0번 웹캠을 켜고, "vtest.avi"라고 적으면 동영상 파일을 열도록 구분합니다.
    @staticmethod
    def _parse_source(source: str) -> int | str:
        """Convert numeric strings like '0' to webcam index int; keep file paths as str."""
        stripped = source.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped

    # [1차]: 단일 프레임에 대해 배경차분 → 감지 → 추적 → 특징점매칭 → 시각화 파이프라인을 순차 실행합니다.
    # [2차]: 사진 1장이 들어오면 5명의 팀원이 컨베이어 벨트처럼 차례대로 가공해 완성본을 내놓습니다.
    def process_frame(
        self, frame: np.ndarray, fps: float, paused: bool = False
    ) -> tuple[np.ndarray, Optional[MatchResult]]:
        """Run full CV pipeline on a single frame and return the annotated frame and match result."""
        fg_mask = self.bg_subtractor.apply(frame)
        detections = self.detector.detect(fg_mask)
        tracked = self.tracker.update(detections)

        match_result: Optional[MatchResult] = None
        if self.feature_matcher is not None:
            match_result = self.feature_matcher.match(frame)

        bg_info = (
            f"BG: {self.config.bg_method.upper()} (lr={self.config.learning_rate}) | "
            f"VelPred: {'ON' if self.config.use_velocity_prediction else 'OFF'}"
        )
        rendered = self.visualizer.draw(
            frame=frame,
            tracked_objects=tracked,
            fps=fps,
            match_result=match_result,
            paused=paused,
            bg_info=bg_info,
        )
        return rendered, match_result

    # [1차]: VideoCapture를 열어 프레임 루프를 돌며 FPS 계산, 키보드 제어(q/s/p), 리소스 해제를 수행합니다.
    # [2차]: 재생 버튼을 눌러 영상이 끝날 때까지 실시간 분석을 돌리고, 끝나면 장비를 깔끔하게 정리합니다.
    def run(self) -> RunSummary:
        """Open video source and execute frame-by-frame analysis loop."""
        # [1차]: 파싱된 소스로 OpenCV VideoCapture를 열고 실패 시 RuntimeError를 발생시킵니다.
        # [2차]: 웹캠이나 영상 파일을 열어보고, 안 열리면 즉시 오류를 알립니다.
        capture_source = self._parse_source(self.config.source)
        cap = cv2.VideoCapture(capture_source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.config.source}")

        # [1차]: save_video 옵션이 설정된 경우 출력 MP4 VideoWriter를 초기화합니다.
        # [2차]: 녹화 옵션이 켜져 있으면 결과 화면을 저장할 빈 비디오 테이프를 준비합니다.
        writer: Optional[cv2.VideoWriter] = None
        if self.config.save_video:
            out_path = Path(self.config.save_video)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (width, height))

        # [1차]: 통계 카운터, 일시정지 플래그, FPS 측정용 타임스탬프를 초기화합니다.
        # [2차]: 스톱워치를 0으로 맞추고 통계 기록장을 펼칩니다.
        frames_processed = 0
        target_detected_frames = 0
        snapshots_saved = 0
        paused = False
        last_rendered: Optional[np.ndarray] = None
        start_time = time.perf_counter()
        prev_time = start_time

        # [1차]: 예외나 중단 시에도 캡처/라이터/창 리소스가 반드시 해제되도록 try-finally로 감쌉니다.
        # [2차]: 도중에 에러가 나도 카메라와 창을 안전하게 닫고 나가도록 안전벨트를 맵니다.
        try:
            while True:
                # [1차]: 일시정지 상태가 아닐 때만 다음 프레임을 읽고 파이프라인을 실행합니다.
                # [2차]: 재생 중이면 새 장면을 읽어 분석하고, 일시정지 중이면 화면을 멈춰둡니다.
                if not paused:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break

                    # [1차]: 직전 프레임과의 경과 시간(0 나눗셈 방지용 1e-6 하한)으로 순간 FPS를 계산합니다.
                    # [2차]: 방금 1장 처리하는 데 걸린 시간으로 초당 몇 장 속도인지 계산합니다.
                    now = time.perf_counter()
                    dt = max(now - prev_time, 1e-6)
                    instant_fps = 1.0 / dt
                    prev_time = now

                    last_rendered, match_result = self.process_frame(
                        frame, fps=instant_fps, paused=False
                    )
                    frames_processed += 1
                    if match_result is not None and match_result.detected:
                        target_detected_frames += 1

                    if writer is not None:
                        writer.write(last_rendered)

                    # [1차]: max_frames 제한에 도달하면 루프를 정상 종료합니다.
                    # [2차]: 미리 정해둔 프레임 수만큼 다 봤으면 재생을 멈춥니다.
                    if (
                        self.config.max_frames is not None
                        and frames_processed >= self.config.max_frames
                    ):
                        break

                # [1차]: GUI 모드일 때 화면을 표시하고 키보드 입력(q/ESC 종료, p 일시정지, s 스냅샷)을 처리합니다.
                # [2차]: 모니터에 결과 화면을 띄우고 사용자의 리모컨 입력(q, p, s)을 실시간으로 받습니다.
                if not self.config.headless and last_rendered is not None:
                    display_frame = (
                        self.visualizer.draw(
                            last_rendered, {}, 0.0, paused=True
                        )
                        if paused
                        else last_rendered
                    )
                    cv2.imshow(self.WINDOW_NAME, display_frame)
                    key = cv2.waitKey(1 if not paused else 30) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("p"):
                        paused = not paused
                    if key == ord("s"):
                        self.visualizer.save_snapshot(
                            last_rendered, self.config.snapshot_dir, frames_processed
                        )
                        snapshots_saved += 1
        finally:
            # [1차]: VideoCapture, VideoWriter, OpenCV GUI 창 리소스를 안전하게 해제합니다.
            # [2차]: 상영이 끝나면 카메라 전원을 끄고 극장 불을 끕니다.
            cap.release()
            if writer is not None:
                writer.release()
            if not self.config.headless:
                cv2.destroyAllWindows()

        # [1차]: 전체 경과 시간 대비 평균 FPS를 계산해 RunSummary로 반환합니다.
        # [2차]: 최종 성적표 영수증을 출력해 호출자에게 돌려줍니다.
        total_elapsed = max(time.perf_counter() - start_time, 1e-6)
        avg_fps = round(frames_processed / total_elapsed, 2)
        return RunSummary(
            frames_processed=frames_processed,
            average_fps=avg_fps,
            total_unique_ids=self.tracker.total_registered,
            target_detected_frames=target_detected_frames,
            snapshots_saved=snapshots_saved,
        )
