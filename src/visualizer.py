"""Visualization overlay renderer and Unicode-safe snapshot saver."""

# [1차]: 타입 힌트 지연 평가를 활성화합니다.
# [2차]: 최신 타입 표기를 안전하게 사용하도록 준비합니다.
from __future__ import annotations

# [1차]: 스냅샷 파일명에 현재 시각 타임스탬프를 붙이기 위해 datetime을 임포트합니다.
# [2차]: 찍은 사진이 서로 덮어쓰이지 않게 촬영 날짜·시간 도장을 찍는 시계입니다.
from datetime import datetime

# [1차]: 디렉터리 생성과 파일 저장을 위해 Path를 임포트합니다.
# [2차]: 앨범 폴더를 만들고 사진을 저장하는 파일 관리자입니다.
from pathlib import Path

# [1차]: 딕셔너리·선택적 타입·튜플 힌트를 임포트합니다.
# [2차]: 함수 입력과 출력 모양을 미리 알려주는 안내 라벨입니다.
from typing import Dict, Optional, Tuple

# [1차]: 사각형·원·글자·폴리라인 그리기와 PNG 인코딩을 위해 OpenCV를 임포트합니다.
# [2차]: 화면 위에 그림과 글씨를 그리는 색연필 세트입니다.
import cv2

# [1차]: 궤적 좌표 배열 생성(np.array)을 위해 NumPy를 임포트합니다.
# [2차]: 발자국 좌표들을 한 줄의 점 목록으로 묶어주는 도구입니다.
import numpy as np

# [1차]: 특징점 매칭 결과와 추적 객체 타입을 임포트합니다.
# [2차]: 화가가 화면에 그려야 할 정보 카드(감식 결과, 손님 번호표)를 가져옵니다.
from src.feature_matcher import MatchResult
from src.tracker import TrackedObject


# [1차]: 추적 결과, FPS HUD, 타겟 감지 배너를 프레임에 오버레이하고 스냅샷을 저장하는 클래스입니다.
# [2차]: 분석 결과를 사람이 한눈에 볼 수 있게 CCTV 화면 위에 깔끔하게 그려주는 전담 화가입니다.
class FrameVisualizer:
    """Draws bounding boxes, unique IDs, trajectories, FPS HUD, and target detection banners."""

    # [1차]: 객체 ID별로 구분하기 쉬운 8가지 고정 BGR 색상 팔레트를 정의합니다.
    # [2차]: 손님마다 서로 다른 형광펜 색깔을 배정하기 위한 8색 크레파스 통입니다.
    _PALETTE = (
        (0, 220, 0),
        (255, 165, 0),
        (0, 191, 255),
        (255, 105, 180),
        (147, 112, 219),
        (64, 224, 208),
        (255, 215, 0),
        (255, 99, 71),
    )

    # [1차]: object_id에 나머지 연산(%)을 적용해 항상 동일한 BGR 색상을 결정론적으로 반환합니다.
    # [2차]: 1번 손님은 늘 같은 색깔, 2번 손님은 늘 다른 색깔로 일관되게 칠해줍니다.
    @classmethod
    def color_for_id(cls, object_id: int) -> Tuple[int, int, int]:
        """Return a deterministic BGR color for each object ID."""
        return cls._PALETTE[object_id % len(cls._PALETTE)]

    # [1차]: 원본 프레임 복사본 위에 바운딩 박스, ID, 궤적, FPS HUD, 타겟 배너, 일시정지 표시를 렌더링합니다.
    # [2차]: 원본 사진 위에 투명 필름을 얹어 네모 상자·번호·발자국·속도계·경고등을 모두 그려줍니다.
    def draw(
        self,
        frame: np.ndarray,
        tracked_objects: Dict[int, TrackedObject],
        fps: float,
        match_result: Optional[MatchResult] = None,
        paused: bool = False,
        bg_info: str = "",
    ) -> np.ndarray:
        """Return a copy of frame with all tracking overlays and HUD elements drawn."""
        # [1차]: 원본 프레임을 변형하지 않도록 복사본(canvas)을 생성합니다.
        # [2차]: 원본 CCTV 화면을 망치지 않게 새 도화지에 복사해서 그립니다.
        canvas = frame.copy()

        # [1차]: 모든 활성 추적 객체를 순회하며 박스, 중심점, ID 텍스트, 이동 궤적을 그립니다.
        # [2차]: 출석부에 있는 손님 한 명씩 차례로 테두리·번호표·걸어온 길 선을 그려줍니다.
        for obj_id, obj in tracked_objects.items():
            color = self.color_for_id(obj_id)
            x, y, w, h = obj.bbox

            # [1차]: 바운딩 박스 직사각형과 중심점 채워진 원(반지름 4px)을 그립니다.
            # [2차]: 손님 둘레에 네모 박스를 치고 배꼽 위치에 점을 찍습니다.
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
            cv2.circle(canvas, obj.centroid, 4, color, -1)

            # [1차]: 바운딩 박스 위쪽(화면 밖으로 잘리지 않도록 최소 y=16 보장)에 ID 라벨을 출력합니다.
            # [2차]: 네모 상자 바로 위에 "ID 1" 이름표를 달아줍니다.
            label = f"ID {obj_id}"
            text_y = max(y - 8, 16)
            cv2.putText(
                canvas,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            # [1차]: 궤적 좌표가 2개 이상이면 (N, 1, 2) int32 배열로 변환해 열린 다각형 선(polylines)을 그립니다.
            # [2차]: 발자국이 2개 이상 모이면 점들을 실로 꿰어 지나온 이동 경로를 선으로 보여줍니다.
            if len(obj.trajectory) >= 2:
                pts = np.array(list(obj.trajectory), dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)

        # [1차]: 좌상단 HUD 영역에 반투명 검은 배경 박스(addWeighted alpha=0.65)를 합성합니다.
        # [2차]: 글씨가 잘 보이도록 화면 왼쪽 위에 반투명 검은 유리판을 덧댑니다.
        hud_w = 340
        hud_h = 88 if bg_info else 68
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (8 + hud_w, 8 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        # [1차]: 현재 초당 프레임(FPS)과 추적 중인 객체 수를 HUD 첫 줄에 표시합니다.
        # [2차]: 자동차 계기판처럼 현재 속도와 감지된 사람 수를 띄워줍니다.
        cv2.putText(
            canvas,
            f"FPS: {fps:.1f} | Objects: {len(tracked_objects)}",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 180),
            2,
            cv2.LINE_AA,
        )
        # [1차]: 키보드 단축키 안내(q 종료, s 스냅샷, p 일시정지)를 둘째 줄에 표시합니다.
        # [2차]: 사용자가 까먹지 않도록 리모컨 버튼 설명을 작게 적어둡니다.
        cv2.putText(
            canvas,
            "Keys: [q] Quit  [s] Snapshot  [p] Pause",
            (18, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        # [1차]: 배경 차분 방식 및 학습률 정보가 있으면 셋째 줄에 표시합니다.
        # [2차]: 현재 어떤 배경 지우개와 학습 속도로 돌아가는지 상태 줄에 적어줍니다.
        if bg_info:
            cv2.putText(
                canvas,
                bg_info,
                (18, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 220, 255),
                1,
                cv2.LINE_AA,
            )

        # [1차]: 특징점 매칭 성공 시 하단에 초록색 배너와 TARGET DETECTED (Target Found) 메시지를 그립니다.
        # [2차]: 수배 물건이 화면에 나타나면 하단에 커다란 초록색 알림 띠를 띄워 알려줍니다.
        if match_result is not None and match_result.detected:
            h_img, w_img = canvas.shape[:2]
            banner_text = (
                f"TARGET DETECTED (Target Found) | "
                f"{match_result.method.upper()} {match_result.good_matches} ({match_result.match_rate:.1f}%)"
            )
            cv2.rectangle(canvas, (8, h_img - 48), (w_img - 8, h_img - 10), (0, 140, 0), -1)
            cv2.putText(
                canvas,
                banner_text,
                (18, h_img - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # [1차]: 일시정지 상태라면 우상단에 주황색 PAUSED 배지를 표시합니다.
        # [2차]: 리모컨 일시정지 버튼을 누르면 화면 구석에 "일시정지 중" 표시등을 켭니다.
        if paused:
            cv2.putText(
                canvas,
                "PAUSED",
                (canvas.shape[1] - 130, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        return canvas

    # [1차]: cv2.imencode + Path.write_bytes 조합으로 한글 경로에서도 깨지지 않게 PNG 스냅샷을 저장합니다.
    # [2차]: OpenCV가 한글 폴더에 저장을 못 하는 문제를 피해 메모리에서 PNG로 압축한 뒤 파이썬이 직접 파일로 써줍니다.
    @staticmethod
    def save_snapshot(frame: np.ndarray, snapshot_dir: str | Path, frame_idx: int) -> Path:
        """Save current rendered frame as PNG safely even with Unicode paths."""
        # [1차]: 저장 폴더가 없으면 부모 경로까지 포함해 자동 생성합니다.
        # [2차]: 사진 앨범 폴더가 없으면 새로 만들어 둡니다.
        out_dir = Path(snapshot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # [1차]: 현재 시각과 프레임 번호를 조합해 중복되지 않는 파일명을 만듭니다.
        # [2차]: "20261125_190500_f00042.png"처럼 고유한 사진 이름을 붙입니다.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"snapshot_{stamp}_f{frame_idx:05d}.png"
        # [1차]: 메모리 버퍼에서 PNG 포맷으로 인코딩하고 실패 시 RuntimeError를 발생시킵니다.
        # [2차]: 픽셀 그림을 PNG 파일 데이터로 압축합니다.
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("Failed to encode snapshot frame as PNG.")
        # [1차]: 인코딩된 바이트를 파일로 기록하고 저장 경로를 반환합니다.
        # [2차]: 압축된 사진을 앨범 폴더에 안전하게 저장하고 파일 주소를 돌려줍니다.
        out_path.write_bytes(encoded.tobytes())
        return out_path
