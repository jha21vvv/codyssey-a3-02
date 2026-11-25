"""Background subtraction and morphological noise removal."""

# [1차]: 파이썬 타입 힌트의 지연 평가를 활성화합니다.
# [2차]: 타입 이름표를 미리 평가하지 않고 실행 시 유연하게 처리합니다.
from __future__ import annotations

# [1차]: None 허용 타입 힌트 Optional을 가져옵니다.
# [2차]: 내부 배경 모델이 초기에는 비어 있을 수 있음을 표시합니다.
from typing import Optional

# [1차]: 고전적 영상 처리(MOG2, absdiff, 모폴로지 연산)를 위해 OpenCV를 임포트합니다.
# [2차]: 사진·영상 편집 공구함(OpenCV)을 테이블 위에 펼쳐놓습니다.
import cv2

# [1차]: 영상 프레임(다차원 배열) 연산을 위해 NumPy를 임포트합니다.
# [2차]: 픽셀 숫자 행렬을 초고속으로 계산하는 전자계산기(NumPy)를 꺼냅니다.
import numpy as np


# [1차]: 정지 배경과 움직이는 전경을 분리하는 BackgroundSubtractor 클래스를 정의합니다.
# [2차]: CCTV 화면에서 가만히 있는 배경은 까맣게 지우고 움직이는 사람만 하얗게 남기는 필터 기계입니다.
class BackgroundSubtractor:
    """Separates moving foreground objects from a static or slowly changing background."""

    # [1차]: 배경 차분 방식, 학습률, MOG2 파라미터, 모폴로지 커널 크기를 초기화합니다.
    # [2차]: 필터 기계를 조립하며 지우개 종류, 배경 적응 속도, 빗자루 크기를 세팅합니다.
    def __init__(
        self,
        method: str = "mog2",
        learning_rate: float = 0.005,
        history: int = 500,
        var_threshold: float = 16.0,
        morph_kernel_size: int = 5,
        diff_threshold: int = 25,
    ) -> None:
        # [1차]: 지원하지 않는 배경 차분 방식이 들어오면 ValueError를 발생시킵니다.
        # [2차]: MOG2나 absdiff가 아닌 엉뚱한 버튼을 누르면 즉시 작동을 멈추고 알립니다.
        if method not in {"mog2", "absdiff"}:
            raise ValueError(f"Unsupported background subtraction method: {method}")
        # [1차]: learning_rate가 -1.0 또는 [0.0, 1.0] 범위인지 검증합니다.
        # [2차]: 배경 학습 속도가 허용 범위를 벗어나면 오류를 냅니다.
        if not (learning_rate == -1.0 or 0.0 <= learning_rate <= 1.0):
            raise ValueError("learning_rate must be -1.0 or within [0.0, 1.0].")
        # [1차]: 모폴로지 커널 크기가 양의 홀수인지 확인합니다.
        # [2차]: 대칭이 맞는 홀수 크기 빗자루만 받도록 검사합니다.
        if morph_kernel_size <= 0 or morph_kernel_size % 2 == 0:
            raise ValueError("morph_kernel_size must be a positive odd integer.")

        # [1차]: 선택된 알고리즘, 학습률, 차영상 임계값을 인스턴스 속성에 저장합니다.
        # [2차]: 기계 본체에 설정값을 메모해 둡니다.
        self.method = method
        self.learning_rate = learning_rate
        self.diff_threshold = diff_threshold
        # [1차]: 타원형(MORPH_ELLIPSE) 모폴로지 구조 요소 커널을 생성합니다.
        # [2차]: 둥글둥글한 사람 실루엣을 자연스럽게 다듬기 위해 둥근 모양 스탬프를 만듭니다.
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        # [1차]: method가 'mog2'일 때 OpenCV MOG2 가우시안 혼합 배경 모델 객체를 생성합니다.
        # [2차]: 각 픽셀의 평소 색깔 분포를 통계적으로 외우는 MOG2 두뇌를 장착합니다.
        self._mog2 = (
            cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=True,
            )
            if method == "mog2"
            else None
        )
        # [1차]: absdiff 모드에서 이동 평균 배경을 저장할 float32 배열을 None으로 초기화합니다.
        # [2차]: absdiff 방식이 첫 프레임을 기준 배경 사진으로 담아둘 빈 액자입니다.
        self._bg_model: Optional[np.ndarray] = None

    # [1차]: 입력 BGR/그레이스케일 프레임에서 노이즈가 제거된 이진 전경 마스크(0 또는 255)를 반환합니다.
    # [2차]: 새 CCTV 장면이 들어오면 움직이는 물체만 하얗게 칠한 흑백 도장을 찍어줍니다.
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Compute a clean binary foreground mask (0 or 255) for the given BGR/gray frame."""
        # [1차]: 입력 프레임이 None이거나 NumPy 배열이 아니거나 비어 있으면 ValueError를 발생시킵니다.
        # [2차]: 카메라에서 빈 종이가 들어오면 고장 나기 전에 입구에서 차단합니다.
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame must be a non-empty numpy array.")

        # [1차]: MOG2 모드일 때 learningRate를 전달해 전경 마스크를 계산합니다.
        # [2차]: MOG2 두뇌에게 새 화면을 보여주고 "평소 배경과 다른 곳만 표시해!"라고 시킵니다.
        if self.method == "mog2":
            raw_mask = self._mog2.apply(frame, learningRate=self.learning_rate)
            # [1차]: MOG2가 그림자를 회색(127)으로 표시하므로 200 기준 이진화로 그림자를 제거합니다.
            # [2차]: 사람 발밑의 흐릿한 그림자(127)는 지우고 진짜 몸통(255)만 남깁니다.
            _, binary_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)
        else:
            # [1차]: 3채널 컬러 영상이면 그레이스케일로 변환하고, 이미 2D면 복사합니다.
            # [2차]: 밝기 차이만 빠르게 비교하기 위해 컬러 사진을 흑백 사진으로 바꿉니다.
            gray = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3
                else frame.copy()
            )
            # [1차]: 5x5 가우시안 블러로 미세한 센서 노이즈를 부드럽게 완화합니다.
            # [2차]: 카메라의 자글자글한 모래알 잡음을 살짝 흐리게 문질러 없앱니다.
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # [1차]: 첫 프레임은 기준 배경 모델로 등록하고 빈 마스크를 반환합니다.
            # [2차]: 비교할 기준 사진이 아직 없으니 첫 장면을 기준 배경으로 액자에 끼워둡니다.
            if self._bg_model is None:
                self._bg_model = blurred.astype(np.float32)
                return np.zeros_like(blurred, dtype=np.uint8)

            # [1차]: 누적 배경 모델(float32)을 uint8로 변환해 현재 프레임과 절대 차이(absdiff)를 구합니다.
            # [2차]: 기준 배경 사진과 지금 사진을 겹쳐 놓고 달라진 픽셀만 뺄셈으로 찾아냅니다.
            bg_uint8 = cv2.convertScaleAbs(self._bg_model)
            diff = cv2.absdiff(bg_uint8, blurred)
            # [1차]: 차이값이 diff_threshold를 넘는 픽셀만 255(전경)로 이진화합니다.
            # [2차]: 미세한 차이는 무시하고 확실히 달라진 곳만 흰색 형광펜으로 칠합니다.
            _, binary_mask = cv2.threshold(
                diff, self.diff_threshold, 255, cv2.THRESH_BINARY
            )
            # [1차]: 학습률(alpha)에 따라 지수이동평균(accumulateWeighted)으로 배경 모델을 서서히 갱신합니다.
            # [2차]: 시간이 지날수록 현재 장면을 아주 조금씩 기준 배경에 섞어 최신화합니다.
            alpha = 0.01 if self.learning_rate < 0 else self.learning_rate
            if alpha > 0.0:
                cv2.accumulateWeighted(blurred, self._bg_model, alpha)

        # [1차]: 모폴로지 열림(MORPH_OPEN: 침식 후 팽창)으로 흩어진 점 노이즈를 제거합니다.
        # [2차]: 빗자루로 바닥에 떨어진 작은 흰 먼지들을 먼저 싹 쓸어냅니다.
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.kernel)
        # [1차]: 모폴로지 닫힘(MORPH_CLOSE: 팽창 후 침식)으로 객체 내부의 빈 구멍을 메웁니다.
        # [2차]: 사람 옷 색깔 때문에 몸통 안쪽에 뚫린 작은 구멍들을 퍼티로 매끈하게 채워 하나의 덩어리로 만듭니다.
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel)
        # [1차]: 정제된 최종 이진 마스크를 반환합니다.
        # [2차]: 깔끔하게 손질된 흑백 실루엣을 다음 단계(객체 감지기)로 넘겨줍니다.
        return cleaned

    # [1차]: 내부 배경 모델을 초기 상태로 리셋합니다.
    # [2차]: 지금까지 외운 배경 기억을 싹 지우고 새 출발합니다.
    def reset(self) -> None:
        """Reset internal background model state."""
        # [1차]: MOG2 모드라면 기존 하이퍼파라미터를 유지한 채 새 MOG2 인스턴스로 교체합니다.
        # [2차]: 설정은 그대로 두고 MOG2의 기억 메모리만 새것으로 갈아 끼웁니다.
        if self.method == "mog2" and self._mog2 is not None:
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history=self._mog2.getHistory(),
                varThreshold=self._mog2.getVarThreshold(),
                detectShadows=True,
            )
        # [1차]: absdiff용 배경 모델 참조를 None으로 비웁니다.
        # [2차]: absdiff 액자 속 기준 사진도 비워둡니다.
        self._bg_model = None
