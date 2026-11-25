"""Contour-based motion object detection with area filtering."""

# [1차]: 타입 힌트 지연 평가를 활성화합니다.
# [2차]: 타입 표기를 깔끔하고 안전하게 처리합니다.
from __future__ import annotations

# [1차]: 감지 결과를 담을 불변 구조체 생성을 위해 dataclass를 임포트합니다.
# [2차]: 발견한 물체의 좌표와 넓이를 적어둘 결과 카드 양식을 가져옵니다.
from dataclasses import dataclass

# [1차]: 리스트와 튜플 타입 힌트를 임포트합니다.
# [2차]: "목록(List)과 묶음(Tuple) 형태로 돌려준다"는 약속을 적기 위한 도구입니다.
from typing import List, Tuple

# [1차]: 윤곽선 추출(findContours), 면적 계산(contourArea), 바운딩 박스(boundingRect)를 위해 OpenCV를 임포트합니다.
# [2차]: 흰 덩어리의 테두리를 따라 그리고 크기를 재는 도구를 꺼냅니다.
import cv2

# [1차]: 입력 이진 마스크 타입 검증을 위해 NumPy를 임포트합니다.
# [2차]: 마스크가 올바른 픽셀 배열인지 확인하는 검사기입니다.
import numpy as np


# [1차]: 감지된 단일 움직임 영역의 바운딩 박스, 중심점, 면적을 저장하는 불변 데이터클래스입니다.
# [2차]: 찾은 물체 1개마다 "사각형 위치, 한가운데 점, 덩치 크기"를 적어두는 신분증 카드입니다.
@dataclass(frozen=True)
class Detection:
    """Single detected foreground region."""

    # [1차]: 객체의 좌상단 좌표와 너비·높이 (x, y, w, h) 튜플입니다.
    # [2차]: 물체를 감싸는 네모 상자의 위치와 크기입니다.
    bbox: Tuple[int, int, int, int]
    # [1차]: 바운딩 박스의 기하학적 중심점 (cx, cy) 정수 좌표입니다.
    # [2차]: 추적기가 거리를 잴 때 사용할 물체의 배꼽(중심점) 좌표입니다.
    centroid: Tuple[int, int]
    # [1차]: 윤곽선 내부의 실제 픽셀 면적(float)입니다.
    # [2차]: 이 물체가 얼마나 큰 덩어리인지 나타내는 넓이 숫자입니다.
    area: float


# [1차]: 이진 전경 마스크에서 윤곽선을 추출하고 면적 필터링을 수행하는 MotionDetector 클래스입니다.
# [2차]: 흑백 실루엣 그림에서 진짜 움직이는 물체만 골라 네모 박스를 쳐주는 관찰자입니다.
class MotionDetector:
    """Extracts moving object bounding boxes and centroids from a binary foreground mask."""

    # [1차]: 면적 필터링의 하한(min_area)과 상한(max_area)을 검증 후 저장합니다.
    # [2차]: 너무 작은 먼지와 너무 거대한 화면 전체 번짐을 거를 체의 크기를 설정합니다.
    def __init__(self, min_area: int = 500, max_area: int = 100000) -> None:
        # [1차]: 면적 범위가 논리적으로 유효하지 않으면 ValueError를 발생시킵니다.
        # [2차]: 최소 크기가 0 이하이거나 최대 크기보다 크면 잘못된 체이므로 거부합니다.
        if min_area <= 0 or max_area < min_area:
            raise ValueError("min_area must be positive and <= max_area.")
        self.min_area = min_area
        self.max_area = max_area

    # [1차]: 2D 이진 마스크에서 외곽선을 찾고 면적 필터링 후 결정론적으로 정렬된 Detection 목록을 반환합니다.
    # [2차]: 하얀 덩어리들의 테두리를 따서 크기가 적당한 것만 추린 뒤 왼쪽-위 순서로 줄 세워 보고합니다.
    def detect(self, fg_mask: np.ndarray) -> List[Detection]:
        """Find external contours, filter by area, and return deterministic sorted detections."""
        # [1차]: 입력 마스크가 비어 있거나 배열이 아니면 ValueError를 발생시킵니다.
        # [2차]: 빈 도화지가 들어오면 작업을 멈추고 알립니다.
        if fg_mask is None or not isinstance(fg_mask, np.ndarray) or fg_mask.size == 0:
            raise ValueError("fg_mask must be a non-empty 2D numpy array.")
        # [1차]: 3채널 컬러 영상이 잘못 들어오는 것을 막기 위해 2차원(단일 채널)인지 검사합니다.
        # [2차]: 흑백(2D) 마스크가 아니라 컬러(3D) 사진이 실수로 들어오면 거절합니다.
        if fg_mask.ndim != 2:
            raise ValueError("fg_mask must be a single-channel 2D binary image.")

        # [1차]: RETR_EXTERNAL로 최외곽 윤곽선만 추출하고 CHAIN_APPROX_SIMPLE로 꼭짓점을 압축 저장합니다.
        # [2차]: 덩어리 안쪽의 자잘한 구멍은 무시하고 바깥 테두리 선만 깔끔하게 따라 그립니다.
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # [1차]: 면적 조건을 통과한 감지 결과를 담을 리스트를 초기화합니다.
        # [2차]: 합격한 물체 신분증을 모아둘 빈 바구니를 준비합니다.
        detections: List[Detection] = []
        # [1차]: 추출된 모든 윤곽선을 순회하며 면적을 계산하고 필터링합니다.
        # [2차]: 찾은 흰 덩어리를 하나씩 저울에 올려 적당한 크기인지 확인합니다.
        for contour in contours:
            # [1차]: 윤곽선 내부 픽셀 면적을 부동소수점으로 계산합니다.
            # [2차]: 이 덩어리가 몇 픽셀짜리인지 넓이를 잽니다.
            area = float(cv2.contourArea(contour))
            # [1차]: 최소 면적 미만(노이즈) 또는 최대 면적 초과(조명 붕괴 등)는 건너뜁니다.
            # [2차]: 너무 작으면 먼지, 너무 크면 조명 번짐이므로 과감히 탈락시킵니다.
            if area < self.min_area or area > self.max_area:
                continue

            # [1차]: 윤곽선을 감싸는 최소 직교 사각형(x, y, w, h)을 계산합니다.
            # [2차]: 물체에 딱 맞는 직사각형 택배 상자 좌표를 구합니다.
            x, y, w, h = cv2.boundingRect(contour)
            # [1차]: 바운딩 박스의 가로·세로 중앙 좌표(cx, cy)를 정수로 산출합니다.
            # [2차]: 택배 상자의 정중앙에 빨간 핀을 꽂아 중심점을 구합니다.
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            # [1차]: Detection 인스턴스를 생성해 결과 리스트에 추가합니다.
            # [2차]: 합격한 물체의 신분증을 작성해 바구니에 담습니다.
            detections.append(
                Detection(
                    bbox=(int(x), int(y), int(w), int(h)),
                    centroid=(cx, cy),
                    area=area,
                )
            )

        # [1차]: (중심 X, 중심 Y, -면적) 기준으로 정렬하여 매 실행마다 동일한 순서를 보장합니다.
        # [2차]: 매번 실행할 때마다 순서가 뒤죽박죽되지 않도록 왼쪽→오른쪽, 큰 덩어리 순으로 줄을 세웁니다.
        detections.sort(key=lambda d: (d.centroid[0], d.centroid[1], -d.area))
        # [1차]: 정렬된 감지 객체 리스트를 반환합니다.
        # [2차]: 정리된 물체 신분증 바구니를 추적기(Tracker)에게 전달합니다.
        return detections
