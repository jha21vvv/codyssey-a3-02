"""Feature-based target object re-identification using ORB or SIFT."""

# [1차]: 타입 힌트 지연 평가를 활성화하여 신형 유니온 타입(str | Path)을 안전하게 지원합니다.
# [2차]: 타입 이름표를 미리 평가하지 않고 유연하게 읽도록 설정합니다.
from __future__ import annotations

# [1차]: 특징점 매칭 결과를 불변 객체로 반환하기 위해 dataclass를 임포트합니다.
# [2차]: 감식반이 작성해 돌려줄 "수배 물건 판독 결과지" 양식을 가져옵니다.
from dataclasses import dataclass

# [1차]: OS 독립적인 파일 경로 처리를 위해 pathlib.Path를 임포트합니다.
# [2차]: 윈도우·맥 어디서나 안전하게 파일 주소를 다루는 내비게이션 도구입니다.
from pathlib import Path

# [1차]: Optional 타입 힌트를 임포트합니다.
# [2차]: 파일 로드 실패 시 None이 반환될 수 있음을 명시합니다.
from typing import Optional

# [1차]: ORB/SIFT 특징점 추출기, BFMatcher, 이미지 디코딩·회전 변환을 위해 OpenCV를 임포트합니다.
# [2차]: 지문 채취와 대조를 담당하는 과학수사 공구함(OpenCV)을 엽니다.
import cv2

# [1차]: 한글 경로 안전 로딩(np.fromfile)과 배열 처리를 위해 NumPy를 임포트합니다.
# [2차]: 파일의 바이트 조각을 숫자 배열로 읽어오는 도우미입니다.
import numpy as np


# [1차]: 단일 프레임에 대한 특징점 매칭 정량 결과를 담는 불변 데이터클래스입니다.
# [2차]: "찾았는지 여부, 기준 지문 수, 화면 지문 수, 일치 개수, 일치율(%)"이 적힌 감식 보고서입니다.
@dataclass(frozen=True)
class MatchResult:
    """Quantitative result of feature matching on a single frame."""

    # [1차]: 유효 매칭 수가 min_match_count 이상인지 여부(TARGET DETECTED 판정)입니다.
    # [2차]: 화면에 수배 물건이 등장했는지(True/False) 도장을 찍는 칸입니다.
    detected: bool
    # [1차]: 등록된 기준 타겟 이미지에서 추출된 특징점 총 개수입니다.
    # [2차]: 수배 전단지 사진에서 뽑아낸 지문 총 개수입니다.
    target_keypoints: int
    # [1차]: 현재 영상 프레임에서 추출된 특징점 총 개수입니다.
    # [2차]: 지금 CCTV 화면 전체에서 뽑아낸 지문 총 개수입니다.
    frame_keypoints: int
    # [1차]: Lowe's Ratio Test를 통과한 신뢰성 있는 매칭 쌍 개수입니다.
    # [2차]: 수배 사진과 CCTV 화면에서 진짜로 똑같다고 판명된 지문 개수입니다.
    good_matches: int
    # [1차]: 기준 특징점 대비 유효 매칭 비율(백분율, 소수 둘째 자리)입니다.
    # [2차]: 수배 지문 중 몇 퍼센트가 화면에서 발견되었는지 보여주는 일치율입니다.
    match_rate: float
    # [1차]: 사용된 특징점 알고리즘 이름('orb' 또는 'sift')입니다.
    # [2차]: 어떤 감식 기법을 썼는지 적어두는 칸입니다.
    method: str


# [1차]: Windows 환경에서 한글·공백 포함 경로의 cv2.imread 실패 버그를 우회하는 안전한 이미지 로더입니다.
# [2차]: OpenCV가 한글 폴더 이름을 못 읽는 약점을 피해 파이썬으로 먼저 바이트를 읽은 뒤 그림으로 복원합니다.
def _imread_unicode(path: Path, flags: int = cv2.IMREAD_GRAYSCALE) -> Optional[np.ndarray]:
    """Read an image safely even when the Windows file path contains Korean or spaces."""
    # [1차]: 파일이 존재하지 않거나 디렉터리라면 None을 반환합니다.
    # [2차]: 사진 파일이 실제로 그 자리에 있는지 먼저 확인합니다.
    if not path.exists() or not path.is_file():
        return None
    # [1차]: 파일의 원시 바이트를 uint8 1차원 NumPy 배열로 읽습니다.
    # [2차]: 파일 이름을 신경 쓰지 않고 사진 파일의 알맹이 숫자 덩어리를 그대로 가져옵니다.
    raw = np.fromfile(str(path), dtype=np.uint8)
    # [1차]: 빈 파일(0바이트)인 경우 None을 반환합니다.
    # [2차]: 속이 텅 빈 파일이면 깨지기 전에 돌려보냅니다.
    if raw.size == 0:
        return None
    # [1차]: 메모리 버퍼의 바이트 배열을 OpenCV 이미지 행렬로 디코딩합니다.
    # [2차]: 숫자 덩어리를 흑백(또는 컬러) 사진으로 펼쳐줍니다.
    return cv2.imdecode(raw, flags)


# [1차]: ORB 또는 SIFT 특징점 매칭으로 특정 물체를 매 프레임 재인식하는 FeatureMatcher 클래스입니다.
# [2차]: 미리 등록해 둔 물건의 지문을 외워두고 CCTV 프레임마다 똑같은 물건이 나타났는지 감시하는 감식반입니다.
class FeatureMatcher:
    """Re-identifies a pre-registered target object image in video frames using ORB or SIFT."""

    # [1차]: 타겟 이미지를 로드하고 선택한 알고리즘(ORB/SIFT)으로 기준 특징점과 디스크립터를 사전 계산합니다.
    # [2차]: 출근하자마자 수배 물건 사진을 받아 지문을 미리 따서 머릿속에 저장해 둡니다.
    def __init__(
        self,
        target_image_path: str | Path,
        method: str = "orb",
        ratio_thresh: float = 0.75,
        min_match_count: int = 15,
        nfeatures: int = 1000,
    ) -> None:
        # [1차]: 알고리즘 이름, 비율 임계값, 최소 매칭 수의 유효성을 검증합니다.
        # [2차]: 감식 방식과 합격 기준이 올바른지 입구에서 확인합니다.
        if method not in {"orb", "sift"}:
            raise ValueError(f"Unsupported feature method: {method}. Use 'orb' or 'sift'.")
        if not (0.0 < ratio_thresh < 1.0):
            raise ValueError("ratio_thresh must be between 0.0 and 1.0.")
        if min_match_count <= 0:
            raise ValueError("min_match_count must be positive.")

        self.target_path = Path(target_image_path)
        self.method = method
        self.ratio_thresh = ratio_thresh
        self.min_match_count = min_match_count

        # [1차]: 유니코드 안전 로더로 기준 이미지를 흑백(그레이스케일)으로 읽어옵니다.
        # [2차]: 수배 물건 사진을 흑백 지문 분석용으로 꺼내옵니다.
        target_gray = _imread_unicode(self.target_path, cv2.IMREAD_GRAYSCALE)
        # [1차]: 이미지 로드 실패 시 FileNotFoundError를 발생시킵니다.
        # [2차]: 수배 사진이 없으면 감식을 시작할 수 없으므로 오류를 알립니다.
        if target_gray is None:
            raise FileNotFoundError(f"Failed to load target image: {self.target_path}")

        # [1차]: ORB는 이진 디스크립터이므로 해밍 거리(NORM_HAMMING), SIFT는 실수 벡터이므로 L2 거리를 사용합니다.
        # [2차]: ORB 지문은 0/1 비트 차이로, SIFT 지문은 직선 거리로 닮은 정도를 재는 전용 자를 준비합니다.
        if method == "orb":
            self._extractor = cv2.ORB_create(nfeatures=nfeatures)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError("cv2.SIFT_create is not available in this OpenCV build.")
            self._extractor = cv2.SIFT_create(nfeatures=nfeatures)
            self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # [1차]: 기준 이미지에서 특징점(Keypoints)과 기술자(Descriptors)를 미리 추출해 캐싱합니다.
        # [2차]: 매번 수배 사진을 다시 분석하면 느리므로 처음 1번만 지문을 따서 보관해 둡니다.
        self._target_kp, self._target_des = self._extractor.detectAndCompute(
            target_gray, None
        )
        # [1차]: 민무늬 이미지처럼 특징점이 전혀 추출되지 않으면 ValueError를 발생시킵니다.
        # [2차]: 하얀 종이처럼 지문이 하나도 없는 사진은 찾을 수 없으므로 거절합니다.
        if self._target_des is None or len(self._target_kp) == 0:
            raise ValueError(f"No features could be extracted from target image: {self.target_path}")

    # [1차]: 기준 타겟 이미지에서 추출된 특징점 개수를 읽기 전용 속성으로 제공합니다.
    # [2차]: 보관 중인 수배 지문이 총 몇 개인지 알려주는 안내창입니다.
    @property
    def target_keypoint_count(self) -> int:
        return len(self._target_kp)

    # [1차]: 현재 프레임에서 특징점을 추출하고 knnMatch(k=2) + Lowe's Ratio Test로 재인식 결과를 반환합니다.
    # [2차]: CCTV 화면에서 지문을 뽑아 수배 지문과 비교한 뒤 "TARGET DETECTED" 여부를 판정합니다.
    def match(self, frame: np.ndarray) -> MatchResult:
        """Match the registered target image against a BGR or grayscale video frame."""
        # [1차]: 빈 프레임 입력 시 ValueError로 즉시 방어합니다.
        # [2차]: 빈 화면이 들어오면 오류를 알립니다.
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame must be a non-empty numpy array.")

        # [1차]: 3채널 컬러 영상이면 흑백으로 변환하고, 이미 2D면 그대로 사용합니다.
        # [2차]: 지문 비교를 위해 컬러 화면을 흑백으로 바꿉니다.
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.ndim == 3
            else frame
        )
        # [1차]: 현재 프레임의 특징점과 디스크립터를 추출합니다.
        # [2차]: 지금 CCTV 화면 곳곳에서 지문을 채취합니다.
        frame_kp, frame_des = self._extractor.detectAndCompute(gray, None)

        target_count = len(self._target_kp)
        frame_count = 0 if frame_kp is None else len(frame_kp)

        # [1차]: knnMatch(k=2)는 최소 2개의 비교 대상이 필요하므로 2개 미만이면 안전하게 미검출을 반환합니다.
        # [2차]: 어두운 화면이라 지문이 2개도 안 나오면 억지로 비교하다 뻗지 않고 "못 찾음"으로 안전 보고합니다.
        if frame_des is None or frame_count < 2 or len(self._target_des) < 2:
            return MatchResult(
                detected=False,
                target_keypoints=target_count,
                frame_keypoints=frame_count,
                good_matches=0,
                match_rate=0.0,
                method=self.method,
            )

        # [1차]: 각 타겟 특징점에 대해 가장 가까운 1등(m)과 2등(n) 매칭 후보를 찾습니다.
        # [2차]: 수배 지문마다 CCTV에서 가장 닮은 후보 1등과 2등을 뽑아옵니다.
        knn_matches = self._matcher.knnMatch(self._target_des, frame_des, k=2)
        # [1차]: Lowe's Ratio Test(m.distance < ratio_thresh * n.distance)를 통과한 확실한 매칭만 카운트합니다.
        # [2차]: 1등이 2등과 비슷하게 애매하면 탈락시키고, 1등이 압도적으로 닮았을 때만 진짜 일치로 인정합니다.
        good_count = 0
        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_count += 1

        # [1차]: 타겟 특징점 대비 유효 매칭 비율(%)과 임계값 통과 여부를 계산해 반환합니다.
        # [2차]: 일치율(%)을 계산하고 기준 개수(15개) 이상이면 "찾았다(True)" 도장을 찍어 보고합니다.
        match_rate = round((good_count / target_count) * 100.0, 2) if target_count > 0 else 0.0
        detected = good_count >= self.min_match_count

        return MatchResult(
            detected=detected,
            target_keypoints=target_count,
            frame_keypoints=frame_count,
            good_matches=good_count,
            match_rate=match_rate,
            method=self.method,
        )

    # [1차]: 회전(warpAffine) 및 좌측 가림(픽셀 마스킹) 변형 이미지를 생성하는 정량 벤치마크 유틸리티입니다.
    # [2차]: 수배 사진을 30도/60도 돌리거나 손으로 30%/50% 가려보는 모의고사 시험지 제작기입니다.
    @staticmethod
    def transform_image(
        image: np.ndarray,
        angle_deg: float = 0.0,
        occlusion_ratio: float = 0.0,
    ) -> np.ndarray:
        """Create controlled rotation/occlusion variants of an image for quantitative benchmarking."""
        # [1차]: 빈 이미지 입력을 방어하고 원본을 훼손하지 않기 위해 복사본을 만듭니다.
        # [2차]: 원본 사진은 소중하니 복사본을 떠서 실험합니다.
        if image is None or image.size == 0:
            raise ValueError("image must be non-empty.")
        result = image.copy()
        h, w = result.shape[:2]

        # [1차]: 회전 각도가 0이 아니면 이미지 중심을 기준으로 아핀 회전 변환을 적용합니다.
        # [2차]: 사진 한가운데에 압정을 꽂고 지정한 각도만큼 빙그르르 돌려놓습니다.
        if angle_deg != 0.0:
            center = (w / 2.0, h / 2.0)
            rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            result = cv2.warpAffine(
                result,
                rot_mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        # [1차]: 가림 비율(0.0~1.0)만큼 왼쪽 영역의 픽셀을 0(검정)으로 덮어 가림(occlusion)을 재현합니다.
        # [2차]: 사진 왼쪽부터 지정한 비율만큼 검은 종이로 가려 물체가 숨은 상황을 만듭니다.
        if occlusion_ratio > 0.0:
            clamped = min(max(occlusion_ratio, 0.0), 1.0)
            occ_w = int(round(w * clamped))
            result[:, :occ_w] = 0

        return result
