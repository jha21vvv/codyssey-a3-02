"""Application configuration and strict validation."""

# [1차]: 파이썬 타입 힌트를 문자열처럼 지연 평가하여 순환 참조와 신형 타입 표기를 안전하게 지원합니다.
# [2차]: 이름표를 지금 당장 검사하지 않고 나중에 실제로 쓸 때 확인하도록 여유를 둡니다.
from __future__ import annotations

# [1차]: 보일러플레이트 없이 설정 클래스를 선언하기 위해 dataclass 데코레이터를 임포트합니다.
# [2차]: 매번 생성자(__init__)를 손으로 쓰지 않게 해주는 자동 양식 복사기를 꺼냅니다.
from dataclasses import dataclass

# [1차]: None을 허용하는 선택적 타입 힌트 Optional을 임포트합니다.
# [2차]: "값이 있을 수도 있고 비어 있을 수도 있음"을 표시하는 선택 항목 스티커입니다.
from typing import Optional


# [1차]: frozen=True로 불변(immutable) 설정 데이터클래스 AppConfig를 정의합니다.
# [2차]: 요리 시작 후 누군가 실수로 레시피 숫자를 고치지 못하도록 코팅해 둔 설정표입니다.
@dataclass(frozen=True)
class AppConfig:
    """Configuration container for the real-time video analysis application."""

    # [1차]: 입력 영상 소스(웹캠 인덱스 문자열 '0' 또는 비디오 파일 경로)의 기본값을 지정합니다.
    # [2차]: CCTV를 볼지 녹화 테이프를 볼지 적어두는 재생 소스 칸입니다.
    source: str = "data/videos/vtest.avi"
    # [1차]: 배경 차분 알고리즘 선택('mog2' 또는 'absdiff') 기본값입니다.
    # [2차]: 배경을 지우는 지우개 종류(똑똑한 MOG2 vs 단순 빼기 absdiff)를 고릅니다.
    bg_method: str = "mog2"
    # [1차]: 배경 모델이 새 프레임에 적응하는 학습률(0.0~1.0 또는 자동 -1.0)입니다.
    # [2차]: 방 안 풍경이 바뀌었을 때 얼마나 빨리 "이게 원래 배경이야"라고 외울지 정하는 속도입니다.
    learning_rate: float = 0.005
    # [1차]: MOG2가 배경 분포를 학습할 때 참고하는 과거 프레임 길이입니다.
    # [2차]: 최근 며칠(500프레임) 치 기억을 바탕으로 평균 배경을 계산할지 정합니다.
    history: int = 500
    # [1차]: MOG2에서 픽셀이 배경과 얼마나 달라야 전경(움직임)으로 볼지 정하는 마할라노비스 거리 임계값입니다.
    # [2차]: "이 정도 색 차이는 그냥 흔들림, 이 이상은 진짜 움직임!"을 가르는 민감도 눈금입니다.
    var_threshold: float = 16.0
    # [1차]: 모폴로지 열림/닫힘 연산에 사용할 구조 요소 커널의 크기(홀수)입니다.
    # [2차]: 노이즈 먼지를 털어내고 구멍을 메우는 빗자루의 굵기(5x5 픽셀)입니다.
    morph_kernel_size: int = 5
    # [1차]: 노이즈로 무시하지 않고 객체로 인정할 최소 윤곽선 면적(픽셀²)입니다.
    # [2차]: 먼지나 날파리처럼 너무 작은 점은 무시하는 체(거름망)의 최소 구멍 크기입니다.
    min_area: int = 500
    # [1차]: 조명 급변 등으로 화면 전체가 잡히는 오검출을 막기 위한 최대 윤곽선 면적입니다.
    # [2차]: 화면 절반을 덮는 거대한 빛 번짐은 사람으로 착각하지 않도록 거르는 상한선입니다.
    max_area: int = 100000
    # [1차]: 객체가 화면에서 사라진 뒤 ID를 즉시 지우지 않고 기다려줄 최대 프레임 수입니다.
    # [2차]: 손님이 기둥 뒤에 잠깐 숨었을 때 몇 초(15프레임)까지 번호표를 유지해줄지 정합니다.
    max_disappeared: int = 15
    # [1차]: 이전 프레임 객체와 현재 감지 객체를 같은 ID로 묶을 최대 유클리드 거리(픽셀)입니다.
    # [2차]: 한 걸음(1프레임) 사이에 사람이 순간이동할 수 있는 최대 보폭(80픽셀) 제한입니다.
    max_distance: float = 80.0
    # [1차]: 화면에 선으로 그릴 이동 궤적 좌표의 최대 저장 개수입니다.
    # [2차]: 발자국을 무한정 남기면 지저분하므로 최근 50걸음의 꼬리만 기억합니다.
    max_trajectory_len: int = 50
    # [1차]: 직전 속도 벡터로 다음 위치를 예측해 매칭할지 여부(규칙 기반 개선 비교용)입니다.
    # [2차]: "방금 오른쪽으로 뛰었으니 다음에도 오른쪽에 있겠지"라는 관성 예측 규칙 스위치입니다.
    use_velocity_prediction: bool = False
    # [1차]: 특징점 매칭으로 재인식할 기준 물체 이미지 경로(None이면 비활성화)입니다.
    # [2차]: 영상 속에서 찾아낼 수배 물건의 정면 사진 파일 경로입니다.
    target_image: Optional[str] = "data/targets/box.png"
    # [1차]: 특징점 추출 알고리즘 선택('orb' 또는 'sift')입니다.
    # [2차]: 물체의 지문을 따는 방식(빠른 ORB vs 정밀한 SIFT)을 선택합니다.
    feature_method: str = "orb"
    # [1차]: Lowe's Ratio Test에서 1등 매칭이 2등보다 얼마나 뚜렷해야 통과시킬지 정하는 비율입니다.
    # [2차]: 닮은꼴 후보 중 1등이 2등보다 확실히(75% 미만 거리로) 압도적일 때만 합격시킵니다.
    ratio_thresh: float = 0.75
    # [1차]: TARGET DETECTED 배너를 띄우기 위해 필요한 최소 유효 매칭 개수입니다.
    # [2차]: 지문이 최소 15군데 이상 일치해야 "찾았다!"라고 확신하는 기준선입니다.
    min_match_count: int = 15
    # [1차]: 키보드 's' 입력 시 스냅샷 PNG를 저장할 디렉터리 경로입니다.
    # [2차]: 순간 포착 사진을 모아둘 앨범 폴더 이름입니다.
    snapshot_dir: str = "output/snapshots"
    # [1차]: GUI 창(cv2.imshow) 없이 백그라운드/자동 검증 모드로 실행할지 여부입니다.
    # [2차]: 모니터 화면을 띄우지 않고 조용히 결과만 계산하는 무인 실행 스위치입니다.
    headless: bool = False
    # [1차]: 최대 처리 프레임 수 제한(None이면 영상 끝까지 실행)입니다.
    # [2차]: 긴 영화 전체 대신 앞부분 몇 장면만 빠르게 테스트할 때 쓰는 정지 타이머입니다.
    max_frames: Optional[int] = None
    # [1차]: 렌더링된 결과 영상을 MP4 파일로 저장할 경로(None이면 미저장)입니다.
    # [2차]: 분석 결과가 그려진 화면을 새 비디오 파일로 녹화할지 정하는 출력 경로입니다.
    save_video: Optional[str] = None

    # [1차]: dataclass 초기화 직후 자동 호출되어 모든 설정값의 도메인 유효성을 검증합니다.
    # [2차]: 주방에 들어가기 전에 재료가 상하지 않았는지 입구에서 검수하는 체크리스트입니다.
    def __post_init__(self) -> None:
        # [1차]: source가 빈 문자열이거나 공백만 있는지 검사합니다.
        # [2차]: 재생할 영상 주소가 비어 있으면 출발 전에 바로 경고를 울립니다.
        if not self.source or not str(self.source).strip():
            raise ValueError("source must be a non-empty webcam index or video path.")
        # [1차]: 지원하는 배경 차분 방식('mog2', 'absdiff')인지 확인합니다.
        # [2차]: 메뉴판에 없는 배경 지우개 이름을 적었는지 확인합니다.
        if self.bg_method not in {"mog2", "absdiff"}:
            raise ValueError(f"Unsupported bg_method: {self.bg_method}. Use 'mog2' or 'absdiff'.")
        # [1차]: 학습률이 -1.0(자동) 또는 0.0~1.0 범위 안에 있는지 검증합니다.
        # [2차]: 속도 다이얼이 0%~100% 밖의 엉뚱한 숫자로 돌아가 있지 않은지 체크합니다.
        if not (self.learning_rate == -1.0 or 0.0 <= self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be -1.0 (auto) or in [0.0, 1.0].")
        # [1차]: history 프레임 수가 양수인지 검사합니다.
        # [2차]: 기억 길이가 0 이하일 수 없으므로 양수인지 확인합니다.
        if self.history <= 0:
            raise ValueError("history must be positive.")
        # [1차]: var_threshold가 양수인지 검사합니다.
        # [2차]: 민감도 눈금이 0 이하로 내려가지 않았는지 확인합니다.
        if self.var_threshold <= 0:
            raise ValueError("var_threshold must be positive.")
        # [1차]: 모폴로지 커널 크기가 양의 홀수(3, 5, 7 등)인지 검증합니다.
        # [2차]: 중심 픽셀이 딱 가운데에 오려면 빗자루 크기가 반드시 홀수여야 합니다.
        if self.morph_kernel_size <= 0 or self.morph_kernel_size % 2 == 0:
            raise ValueError("morph_kernel_size must be a positive odd integer.")
        # [1차]: 최소 면적이 양수이며 최대 면적보다 작거나 같은지 검사합니다.
        # [2차]: 체의 최소 구멍이 최대 구멍보다 크면 아무것도 통과 못 하므로 순서를 확인합니다.
        if self.min_area <= 0 or self.max_area < self.min_area:
            raise ValueError("min_area must be positive and <= max_area.")
        # [1차]: max_disappeared가 음수가 아닌지 검증합니다.
        # [2차]: 기다려주는 프레임 수가 마이너스일 수는 없으므로 0 이상인지 확인합니다.
        if self.max_disappeared < 0:
            raise ValueError("max_disappeared must be non-negative.")
        # [1차]: max_distance가 양수인지 검증합니다.
        # [2차]: 최대 이동 보폭은 0보다 커야 매칭이 가능합니다.
        if self.max_distance <= 0:
            raise ValueError("max_distance must be positive.")
        # [1차]: max_trajectory_len이 양수인지 검증합니다.
        # [2차]: 궤적은 최소 1개 이상의 점을 담을 수 있어야 합니다.
        if self.max_trajectory_len <= 0:
            raise ValueError("max_trajectory_len must be positive.")
        # [1차]: feature_method가 'orb' 또는 'sift'인지 검증합니다.
        # [2차]: 지원하는 지문 감식 방식인지 확인합니다.
        if self.feature_method not in {"orb", "sift"}:
            raise ValueError(f"Unsupported feature_method: {self.feature_method}. Use 'orb' or 'sift'.")
        # [1차]: ratio_thresh가 0.0 초과 1.0 미만인지 검증합니다.
        # [2차]: 1등과 2등 비교 비율은 0과 1 사이의 소수여야 의미가 있습니다.
        if not (0.0 < self.ratio_thresh < 1.0):
            raise ValueError("ratio_thresh must be strictly between 0.0 and 1.0.")
        # [1차]: min_match_count가 양수인지 검증합니다.
        # [2차]: 최소 일치 지문 수는 1개 이상이어야 합니다.
        if self.min_match_count <= 0:
            raise ValueError("min_match_count must be positive.")
        # [1차]: max_frames가 지정된 경우 양수인지 검증합니다.
        # [2차]: 프레임 제한을 걸었다면 최소 1프레임 이상 처리하도록 확인합니다.
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when specified.")
