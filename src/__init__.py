"""Classic Computer Vision Motion Detection, Object Tracking, and Re-ID Package."""

# [1차]: src.background 모듈에서 배경 차분 담당 클래스 BackgroundSubtractor를 가져옵니다.
# [2차]: 요리하기 전 재료 손질 담당(배경 분리 담당자)을 주방으로 불러옵니다.
from src.background import BackgroundSubtractor

# [1차]: src.config 모듈에서 설정 데이터클래스 AppConfig를 가져옵니다.
# [2차]: 모든 레시피 설정(불 세기, 소금 양)이 적힌 설정표를 꺼내옵니다.
from src.config import AppConfig

# [1차]: src.detector 모듈에서 감지 결과 구조체 Detection과 감지기 MotionDetector를 가져옵니다.
# [2차]: 움직이는 물체를 찾아 좌표를 적어주는 관찰자와 기록 카드를 불러옵니다.
from src.detector import Detection, MotionDetector

# [1차]: src.feature_matcher 모듈에서 특징점 매칭기 FeatureMatcher와 결과 구조체 MatchResult를 가져옵니다.
# [2차]: 수배 전단지와 CCTV 화면을 대조해 범인을 찾는 감식반과 보고서 양식을 불러옵니다.
from src.feature_matcher import FeatureMatcher, MatchResult

# [1차]: src.tracker 모듈에서 추적 객체 TrackedObject와 추적기 Tracker 클래스를 가져옵니다.
# [2차]: 각 손님에게 번호표를 붙여 동선을 따라가는 안내원과 번호표 기록부를 불러옵니다.
from src.tracker import TrackedObject, Tracker

# [1차]: src.visualizer 모듈에서 화면 오버레이 렌더러 FrameVisualizer를 가져옵니다.
# [2차]: 분석 결과를 예쁜 박스와 선으로 화면에 그려주는 화가를 불러옵니다.
from src.visualizer import FrameVisualizer

# [1차]: 패키지 외부에서 `from src import *` 사용 시 노출할 공개 심볼 목록을 정의합니다.
# [2차]: 가게 진열대에 공식적으로 올려둘 대표 메뉴판 목록을 고정해 둡니다.
__all__ = [
    "AppConfig",
    "BackgroundSubtractor",
    "Detection",
    "FeatureMatcher",
    "FrameVisualizer",
    "MatchResult",
    "MotionDetector",
    "TrackedObject",
    "Tracker",
]
