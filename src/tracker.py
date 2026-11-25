"""Distance-based Multi-Object Tracker separated into its own module (tracker.py)."""

# [1차]: 타입 힌트의 지연 평가를 활성화합니다.
# [2차]: 클래스와 타입 이름을 유연하게 참조할 수 있게 합니다.
from __future__ import annotations

# [1차]: 최대 길이가 자동으로 제한되는 양방향 큐 deque를 임포트하여 궤적 메모리를 관리합니다.
# [2차]: 새 발자국이 찍히면 가장 오래된 발자국을 자동으로 밀어내는 컨베이어 벨트입니다.
from collections import deque

# [1차]: 추적 객체 상태를 표현하기 위해 dataclass와 가변 기본값 팩토리 field를 임포트합니다.
# [2차]: 객체마다 독립된 궤적 노트(deque)를 안전하게 나눠주기 위한 양식 도구입니다.
from dataclasses import dataclass, field

# [1차]: 두 점 사이의 유클리드 거리 계산(math.hypot)을 위해 math 표준 모듈을 임포트합니다.
# [2차]: 피타고라스 정리로 두 좌표 사이의 직선 거리를 재는 줄자입니다.
import math

# [1차]: 컬렉션 및 시퀀스 타입 힌트를 임포트합니다.
# [2차]: 어떤 상자에 어떤 자료가 들어가는지 표시하는 안내 라벨입니다.
from typing import Deque, Dict, List, Sequence, Tuple

# [1차]: 감지기 모듈에서 전달받는 Detection 데이터클래스를 임포트합니다.
# [2차]: 앞 단계 관찰자가 넘겨주는 물체 신분증 양식을 가져옵니다.
from src.detector import Detection


# [1차]: 추적 중인 개별 객체의 ID, 위치, 박스, 미등장 횟수, 궤적, 속도를 담는 데이터클래스입니다.
# [2차]: 번호표를 받은 손님 1명의 현재 위치, 안 보인 시간, 지나온 발자국, 뛰는 속도를 적은 수첩입니다.
@dataclass
class TrackedObject:
    """Represents a single tracked object with ID, bounding box, trajectory, and velocity."""

    # [1차]: 객체에 부여된 고유 정수 식별자입니다.
    # [2차]: 손님 가슴에 붙여둔 고유 번호표(ID 1, ID 2...)입니다.
    object_id: int
    # [1차]: 현재 프레임에서의 중심점 (cx, cy)입니다.
    # [2차]: 지금 이 손님이 서 있는 위치 좌표입니다.
    centroid: Tuple[int, int]
    # [1차]: 현재 프레임에서의 바운딩 박스 (x, y, w, h)입니다.
    # [2차]: 화면에 그릴 손님 테두리 사각형입니다.
    bbox: Tuple[int, int, int, int]
    # [1차]: 연속으로 화면에서 감지되지 않은 프레임 누적 횟수입니다.
    # [2차]: 손님이 시야에서 사라진 뒤 몇 프레임째 안 보이는지 세는 타이머입니다.
    disappeared: int = 0
    # [1차]: 최근 중심점 이동 경로를 저장하는 고정 길이 deque입니다.
    # [2차]: 손님이 걸어온 발자국 좌표를 최대 50개까지 모아두는 꼬리입니다.
    trajectory: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=50))
    # [1차]: 직전 프레임 대비 중심점 이동 벡터 (vx, vy)입니다.
    # [2차]: 방금 1프레임 사이에 가로·세로로 몇 픽셀 움직였는지 적어둔 속도계입니다.
    velocity: Tuple[float, float] = (0.0, 0.0)


# [1차]: 유클리드 거리 기반 최근접 매칭으로 다중 객체를 추적하는 독립 Tracker 클래스입니다.
# [2차]: 이전 화면의 손님들과 지금 화면의 손님들 사이 거리를 재어 가장 가까운 사람끼리 같은 번호표를 이어주는 안내원입니다.
class Tracker:
    """Euclidean distance-based centroid tracker with trajectory storage and optional velocity prediction."""

    # [1차]: 미등장 허용 프레임, 최대 매칭 거리, 궤적 길이, 속도 예측 사용 여부를 검증 후 초기화합니다.
    # [2차]: 안내원의 규칙(몇 초 기다릴지, 최대 보폭, 꼬리 길이, 관성 예측 여부)을 설정하고 출석부를 펼칩니다.
    def __init__(
        self,
        max_disappeared: int = 15,
        max_distance: float = 80.0,
        max_trajectory_len: int = 50,
        use_velocity_prediction: bool = False,
    ) -> None:
        # [1차]: 잘못된 하이퍼파라미터 입력을 즉시 차단합니다.
        # [2차]: 말이 안 되는 숫자(음수 대기 시간, 0 이하 보폭 등)가 들어오면 바로 경고를 냅니다.
        if max_disappeared < 0:
            raise ValueError("max_disappeared must be non-negative.")
        if max_distance <= 0:
            raise ValueError("max_distance must be positive.")
        if max_trajectory_len <= 0:
            raise ValueError("max_trajectory_len must be positive.")

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_trajectory_len = max_trajectory_len
        self.use_velocity_prediction = use_velocity_prediction

        # [1차]: 다음에 부여할 고유 ID 카운터(1부터 단조 증가)를 초기화합니다.
        # [2차]: 번호표 기계의 다음 숫자를 1번으로 맞춰둡니다.
        self.next_object_id: int = 1
        # [1차]: 현재 활성 추적 객체를 `{object_id: TrackedObject}` 딕셔너리로 관리합니다.
        # [2차]: 지금 매장에 있는 손님들의 출석부입니다.
        self.objects: Dict[int, TrackedObject] = {}
        # [1차]: 전체 실행 기간 동안 누적 등록된 고유 ID 총 개수를 기록합니다.
        # [2차]: 오늘 하루 동안 총 몇 명의 손님에게 번호표를 뽑아주었는지 세는 카운터입니다.
        self.total_registered: int = 0

    # [1차]: 새로 등장한 감지 객체에 고유 ID를 발급하고 출석부(self.objects)에 등록합니다.
    # [2차]: 새로 들어온 손님에게 새 번호표를 뽑아주고 첫 발자국을 기록합니다.
    def register(self, detection: Detection) -> TrackedObject:
        """Assign a new unique ID to a newly appeared object."""
        # [1차]: 현재 ID를 할당하고 다음 ID와 누적 등록 카운터를 1씩 증가시킵니다.
        # [2차]: 번호표를 한 장 뜯어주고 기계 숫자를 다음 번호로 넘깁니다.
        obj_id = self.next_object_id
        self.next_object_id += 1
        self.total_registered += 1

        # [1차]: 최대 길이 제한이 걸린 궤적 큐를 만들고 시작 중심점을 삽입합니다.
        # [2차]: 새 발자국 수첩을 열어 첫 번째 위치를 적어둡니다.
        traj: Deque[Tuple[int, int]] = deque(maxlen=self.max_trajectory_len)
        traj.append(detection.centroid)
        # [1차]: TrackedObject 인스턴스를 생성해 딕셔너리에 저장하고 반환합니다.
        # [2차]: 손님 정보를 출석부에 정식 등록합니다.
        tracked = TrackedObject(
            object_id=obj_id,
            centroid=detection.centroid,
            bbox=detection.bbox,
            disappeared=0,
            trajectory=traj,
            velocity=(0.0, 0.0),
        )
        self.objects[obj_id] = tracked
        return tracked

    # [1차]: 지정된 object_id를 추적 목록에서 안전하게 제거합니다(없어도 예외 없음).
    # [2차]: 매장을 떠난 손님의 번호표를 출석부에서 지워줍니다.
    def deregister(self, object_id: int) -> None:
        """Remove an object ID that has disappeared beyond the allowed threshold."""
        self.objects.pop(object_id, None)

    # [1차]: 거리 비교에 사용할 기준 좌표를 계산합니다(속도 예측 ON 시 현재 위치 + 직전 속도).
    # [2차]: 기본 모드는 "방금 서 있던 곳", 속도 예측 모드는 "뛰던 방향으로 한 걸음 더 간 예상 위치"를 기준으로 삼습니다.
    def _reference_point(self, obj: TrackedObject) -> Tuple[float, float]:
        """Compute matching reference point (current centroid or constant-velocity prediction)."""
        if self.use_velocity_prediction and len(obj.trajectory) >= 2:
            return (
                obj.centroid[0] + obj.velocity[0],
                obj.centroid[1] + obj.velocity[1],
            )
        return (float(obj.centroid[0]), float(obj.centroid[1]))

    # [1차]: math.hypot을 이용해 두 2D 점 사이의 유클리드 직선 거리를 수치적으로 안정하게 계산합니다.
    # [2차]: 가로 차이와 세로 차이로 대각선 직선 거리를 재는 피타고라스 줄자입니다.
    @staticmethod
    def _euclidean_distance(
        p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # [1차]: 현재 프레임의 감지 목록을 받아 기존 객체와 1:1 최근접 매칭·갱신·삭제·신규등록을 수행합니다.
    # [2차]: 새 장면이 올 때마다 기존 손님과 새 위치들을 짝지어 출석부를 최신화하는 핵심 두뇌입니다.
    def update(self, detections: Sequence[Detection]) -> Dict[int, TrackedObject]:
        """Update tracked objects for the current frame using greedy nearest-distance matching."""
        # [1차]: 현재 프레임에 감지된 객체가 0개면 모든 기존 객체의 미등장 카운터를 1씩 증가시킵니다.
        # [2차]: 화면이 텅 비었으면 기존 손님들의 "안 보인 시간"을 1씩 올리고 너무 오래 안 보이면 퇴장 처리합니다.
        if len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].disappeared += 1
                if self.objects[obj_id].disappeared > self.max_disappeared:
                    self.deregister(obj_id)
            return dict(self.objects)

        # [1차]: 기존에 추적 중인 객체가 하나도 없으면 이번 감지를 모두 신규 등록합니다.
        # [2차]: 출석부가 비어 있으니 지금 보이는 사람들에게 차례로 새 번호표를 나눠줍니다.
        if len(self.objects) == 0:
            for det in detections:
                self.register(det)
            return dict(self.objects)

        # [1차]: 결정론적 순서를 위해 기존 ID를 오름차순 정렬하고 모든 (기존객체, 신규감지) 쌍의 거리를 계산합니다.
        # [2차]: 모든 기존 손님과 새 위치 조합 사이의 거리를 줄자로 하나씩 재어 표에 적어둡니다.
        object_ids = sorted(self.objects.keys())
        pairs: List[Tuple[float, int, int]] = []
        for obj_id in object_ids:
            ref_pt = self._reference_point(self.objects[obj_id])
            for det_idx, det in enumerate(detections):
                dist = self._euclidean_distance(
                    ref_pt, (float(det.centroid[0]), float(det.centroid[1]))
                )
                # [1차]: (거리, object_id, det_idx) 튜플로 저장하여 동률 거리 시 낮은 ID가 우선 매칭되게 합니다.
                # [2차]: 거리가 똑같이 10px로 비길 때 운에 맡기지 않고 먼저 온 번호표(작은 ID)에게 우선권을 줍니다.
                pairs.append((dist, obj_id, det_idx))

        # [1차]: 거리 오름차순(동률 시 ID, 인덱스 오름차순)으로 후보 쌍을 정렬합니다.
        # [2차]: 가장 가까운 커플 후보부터 차례대로 줄을 세웁니다.
        pairs.sort(key=lambda item: (item[0], item[1], item[2]))

        # [1차]: 1:1 매칭을 보장하기 위해 이미 짝지어진 기존 ID와 신규 감지 인덱스를 집합으로 추적합니다.
        # [2차]: 이미 짝을 찾은 손님과 위치에 "매칭 완료" 도장을 찍어 중복 배정을 막습니다.
        used_object_ids = set()
        used_det_indices = set()

        # [1차]: 가장 가까운 쌍부터 탐욕(Greedy) 방식으로 1:1 매칭을 확정합니다.
        # [2차]: 줄 선 후보 중 제일 가까운 쌍부터 짝을 지어줍니다.
        for dist, obj_id, det_idx in pairs:
            # [1차]: 정렬되어 있으므로 현재 거리가 max_distance를 넘으면 이후 쌍도 모두 초과하므로 즉시 중단합니다.
            # [2차]: 최대 보폭보다 먼 후보는 같은 사람일 수 없으므로 여기서 매칭을 끝냅니다.
            if dist > self.max_distance:
                break
            # [1차]: 이미 매칭된 객체나 감지는 건너뜁니다.
            # [2차]: 둘 중 하나라도 이미 짝이 있으면 패스합니다.
            if obj_id in used_object_ids or det_idx in used_det_indices:
                continue

            # [1차]: 매칭 확정된 객체의 속도, 중심점, 바운딩 박스, 미등장 카운터, 궤적을 갱신합니다.
            # [2차]: "아, 아까 그 손님이 여기로 이동했구나!" 하고 위치·속도·발자국을 업데이트합니다.
            det = detections[det_idx]
            obj = self.objects[obj_id]
            vx = float(det.centroid[0] - obj.centroid[0])
            vy = float(det.centroid[1] - obj.centroid[1])
            obj.velocity = (vx, vy)
            obj.centroid = det.centroid
            obj.bbox = det.bbox
            obj.disappeared = 0
            obj.trajectory.append(det.centroid)

            used_object_ids.add(obj_id)
            used_det_indices.add(det_idx)

        # [1차]: 이번 프레임에서 짝을 찾지 못한 기존 객체는 미등장 카운터를 늘리고 한도 초과 시 삭제합니다.
        # [2차]: 이번에 안 보인 손님은 대기 카운트를 올리고, 너무 오래 안 돌아오면 출석부에서 지웁니다.
        for obj_id in object_ids:
            if obj_id not in used_object_ids:
                self.objects[obj_id].disappeared += 1
                if self.objects[obj_id].disappeared > self.max_disappeared:
                    self.deregister(obj_id)

        # [1차]: 기존 객체와 짝지어지지 않은 신규 감지는 새 ID를 발급해 등록합니다.
        # [2차]: 짝이 남는 새 위치는 새로 입장한 손님이므로 새 번호표를 뽑아줍니다.
        for det_idx, det in enumerate(detections):
            if det_idx not in used_det_indices:
                self.register(det)

        # [1차]: 외부에서 내부 딕셔너리를 직접 변형하지 않도록 얕은 복사본을 반환합니다.
        # [2차]: 원본 출석부는 금고에 두고 복사본을 화가(시각화 모듈)에게 건네줍니다.
        return dict(self.objects)
