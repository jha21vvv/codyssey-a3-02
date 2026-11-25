# PLAN.md — 고전적 컴퓨터 비전 기반 실시간 객체 감지·추적 및 재인식 시스템

> 본 계획서는 [코디세이 a3-02.md](코디세이%20a3-02.md)의 모든 기능·제약·보고서 요구사항과 [PROJECT_WORKFLOW_PROMPT_TEMPLATE.md](PROJECT_WORKFLOW_PROMPT_TEMPLATE.md)의 8단계 표준 워크플로우를 충족하기 위한 상세 설계서입니다.

---

## 1. 과제 목표 및 핵심 제약 사항

### 1.1 핵심 목표
1. **실시간 영상 분석 애플리케이션**: 웹캠 또는 비디오 파일에서 프레임 단위로 움직임을 감지하고, 고유 ID·바운딩 박스·이동 궤적·FPS를 오버레이하며 키보드(`q`, `s`, `p`)로 제어한다.
2. **배경 차분 + 객체 감지**: MOG2(및 선택적 `absdiff`) 배경 차분과 모폴로지 연산(열림/닫힘), 윤곽선·면적 필터링으로 전경 객체를 추출한다.
3. **거리 기반 객체 추적 (`tracker.py` 분리)**: 유클리드 거리 기반 ID 매칭, 일정 프레임 미등장 시 ID 삭제, 신규 객체 ID 부여, 궤적(trajectory) 저장을 독립 모듈로 구현한다.
4. **특징점 기반 특정 물체 재인식**: 등록 이미지 1장을 ORB/SIFT로 특징점 추출 후 매 프레임 매칭하여 `"TARGET DETECTED"`(Target Found)를 표시한다.
5. **정량적 성능 측정 및 한계 분석 보고서**: 조건별(단일 이동, 두 객체 겹침, 일시 정지, 조명 변화) 추적 성능(ID Switch, 추적 실패)과 특징점 매칭 성능(정면, 30°/60° 회전, 30%/50% 가림)을 실측하고, 고전적 컴퓨터 비전의 한계와 딥러닝의 필요성을 증명한다.

### 1.2 엄격한 제약 사항 (Constraint Enforcement)
- **딥러닝 모델 금지**: YOLO, SSD, R-CNN 등 딥러닝 객체 탐지 모델 사용 금지
- **OpenCV `dnn` 모듈 금지**: `cv2.dnn` import 및 호출 금지 (단위 테스트에서 정적 검사로 강제)
- **고수준 AI 라이브러리 금지**: MediaPipe, torch, tensorflow, ultralytics 등 사용 금지
- **필수 파일 분리**: `Tracker` 클래스는 반드시 별도 파일 `src/tracker.py`로 분리
- **허용 라이브러리**: Python 3.10+ 표준 라이브러리, `opencv-python` (`cv2`), `numpy`, (시각화·다이어그램 전용) `matplotlib`

---

## 2. 디렉터리 및 모듈 아키텍처 (SRP 설계)

```text
codyssey-a3-02/
├── PLAN.md                                 # [Step 1] 설계 계획서
├── README.md                               # 프로젝트 소개, 실행법, 다이어그램, 평가자 검증 공간
├── REPORT.md                               # [Step 5] 정량적 한계 분석 + 종합 엔지니어링 보고서
├── 스터디.md                                # [Step 4] 과제 전문 + 1:1 매핑표 + 주니어 비유 + 다이어그램
├── NOTEBOOKLM_FULL_SOURCE.txt              # [Step 6] src/*.py 순수 소스코드 통합 텍스트 (2단 주석 포함)
├── NOTEBOOKLM_GUIDE.md                     # [Step 6] NotebookLM 업로드 목록 및 팟캐스트 기획서
├── study/
│   └── 핵심_코드_스터디.md                   # [Step 8] 핵심 코드 8선 단어분해 + 비유 학습서
├── src/                                    # [Step 2 & 7] 핵심 구현 모듈 (SRP 분리, 2단 주석 적용)
│   ├── __init__.py                         # 패키지 초기화 및 공개 API 노출
│   ├── config.py                           # AppConfig 설정 데이터클래스 및 유효성 검증
│   ├── background.py                       # BackgroundSubtractor (MOG2 / absdiff + 모폴로지)
│   ├── detector.py                         # MotionDetector (윤곽선 추출, 면적 필터링, 중심점 계산)
│   ├── tracker.py                          # Tracker 클래스 (거리 기반 ID 매칭, 소멸 관리, 궤적, 속도 예측 옵션)
│   ├── feature_matcher.py                  # FeatureMatcher (ORB / SIFT 특징점 추출 및 Lowe Ratio 매칭)
│   ├── visualizer.py                       # 프레임 오버레이 렌더링(박스, ID, 궤적, FPS, TARGET DETECTED)
│   ├── app.py                              # VideoAnalyzerApp (VideoCapture 루프, FPS, q/s/p 제어, Headless)
│   └── main.py                             # CLI 엔트리포인트 (argparse, UTF-8 콘솔 방어)
├── tests/                                  # [Step 3] 초고속 단위 테스트 (Mock + 합성 프레임)
│   ├── test_background_and_detector.py
│   ├── test_tracker.py
│   ├── test_feature_matcher.py
│   └── test_app_and_constraints.py
├── scripts/
│   ├── run_experiments.py                  # 정량 성능 측정 자동화 (4가지 추적 조건 + 5가지 매칭 조건 + 학습률 3종)
│   ├── generate_diagrams_matplotlib.py     # [Step 4] Matplotlib 고해상도 아키텍처/실험 다이어그램 생성
│   ├── verify_requirements.py              # [Step 5] 평가자 원클릭 시각적 검증 러너 (터미널 테이블 / JSON)
│   └── build_notebooklm_source.py          # [Step 6] NOTEBOOKLM_FULL_SOURCE.txt 자동 빌더
├── diagrams/                               # Matplotlib 생성 PNG 다이어그램 (상대경로 임베딩)
├── data/                                   # 공개 테스트 영상 및 타겟 이미지
│   ├── videos/                             # vtest.avi, one-by-one-person-detection.mp4 등
│   └── targets/                            # box.png, box_in_scene.png
└── output/                                 # 스냅샷, 실험 결과 JSON/CSV 저장소
```

---

## 3. 모듈별 상세 인터페이스 및 예외 처리 규격

### 3.1 `src/config.py`
- `AppConfig` dataclass:
  - `source`: `str` (`"0"` 웹캠 또는 비디오 경로)
  - `bg_method`: `"mog2" | "absdiff"` (기본 `"mog2"`)
  - `learning_rate`: `float` (기본 `0.005`, `-1` 또는 `0.0 <= lr <= 1.0`)
  - `morph_kernel_size`: `int` (홀수 양수, 기본 `5`)
  - `min_area`: `int` (기본 `500`), `max_area`: `int` (기본 `100000`)
  - `max_disappeared`: `int` (기본 `15` 프레임)
  - `max_distance`: `float` (기본 `80.0` 픽셀)
  - `max_trajectory_len`: `int` (기본 `50` 포인트)
  - `use_velocity_prediction`: `bool` (기본 `False`, 규칙 기반 개선 비교 실험용)
  - `target_image`: `Optional[str]` (예: `"data/targets/box.png"`)
  - `feature_method`: `"orb" | "sift"` (기본 `"orb"`)
  - `ratio_thresh`: `float` (기본 `0.75`)
  - `min_match_count`: `int` (기본 `15`)
  - `snapshot_dir`: `str` (기본 `"output/snapshots"`)
  - `headless`: `bool` (기본 `False`, 자동화/서버 환경 지원)
- 잘못된 값(음수 면적, 범위 초과 학습률, 짝수 커널 등) 입력 시 명확한 `ValueError` 발생.

### 3.2 `src/background.py` (`BackgroundSubtractor`)
- `MOG2` (`cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)`)와 `absdiff`(초기 프레임/지수이동평균 기반 차영상) 모두 지원.
- 그림자 픽셀(값 `127`) 제거를 위해 이진화(`threshold(..., 200, 255, THRESH_BINARY)`) 적용.
- 모폴로지 연산: `MORPH_OPEN`(소금 노이즈 제거) ➔ `MORPH_CLOSE`(객체 내부 구멍 메우기).
- 빈 프레임(`None` 또는 `size == 0`) 입력 시 `ValueError` 방어.

### 3.3 `src/detector.py` (`MotionDetector`, `Detection`)
- `Detection` dataclass: `bbox: tuple[int, int, int, int]`, `centroid: tuple[int, int]`, `area: float`.
- `cv2.findContours(..., RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`로 외곽선 추출 후 `min_area <= area <= max_area` 필터링.
- 결정론적 순서 보장을 위해 `(centroid_x, centroid_y, -area)` 기준으로 정렬.

### 3.4 `src/tracker.py` (`Tracker`, `TrackedObject`)
- `TrackedObject`: `object_id`, `centroid`, `bbox`, `disappeared`, `trajectory` (`deque(maxlen=max_trajectory_len)`), `velocity`.
- **유클리드 거리 매칭**:
  - 기존 객체(M개)와 신규 감지(N개) 간 거리 행렬 계산.
  - 거리 오름차순 + 동률 시 `(object_id, detection_index)` 사전순 타이 브레이킹으로 탐욕(Greedy) 1:1 매칭 수행.
  - 거리가 `max_distance` 초과 시 매칭 거절(신규 객체로 분리).
- **규칙 기반 개선 옵션 (`use_velocity_prediction=True`)**:
  - 이전 2프레임 이동 벡터(`vx, vy`)로 예상 중심점을 추정한 뒤 거리를 계산하여, 겹침 구간에서 규칙 추가의 한계를 정량 비교.
- **소멸 및 신규 등록**:
  - 미매칭 기존 객체는 `disappeared += 1`, `disappeared > max_disappeared`이면 삭제.
  - 미매칭 신규 감지는 `next_object_id` 부여 후 등록.

### 3.5 `src/feature_matcher.py` (`FeatureMatcher`, `MatchResult`)
- 한글 경로 포함 Windows 환경 방어: `cv2.imread` 대신 `np.fromfile` + `cv2.imdecode` 사용!
- `ORB` (`nfeatures=1000`, `NORM_HAMMING`) 및 `SIFT` (`NORM_L2`) 지원.
- `knnMatch(k=2)` + Lowe's Ratio Test(`m.distance < ratio_thresh * n.distance`).
- `good_matches >= min_match_count`이면 `detected = True`, 매칭률 `match_rate = good_matches / total_target_kp * 100`.
- 특징점이 0개이거나 2개 미만인 예외 프레임에서도 크래시 없이 안전하게 `detected=False` 반환.

### 3.6 `src/visualizer.py` (`FrameVisualizer`)
- 객체별 고유 색상(ID 기반 결정론적 팔레트)으로 바운딩 박스, `ID #N` 라벨, 중심점, 궤적 폴리라인(`cv2.polylines`) 렌더링.
- 좌상단 HUD: `FPS`, `Active Objects`, `BG Method / LR`, `PAUSED` 상태 표시.
- 타겟 감지 시 상단 배너에 `"TARGET DETECTED (Target Found)"` + 매칭 수/매칭률 표시.
- 스냅샷 저장 시 Windows 한글 경로에서도 깨지지 않도록 `cv2.imencode` + `Path.write_bytes` 사용.

### 3.7 `src/app.py` & `src/main.py`
- `cv2.VideoCapture`로 정수 문자열(`"0"`)은 웹캠, 파일 경로는 비디오로 열기.
- 키보드 제어:
  - `q` / `ESC`: 종료
  - `s`: 현재 오버레이 프레임을 타임스탬프 PNG로 저장
  - `p`: 일시정지 토글 (일시정지 중에도 `s` 저장 및 `q` 종료 가능)
- Headless 모드(`--headless`, `--max-frames`, `--save-video`) 지원으로 GUI 없는 환경 및 자동 테스트에서도 완벽 실행.

---

## 4. 정량적 실험 설계 (`scripts/run_experiments.py`)

1. **추적 성능 측정 (4가지 조건, 각 20~30초 구간)**:
   - 단일 객체 이동 (10회): `data/videos/one-by-one-person-detection.mp4`
   - 두 객체 겹침 (10회, 필수): `data/videos/vtest.avi` 및 `data/videos/people-detection.mp4`
   - 객체 일시 정지 (5회): `one-by-one-person-detection.mp4` 정지 구간 + 제어된 일시 정지 시퀀스
   - 조명 변화 (5회): `vtest.avi` 기반 단계적/급격한 조명 밝기 변화 시퀀스 + `person-bicycle-car-detection.mp4`
   - **추가 비교 실험**: 기본 거리 추적 vs 속도 예측 규칙 추가 추적 (`use_velocity_prediction=True`)
2. **배경 학습률 비교 실험**:
   - `learning_rate` 3종 (`0.001`, `0.01`, `0.1`)에 따른 전경 픽셀 비율 및 정지 객체 흡수 속도 측정.
3. **특징점 매칭 성능 측정 (ORB vs SIFT)**:
   - `data/targets/box.png` 기준, `data/targets/box_in_scene.png` 및 제어된 5가지 조건(정면, 30° 회전, 60° 회전, 30% 가림, 50% 가림)에서 추출 특징점 수·매칭 수·매칭률(%) 실측.

---

## 5. 단계별 마일스톤

| 단계 | 작업 내용 | 주요 산출물 |
| :---: | :--- | :--- |
| Step 1 | 요구사항 분해 및 계획 수립 | `PLAN.md` |
| Step 2 | SRP 기반 8개 모듈 구현 | `src/*.py` |
| Step 3 | 초고속 Mock/합성 단위 테스트 작성 | `tests/test_*.py` |
| Step 4 | 정량 실험 수행, Matplotlib 다이어그램 생성, 스터디 문서 작성 | `scripts/run_experiments.py`, `diagrams/*.png`, `스터디.md` |
| Step 5 | 엣지 케이스 보완, 종합 분석 보고서, 평가자 러너, README | `REPORT.md`, `scripts/verify_requirements.py`, `README.md` |
| Step 6~7 | 전 라인 1·2차 2단 주석 내재화 및 NotebookLM 통합 패키지 | `src/*.py`, `NOTEBOOKLM_FULL_SOURCE.txt`, `NOTEBOOKLM_GUIDE.md` |
| Step 8 | 핵심 코드 8선 단어분해·비유 학습서 작성 | `study/핵심_코드_스터디.md` |
