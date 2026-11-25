# NOTEBOOKLM_GUIDE.md — Google NotebookLM Audio Overview 학습 및 팟캐스트 기획서

> ## 📥 [필수] Google NotebookLM 업로드 소스 체크리스트
>
> NotebookLM 새 노트북을 만들고 아래 **4개 핵심 파일**을 그대로 업로드하세요. (`.py` 파일은 NotebookLM이 직접 지원하지 않으므로, 순수 소스 코드 전체를 묶어둔 `NOTEBOOKLM_FULL_SOURCE.txt`를 업로드합니다.)
>
> - [ ] **`NOTEBOOKLM_FULL_SOURCE.txt`** — `src/*.py` 9개 프로덕션 모듈 전체 소스 코드 (1·2차 2단 주석 포함, 테스트/가이드 제외)
> - [ ] **`스터디.md`** — 원본 과제 요구사항 전문, 1:1 매핑표, 3개월 차 주니어 눈높이 일상 비유, 아키텍처 다이어그램 해설
> - [ ] **`REPORT.md`** — 조건별 ID Switch/추적 실패 정량 실험, 배경 학습률 트레이드오프, ORB vs SIFT 벤치마크, 딥러닝 필요성 분석
> - [ ] **`study/핵심_코드_스터디.md`** *(선택 추가)* — 실무·면접 재사용 빈도가 높은 핵심 코드 8선 단어분해 및 비유 학습서

---

## 1. 팟캐스트 컨셉 및 호스트 2인 설정

- **호스트 A (시니어 컴퓨터 비전 아키텍트)**: 10년 차 실무 엔지니어. 왜 YOLO 같은 딥러닝 모델을 금지하고 OpenCV MOG2, 모폴로지, 유클리드 거리 추적기, ORB/SIFT를 직접 바닥부터 짰는지 공학적 관점에서 설명합니다.
- **호스트 B (3개월 차 주니어 멘토 & 진행자)**: 갓 입사한 개발자의 눈높이에서 "두 사람이 지나가다 겹쳤는데 왜 번호표가 바뀌나요?", "학습률을 올리면 조명은 빨리 적응하는데 왜 서 있는 사람이 사라지나요?" 같은 핵심 질문을 던지고, 일상 비유로 정리합니다.

### 대화 유도용 핵심 질문 리스트
1. "동영상은 결국 사진 여러 장인데, 어떻게 딥러닝 없이도 움직이는 사람만 쏙 뽑아낼 수 있나요?"
2. "`Tracker` 클래스를 왜 굳이 `src/tracker.py`로 분리했고, 거리가 똑같이 비길 때(동률) 왜 `(dist, object_id, det_idx)` 순서를 강제했나요?"
3. "두 사람이 겹쳐 지나갈 때 ID가 뒤바뀌는 문제를 '속도 예측 규칙'으로 고쳐보려 했는데, 왜 완벽하게 해결되지 않았나요?"
4. "배경 학습률(`learning_rate`)을 `0.001`, `0.01`, `0.1`로 바꿨을 때 나타나는 '느린 변화 vs 빠른 변화' 딜레마는 무엇인가요?"
5. "ORB와 SIFT로 `box.png`를 30°·60° 돌리고 30%·50% 가려봤을 때 왜 매칭률이 급감했나요? 이게 딥러닝(CNN/Re-ID)이 필요한 이유와 어떻게 연결되나요?"

---

## 2. 핵심 기술적 결정 3가지 & 일상 비유 모음

| # | 핵심 기술적 결정 | 3개월 차 주니어 눈높이 일상 비유 |
| :---: | :--- | :--- |
| **1** | **MOG2 그림자(127) 제거 + 모폴로지 열림/닫힘 파이프라인 (`src/background.py`)** | CCTV 바닥에 떨어진 먼지(소금 노이즈)를 먼저 빗자루(`MORPH_OPEN`)로 쓸어낸 뒤, 사람 몸통 안쪽에 뚫린 작은 구멍을 퍼티(`MORPH_CLOSE`)로 메워 깔끔한 실루엣 도장을 만드는 과정 |
| **2** | **결정론적 탐욕 1:1 유클리드 거리 매칭 & 타이 브레이킹 (`src/tracker.py`)** | 매장 손님들에게 번호표를 붙이고, 다음 순간 가장 가까운 위치에 선 사람을 같은 손님으로 이어주는 안내원. 단, 두 손님이 똑같이 10px 떨어져 비기면 운에 맡기지 않고 먼저 입장한 번호표(작은 ID)에게 우선권을 부여 |
| **3** | **Lowe's Ratio Test + Windows 한글 경로 안전 I/O (`src/feature_matcher.py`, `src/visualizer.py`)** | 수배 사진 지문과 CCTV 지문을 대조할 때 1등 후보가 2등 후보보다 압도적으로(0.75배 미만 거리) 닮았을 때만 합격시키는 깐깐한 감식반. 또한 OpenCV가 한글 폴더를 못 읽는 함정을 `np.fromfile` + `cv2.imdecode`로 원천 차단 |

---

## 3. 3개월 차 주니어가 반드시 깨달아야 할 청취 포인트 3가지

1. **위치 정보(`x, y`)만으로는 정체성(Identity)을 지킬 수 없다**:
   두 객체가 교차하며 전경 마스크가 하나로 뭉치는 순간, 거리 기반 추적기는 외형(옷 색깔·얼굴)을 모르기 때문에 ID Switch가 구조적으로 발생합니다. 속도 예측 규칙을 덧붙여도 교차 직후 방향을 틀면 바로 무너집니다.
2. **규칙 기반(Hand-crafted) 시스템의 피할 수 없는 파라미터 트레이드오프**:
   MOG2 학습률을 낮추면(`0.001`) 잠시 멈춘 사람은 오래 기억하지만 조명 변화 복구에 111프레임이 걸리고, 높이면(`0.1`) 조명은 1프레임에 복구되지만 움직이는 사람마저 배경에 흡수됩니다.
3. **딥러닝은 "마법"이 아니라 고전 비전의 3대 병목을 풀기 위한 필연적 진화**:
   외형 임베딩(DeepSORT/Re-ID)으로 겹침 문제를, 데이터 증강 기반 학습 특징(CNN/ViT)으로 회전·가림 취약성을, 시맨틱 세그멘테이션으로 조명 변화 취약성을 극복한다는 흐름을 체감하는 것이 핵심입니다.

---

## 4. NotebookLM Customize Audio Overview 전용 프롬프트

### 🇰🇷 한국어 프롬프트 (그대로 복사해 붙여넣기)
```text
두 명의 호스트(10년 차 컴퓨터 비전 시니어 아키텍트와 3개월 차 주니어 개발자 멘토)가 진행하는 15분짜리 기술 딥다이브 팟캐스트를 한국어로 만들어줘.
1. 도입: YOLO나 딥러닝 없이 순수 OpenCV(MOG2, 모폴로지, 윤곽선, 유클리드 거리 Tracker, ORB/SIFT)만으로 실시간 객체 추적 및 재인식 앱을 만든 배경을 소개해줘.
2. 코드 아키텍처: NOTEBOOKLM_FULL_SOURCE.txt의 모듈 분리(src/background.py, detector.py, tracker.py, feature_matcher.py)와 일상 비유(빗자루, 번호표 안내원, 지문 감식반)를 살려 쉽게 풀어줘.
3. 정량 실험 하이라이트: REPORT.md의 실제 수치(겹침 시 기본 ID Switch vs 속도 예측 규칙 비교, 학습률 0.001/0.01/0.1의 111프레임 vs 1프레임 트레이드오프, ORB vs SIFT의 50% 가림 매칭률 격차)를 꼭 언급해줘.
4. 결론: "왜 위치만으로는 객체를 구분할 수 없는가?"와 "딥러닝이 왜 필수적인가?"에 대한 명쾌한 인사이트로 마무리해줘.
```

### 🇺🇸 English Prompt (Copy & Paste into NotebookLM)
```text
Create a 15-minute engaging Deep Dive podcast hosted by a 10-year Senior Computer Vision Architect and a friendly Mentor for 3-month junior developers.
1. Frame the mission: Building a real-time motion detection, multi-object tracking, and target re-identification system using strictly Classic OpenCV (MOG2, morphology, contours, Euclidean centroid Tracker in src/tracker.py, ORB/SIFT) with zero deep learning (no YOLO, no cv2.dnn).
2. Walk through the clean SRP architecture in NOTEBOOKLM_FULL_SOURCE.txt using intuitive everyday analogies (broom/putty for morphological open/close, ticket-number usher for Tracker, fingerprint lab with Lowe's Ratio Test for FeatureMatcher).
3. Highlight the real quantitative benchmark findings from REPORT.md: crossing ID switches (baseline vs velocity-prediction rule), the dramatic learning-rate tradeoff (0.001 vs 0.01 vs 0.1), and ORB vs SIFT degradation under 30°/60° rotation and 30%/50% occlusion.
4. Conclude with why hand-crafted rules hit a structural ceiling and why learned appearance features (Deep Learning / DeepSORT) are indispensable in production CV systems.
```
