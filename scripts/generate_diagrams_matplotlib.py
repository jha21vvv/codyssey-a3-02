"""Generate 4 high-resolution architecture and benchmark diagrams into diagrams/*.png."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DIAGRAMS_DIR = ROOT / "diagrams"
RESULTS_PATH = ROOT / "output" / "experiment_results.json"


def _configure_korean_font() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2.0,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.65,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        x + w / 2,
        y + h * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.5,
        color="#374151",
    )


def _draw_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#4B5563", lw=2.0, mutation_scale=15),
    )
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.03,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#374151",
        )


def generate_architecture_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=180)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    fig.patch.set_facecolor("#F9FAFB")

    ax.text(
        0.5,
        0.94,
        "Codyssey A3-02 실시간 영상 분석 및 객체 추적 파이프라인 아키텍처",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.885,
        "Classic Computer Vision Only (No YOLO / No cv2.dnn / No MediaPipe) — SRP 모듈 분리 구조",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#6B7280",
    )

    # Top main tracking pipeline
    _draw_box(ax, 0.03, 0.56, 0.16, 0.22, "1. 영상 입력", "src/app.py\nVideoCapture\n(웹캠 / MP4·AVI)", "#DBEAFE", "#2563EB")
    _draw_box(ax, 0.24, 0.56, 0.19, 0.22, "2. 배경 차분", "src/background.py\nMOG2 / absdiff\n열림·닫힘 모폴로지", "#E0E7FF", "#4F46E5")
    _draw_box(ax, 0.48, 0.56, 0.18, 0.22, "3. 객체 감지", "src/detector.py\nfindContours\n면적 필터·중심점", "#FEF3C7", "#D97706")
    _draw_box(ax, 0.71, 0.56, 0.24, 0.22, "4. 객체 추적 (별도 파일)", "src/tracker.py (Tracker)\n유클리드 거리 매칭\nID 유지·삭제·궤적 저장", "#DCFCE7", "#16A34A")

    _draw_arrow(ax, 0.19, 0.67, 0.24, 0.67, "BGR 프레임")
    _draw_arrow(ax, 0.43, 0.67, 0.48, 0.67, "이진 마스크")
    _draw_arrow(ax, 0.66, 0.67, 0.71, 0.67, "Detection 목록")

    # Bottom branch: Target Re-ID & Visualizer
    _draw_box(ax, 0.24, 0.16, 0.24, 0.22, "5. 특정 물체 재인식", "src/feature_matcher.py\nORB / SIFT + BFMatcher\nLowe Ratio Test → TARGET DETECTED", "#FCE7F3", "#DB2777")
    _draw_box(ax, 0.58, 0.16, 0.37, 0.22, "6. 시각화 & 키보드 제어", "src/visualizer.py + src/app.py\n바운딩 박스·ID·궤적·FPS 오버레이\n[q] 종료 / [s] 스냅샷 / [p] 일시정지", "#F3E8FF", "#9333EA")

    _draw_arrow(ax, 0.11, 0.56, 0.24, 0.27, "원본 프레임")
    _draw_arrow(ax, 0.83, 0.56, 0.83, 0.38, "TrackedObject")
    _draw_arrow(ax, 0.48, 0.27, 0.58, 0.27, "MatchResult")

    out = DIAGRAMS_DIR / "01_system_architecture.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_id_switch_diagram() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), dpi=180)
    fig.patch.set_facecolor("#F9FAFB")
    fig.suptitle(
        "거리 기반 추적기(Tracker)의 겹침(Crossing) 시 ID Switch 발생 메커니즘",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        y=1.02,
    )

    titles = [
        "Step 1: 교차 전 (t = 10)\n정상 거리 매칭 유지",
        "Step 2: 교차 겹침 (t = 20)\n전경 덩어리 병합 & 중심점 접근",
        "Step 3: 교차 후 (t = 25)\n최근접 거리 역전 → ID 1↔2 뒤바뀜",
    ]

    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_facecolor("#FFFFFF")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xticks([])
        ax.set_yticks([])

    # Panel 1: Before crossing
    ax = axes[0]
    ax.scatter([25], [50], s=320, color="#2563EB", zorder=5)
    ax.scatter([75], [50], s=320, color="#DC2626", zorder=5)
    ax.text(25, 62, "ID 1 (객체 A)", ha="center", color="#1D4ED8", fontweight="bold")
    ax.text(75, 62, "ID 2 (객체 B)", ha="center", color="#B91C1C", fontweight="bold")
    ax.annotate("", xy=(42, 50), xytext=(28, 50), arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=2.5))
    ax.annotate("", xy=(58, 50), xytext=(72, 50), arrowprops=dict(arrowstyle="-|>", color="#DC2626", lw=2.5))
    ax.text(50, 20, "서로 멀리 떨어져 있어\n유클리드 거리 매칭 정확함", ha="center", fontsize=9, color="#374151")

    # Panel 2: Overlap
    ax = axes[1]
    circle = plt.Circle((50, 50), 14, color="#F59E0B", alpha=0.30, zorder=2)
    ax.add_patch(circle)
    ax.scatter([47], [50], s=280, color="#2563EB", alpha=0.85, zorder=5)
    ax.scatter([53], [50], s=280, color="#DC2626", alpha=0.85, zorder=5)
    ax.text(50, 72, "배경 차분 마스크가\n1개 덩어리로 합쳐짐!", ha="center", color="#B45309", fontweight="bold", fontsize=9.5)
    ax.text(50, 18, "외형(색상·옷) 정보가 없으므로\n위치 좌표 (x, y)만 기억", ha="center", fontsize=9, color="#374151")

    # Panel 3: After crossing (ID Switch!)
    ax = axes[2]
    # True A moved right (72), True B moved left (28), but IDs swapped!
    ax.scatter([28], [50], s=320, color="#DC2626", edgecolor="#2563EB", linewidth=3.0, zorder=5)
    ax.scatter([72], [50], s=320, color="#2563EB", edgecolor="#DC2626", linewidth=3.0, zorder=5)
    ax.text(28, 63, "실제 B → ID 1 부여!", ha="center", color="#1D4ED8", fontweight="bold", fontsize=9.5)
    ax.text(72, 63, "실제 A → ID 2 부여!", ha="center", color="#B91C1C", fontweight="bold", fontsize=9.5)
    ax.plot([25, 47, 28], [46, 46, 46], color="#2563EB", linestyle=":", lw=2)
    ax.plot([75, 53, 72], [54, 54, 54], color="#DC2626", linestyle=":", lw=2)
    ax.text(50, 18, "ID Switch 발생!\n→ 외형 특징 학습(딥러닝) 필요", ha="center", fontsize=9, fontweight="bold", color="#991B1B")

    out = DIAGRAMS_DIR / "02_tracker_id_switch_mechanism.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_tracking_and_lr_diagram(results: dict) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0), dpi=180)
    fig.patch.set_facecolor("#F9FAFB")

    # Left: Tracking performance across 4 conditions + velocity rule comparison
    t = results["tracking"]
    labels = [
        "단일 이동\n(10회)",
        "겹침-기본\n(10회)",
        "겹침-속도규칙\n(10회)",
        "일시 정지\n(5회)",
        "조명 변화\n(5회)",
    ]
    switches = [
        t["single_object"]["id_switches"],
        t["two_objects_crossing_baseline"]["id_switches"],
        t["two_objects_crossing_velocity_rule"]["id_switches"],
        t["object_pause"]["id_switches"],
        t["lighting_change"]["id_switches"],
    ]
    failures = [
        t["single_object"]["failures"],
        t["two_objects_crossing_baseline"]["failures"],
        t["two_objects_crossing_velocity_rule"]["failures"],
        t["object_pause"]["failures"],
        t["lighting_change"]["failures"],
    ]

    x = np.arange(len(labels))
    width = 0.36
    b1 = ax1.bar(x - width / 2, switches, width, label="ID Switch (회)", color="#EF4444")
    b2 = ax1.bar(x + width / 2, failures, width, label="추적 실패 (회)", color="#F59E0B")
    ax1.set_title("조건별 객체 추적 실패 및 ID Switch 실측 결과", fontsize=11.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("발생 횟수 (회)")
    ax1.set_ylim(0, max(max(switches), max(failures)) + 2.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{int(h)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Right: Learning rate tradeoff
    lr_rows = results["learning_rates"]
    lr_labels = [f"lr={r['learning_rate']}" for r in lr_rows]
    absorb_vals = [200 if isinstance(r["stopped_object_absorbed_frames"], str) else r["stopped_object_absorbed_frames"] for r in lr_rows]
    recover_vals = [200 if isinstance(r["lighting_recovery_frames"], str) else r["lighting_recovery_frames"] for r in lr_rows]

    x2 = np.arange(len(lr_labels))
    c1 = ax2.bar(x2 - width / 2, absorb_vals, width, label="정지 객체 소실까지 프레임 (길수록 유리)", color="#3B82F6")
    c2 = ax2.bar(x2 + width / 2, recover_vals, width, label="조명 급변 복구 프레임 (짧을수록 유리)", color="#10B981")
    ax2.set_title("배경 학습률(Learning Rate) 트레이드오프 실측", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(lr_labels, fontsize=9.5)
    ax2.set_ylabel("소요 프레임 수 (상한 200+)")
    ax2.set_ylim(0, 235)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8.5)
    for bar, orig in zip(c1, [r["stopped_object_absorbed_frames"] for r in lr_rows]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4, f"{orig}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, orig in zip(c2, [r["lighting_recovery_frames"] for r in lr_rows]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4, f"{orig}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    out = DIAGRAMS_DIR / "03_experiment_tracking_and_lr.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_feature_matching_diagram(results: dict) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.8), dpi=180)
    fig.patch.set_facecolor("#F9FAFB")

    orb_rows = results["feature_matching"]["orb"]
    sift_rows = results["feature_matching"]["sift"]
    labels = [r["condition"].replace("실제 장면(box_in_scene)", "실제 장면\n(box_in_scene)") for r in orb_rows]

    orb_rates = [r["match_rate"] for r in orb_rows]
    sift_rates = [r["match_rate"] for r in sift_rows]
    orb_matches = [r["good_matches"] for r in orb_rows]
    sift_matches = [r["good_matches"] for r in sift_rows]

    x = np.arange(len(labels))
    width = 0.36

    b1 = ax1.bar(x - width / 2, orb_rates, width, label="ORB 매칭률 (%)", color="#6366F1")
    b2 = ax1.bar(x + width / 2, sift_rates, width, label="SIFT 매칭률 (%)", color="#EC4899")
    ax1.set_title("조건별 특징점 매칭률 비교 (ORB vs SIFT)", fontsize=11.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("매칭률 (%)")
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.8)

    c1 = ax2.bar(x - width / 2, orb_matches, width, label="ORB 유효 매칭 수", color="#0EA5E9")
    c2 = ax2.bar(x + width / 2, sift_matches, width, label="SIFT 유효 매칭 수", color="#F97316")
    ax2.axhline(15, color="#DC2626", linestyle="--", linewidth=1.4, label="TARGET DETECTED 임계값 (15)")
    ax2.set_title("조건별 유효 매칭 수 (Good Matches)", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("매칭된 특징점 수 (개)")
    ax2.set_ylim(0, max(max(orb_matches), max(sift_matches)) * 1.18)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8.5)
    for bar in list(c1) + list(c2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 8, f"{int(h)}", ha="center", va="bottom", fontsize=8)

    out = DIAGRAMS_DIR / "04_feature_matching_benchmark.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    _configure_korean_font()
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run 'python scripts/run_experiments.py' first."
        )
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    generated = [
        generate_architecture_diagram(),
        generate_id_switch_diagram(),
        generate_tracking_and_lr_diagram(results),
        generate_feature_matching_diagram(results),
    ]
    for path in generated:
        print(f"[OK] Generated diagram: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
