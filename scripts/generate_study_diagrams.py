"""Generate 2 beginner-friendly visual theory diagrams for the junior study guide."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DIAGRAMS_DIR = ROOT / "diagrams"


def _configure_korean_font() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def generate_morphology_and_contour_diagram() -> Path:
    """Visualize the 4-step transformation from raw MOG2 mask to bounding box + centroid."""
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.1), dpi=180)
    fig.patch.set_facecolor("#F9FAFB")
    fig.suptitle(
        "이론 도해 ① : 배경 차분(MOG2) → 모폴로지(열림·닫힘) → 윤곽선·중심점 추출 4단계",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        y=1.02,
    )

    h, w = 60, 60
    # Step 1: Raw MOG2 output (0=bg, 127=shadow, 255=person + salt noise + internal hole)
    raw = np.zeros((h, w), dtype=np.uint8)
    # Person silhouette
    raw[14:46, 22:38] = 255
    # Internal hole (clothes similar to background)
    raw[24:31, 27:33] = 0
    # Shadow (value 127)
    raw[42:52, 26:46] = 127
    # Salt noise dots
    for r, c in [(8, 10), (12, 48), (50, 12), (20, 8), (36, 52), (5, 32)]:
        raw[r : r + 2, c : c + 2] = 255

    # Step 2: Threshold(>200) + MORPH_OPEN (shadow & salt noise removed, hole still remains)
    opened = np.zeros((h, w), dtype=np.uint8)
    opened[14:46, 22:38] = 255
    opened[24:31, 27:33] = 0

    # Step 3: MORPH_CLOSE (internal hole filled into one solid blob)
    closed = np.zeros((h, w), dtype=np.uint8)
    closed[14:46, 22:38] = 255

    # Step 4: Contour + Bounding Box + Centroid
    titles = [
        "1) MOG2 원본 마스크\n흰색(255)+그림자(127)+먼지",
        "2) 그림자 제거 + 열림(OPEN)\n그림자·먼지 삭제 (구멍은 남음)",
        "3) 닫힘(CLOSE) 마감\n몸통 안쪽 구멍 메움 → 1덩어리",
        "4) 윤곽선·박스·중심점\nboundingRect + centroid",
    ]
    captions = [
        "문제: 그림자와 작은 점들이\n가짜 객체로 잡힐 수 있음",
        "threshold(200)로 그림자 제거\n빗자루(OPEN)로 먼지 쓸어냄",
        "퍼티(CLOSE)로 구멍을 메워\n사람이 2조각 나는 것 방지",
        "면적 >= 500px 합격!\n초록 박스 & 빨간 중심점 추출",
    ]

    imgs = [raw, opened, closed, closed]
    for idx, (ax, img, title, cap) in enumerate(zip(axes, imgs, titles, captions)):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9.8, fontweight="bold", pad=8)
        ax.set_xlabel(cap, fontsize=8.5, color="#374151", labelpad=8)
        ax.set_xticks([])
        ax.set_yticks([])

        if idx == 0:
            ax.annotate("그림자(127)", xy=(38, 47), xytext=(36, 56), color="#F59E0B",
                        fontsize=8, fontweight="bold", ha="center",
                        arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=1.2))
            ax.annotate("먼지", xy=(11, 9), xytext=(15, 22), color="#EF4444",
                        fontsize=8, fontweight="bold", ha="center",
                        arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1.2))
        elif idx == 1:
            ax.annotate("몸통 구멍", xy=(30, 27), xytext=(48, 20), color="#EF4444",
                        fontsize=8, fontweight="bold", ha="center",
                        arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1.2))
        elif idx == 3:
            rect = Rectangle((21.5, 13.5), 16, 32, linewidth=2.4, edgecolor="#22C55E", facecolor="none")
            ax.add_patch(rect)
            ax.scatter([29.5], [29.5], s=90, color="#EF4444", zorder=6)
            ax.text(29.5, 10.5, "ID 1 (bbox)", color="#22C55E", fontsize=8.5, fontweight="bold", ha="center")
            ax.text(46, 31, "중심점\n(cx, cy)", color="#EF4444", fontsize=8, fontweight="bold", ha="center")

    out = DIAGRAMS_DIR / "05_morphology_and_contour_theory.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_euclidean_and_lowe_diagram() -> Path:
    """Visualize Euclidean distance matching (left) and Lowe's Ratio Test (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0), dpi=180)
    fig.patch.set_facecolor("#F9FAFB")
    fig.suptitle(
        "이론 도해 ② : 유클리드 거리 매칭(Tracker) & Lowe's Ratio Test(ORB/SIFT)",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        y=1.01,
    )

    # Left: Euclidean Distance + max_distance + Tie-breaking
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("A. 유클리드 거리 매칭 (math.hypot) & 최대 보폭 제한", fontsize=10.8, fontweight="bold", pad=8)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.set_xlabel("X 좌표 (픽셀)")
    ax1.set_ylabel("Y 좌표 (픽셀)")

    # Old ID 1 at (30, 45) with max_distance circle radius=28
    old_pt = (30, 45)
    new_close = (48, 57)  # dist = hypot(18, 12) = 21.6 <= 28 -> Match!
    new_far = (82, 72)    # dist = hypot(52, 27) = 58.6 > 28 -> New ID 2!

    circle = Circle(old_pt, 28, edgecolor="#2563EB", facecolor="#DBEAFE", alpha=0.35, linestyle="--", lw=1.8)
    ax1.add_patch(circle)
    ax1.scatter([old_pt[0]], [old_pt[1]], s=220, color="#2563EB", zorder=5)
    ax1.scatter([new_close[0]], [new_close[1]], s=200, color="#16A34A", marker="s", zorder=5)
    ax1.scatter([new_far[0]], [new_far[1]], s=200, color="#DC2626", marker="^", zorder=5)

    # Pythagoras right triangle for close match
    ax1.plot([old_pt[0], new_close[0]], [old_pt[1], old_pt[1]], color="#6B7280", linestyle=":", lw=1.6)
    ax1.plot([new_close[0], new_close[0]], [old_pt[1], new_close[1]], color="#6B7280", linestyle=":", lw=1.6)
    ax1.plot([old_pt[0], new_close[0]], [old_pt[1], new_close[1]], color="#16A34A", lw=2.4)
    ax1.plot([old_pt[0], new_far[0]], [old_pt[1], new_far[1]], color="#DC2626", linestyle="--", lw=1.8)

    ax1.text(30, 35, "이전 ID 1\n(30, 45)", ha="center", color="#1D4ED8", fontweight="bold", fontsize=9)
    ax1.text(39, 41, "Δx=18", ha="center", color="#4B5563", fontsize=8)
    ax1.text(53, 50, "Δy=12", ha="left", color="#4B5563", fontsize=8)
    ax1.text(50, 64, "새 감지 A (48, 57)\n거리 21.6px ≤ max_distance\n→ ID 1 유지!", ha="center", color="#15803D", fontweight="bold", fontsize=8.5)
    ax1.text(78, 82, "새 감지 B (82, 72)\n거리 58.6px > max_distance\n→ 신규 ID 2 발급!", ha="center", color="#B91C1C", fontweight="bold", fontsize=8.5)
    ax1.text(28, 12, "점선 원 = 허용 최대 보폭(max_distance)\n동률 거리 시 (거리, ID, 인덱스) 순으로 결정론적 선택", ha="left", fontsize=8.2, color="#374151",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F3F4F6", edgecolor="#D1D5DB"))

    # Right: Lowe's Ratio Test (Pass vs Reject)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("B. Lowe's Ratio Test (d₁ < 0.75 × d₂) — 진짜 지문만 합격!", fontsize=10.8, fontweight="bold", pad=8)
    ax2.axis("off")

    # Pass box (top)
    pass_box = FancyBboxPatch((4, 54), 92, 38, boxstyle="round,pad=0.02,rounding_size=3",
                              facecolor="#ECFDF5", edgecolor="#10B981", linewidth=2.0)
    ax2.add_patch(pass_box)
    ax2.text(50, 86, "[합격 사례] 진짜 일치: 1등이 압도적으로 가까움", ha="center", fontsize=10, fontweight="bold", color="#065F46")
    ax2.barh([74, 63], [24, 72], left=20, height=6.5, color=["#10B981", "#9CA3AF"])
    ax2.text(18, 74, "1등(d₁)", ha="right", va="center", fontsize=9, fontweight="bold", color="#065F46")
    ax2.text(18, 63, "2등(d₂)", ha="right", va="center", fontsize=9, fontweight="bold", color="#4B5563")
    ax2.text(46, 74, "거리 30", va="center", fontsize=9, fontweight="bold", color="#065F46")
    ax2.text(93, 63, "거리 90", va="center", ha="right", fontsize=9, color="#374151")
    ax2.text(50, 56.5, "판정: 30 < 0.75 × 90 (67.5) → 확실한 지문! good_matches += 1", ha="center", fontsize=8.8, fontweight="bold", color="#047857")

    # Reject box (bottom)
    rej_box = FancyBboxPatch((4, 8), 92, 38, boxstyle="round,pad=0.02,rounding_size=3",
                             facecolor="#FEF2F2", edgecolor="#EF4444", linewidth=2.0)
    ax2.add_patch(rej_box)
    ax2.text(50, 40, "[탈락 사례] 애매한 오매칭: 1등과 2등이 비슷비슷함", ha="center", fontsize=10, fontweight="bold", color="#991B1B")
    ax2.barh([28, 17], [56, 62], left=20, height=6.5, color=["#EF4444", "#9CA3AF"])
    ax2.text(18, 28, "1등(d₁)", ha="right", va="center", fontsize=9, fontweight="bold", color="#991B1B")
    ax2.text(18, 17, "2등(d₂)", ha="right", va="center", fontsize=9, fontweight="bold", color="#4B5563")
    ax2.text(78, 28, "거리 70", va="center", fontsize=9, fontweight="bold", color="#991B1B")
    ax2.text(84, 17, "거리 78", va="center", fontsize=9, color="#374151")
    ax2.text(50, 10.5, "판정: 70 < 0.75 × 78 (58.5) 거짓 → 배경 무늬 착각! 과감히 탈락", ha="center", fontsize=8.8, fontweight="bold", color="#B91C1C")

    out = DIAGRAMS_DIR / "06_euclidean_and_lowe_ratio_theory.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    _configure_korean_font()
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (generate_morphology_and_contour_diagram(), generate_euclidean_and_lowe_diagram()):
        print(f"[OK] Generated study diagram: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
