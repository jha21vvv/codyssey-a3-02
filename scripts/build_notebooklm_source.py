"""Consolidate pure production source files (src/*.py) into NOTEBOOKLM_FULL_SOURCE.txt."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
OUT_FILE = ROOT / "NOTEBOOKLM_FULL_SOURCE.txt"

ORDERED_FILES = [
    "__init__.py",
    "config.py",
    "background.py",
    "detector.py",
    "tracker.py",
    "feature_matcher.py",
    "visualizer.py",
    "app.py",
    "main.py",
]


def build() -> Path:
    parts = [
        "================================================================================",
        "CODYSSEY A3-02 — PURE PRODUCTION SOURCE CODE BUNDLE (FOR GOOGLE NOTEBOOKLM)",
        "Includes only src/*.py (with 2-tier comments). Excludes tests and prompt guides.",
        "================================================================================",
        "",
    ]

    included = 0
    for name in ORDERED_FILES:
        file_path = SRC_DIR / name
        if not file_path.exists():
            continue
        content = file_path.read_text(encoding="utf-8")
        parts.append(f"#################### FILE: src/{name} ####################")
        parts.append(content.rstrip())
        parts.append("")
        parts.append("")
        included += 1

    OUT_FILE.write_text("\n".join(parts), encoding="utf-8")
    print(f"[OK] Built {OUT_FILE.name} from {included} production source files ({OUT_FILE.stat().st_size} bytes).")
    return OUT_FILE


if __name__ == "__main__":
    build()
