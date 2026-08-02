"""Tabelle für die Domain-Shift-Ergebnisse (Top-1-/Top-5-Accuracy,essbar/giftig)"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

RESULTS_DIR = Path("results")
OUT_PATH = RESULTS_DIR / "domain_shift_table.png"

ACCENT = "#0072B2"
ROW_FILL = "#ffffff"
HEADER_TEXT = "#ffffff"
TEXT_COLOR = "#1a1a1a"

FIG_W = 7.3
COL0_W = 2.3
HEADER_H = 0.95
ROW_H = 0.75
HEADER_FS = 13
CELL_FS = 14.5

HEADERS = ["Modell", "Top-1-\nAccuracy", "Top-5-\nAccuracy", "essbar/\ngiftig"]
MODELS = [("ResNet50", "resnet50"), ("MobileNetV2", "mobilenet_v2")]


def _fmt(fraction: float) -> str:
    return f"{fraction * 100:.1f} %".replace(".", ",")


def load_rows() -> list[tuple[str, str, str, str]]:
    rows = []
    for label, arch in MODELS:
        df = pd.read_csv(RESULTS_DIR / f"{arch}_domain_shift_predictions.csv")
        top1_acc = df["top1_correct"].mean()
        top5_acc = df["top5_correct"].mean()
        binary_acc = df["binary_correct"].mean()
        rows.append((label, _fmt(top1_acc), _fmt(top5_acc), _fmt(binary_acc)))
    return rows


def col_geometry(n_cols: int):
    col_w = (FIG_W - COL0_W) / (n_cols - 1)

    def col_x(i: int) -> float:
        return COL0_W + (i - 1) * col_w if i > 0 else 0.0

    def col_w_(i: int) -> float:
        return COL0_W if i == 0 else col_w

    return col_x, col_w_


def main() -> None:
    rows = load_rows()

    total_h = HEADER_H + len(rows) * ROW_H
    fig, ax = plt.subplots(figsize=(FIG_W, total_h))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    col_x, col_w = col_geometry(len(HEADERS))

    header_y = total_h - HEADER_H
    ax.add_patch(FancyBboxPatch(
        (0, header_y), FIG_W, HEADER_H,
        boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=0, facecolor=ACCENT, zorder=2,
    ))
    for i, label in enumerate(HEADERS):
        cx = col_x(i) + col_w(i) / 2
        ax.text(cx, header_y + HEADER_H / 2, label, ha="center", va="center",
                fontsize=HEADER_FS, weight="bold", color=HEADER_TEXT, linespacing=1.2, zorder=3)

    row_y = header_y
    for row in rows:
        row_y -= ROW_H
        ax.add_patch(plt.Rectangle((0, row_y), FIG_W, ROW_H, facecolor=ROW_FILL, edgecolor="none", zorder=1))
        ax.plot([0, FIG_W], [row_y, row_y], color="#d8dce0", linewidth=1, zorder=2)
        for i, val in enumerate(row):
            cx = col_x(i) + col_w(i) / 2
            ha = "left" if i == 0 else "center"
            x = col_x(i) + 0.25 if i == 0 else cx
            ax.text(x, row_y + ROW_H / 2, val, ha=ha, va="center",
                    fontsize=CELL_FS, color=TEXT_COLOR, zorder=3)

    ax.add_patch(FancyBboxPatch(
        (0, row_y), FIG_W, header_y + HEADER_H - row_y,
        boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=1.6, edgecolor="#c7ccd1", facecolor="none", zorder=4,
    ))
    for i in range(1, len(HEADERS)):
        x = col_x(i)
        ax.plot([x, x], [row_y, header_y], color="#d8dce0", linewidth=1, zorder=2)

    fig.tight_layout(pad=0.08)
    fig.savefig(OUT_PATH, dpi=220)
    plt.close(fig)
    print(f"Gespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
