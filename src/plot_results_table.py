"""python src/plot_results_table.py baut die Ergebnistabelle (essbar/giftig- und
Art-Accuracy, Macro-F1) für den Ergebnis-Abschnitt"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

RESULTS_DIR = Path("results")
OUT_PATH = RESULTS_DIR / "results_table.png"

ACCENT = "#0072B2"
ROW_FILL = "#ffffff"
HEADER_TEXT = "#ffffff"
TEXT_COLOR = "#1a1a1a"

MODELS = [("ResNet50", "resnet50"), ("MobileNetV2", "mobilenet_v2")]

FIG_W = 7.3
COL0_W = 2.0
COL_W = (FIG_W - COL0_W) / 3
HEADER_H = 0.95
ROW_H = 0.85
HEADER_FS = 13.5
CELL_FS = 15


def _fmt(fraction: float) -> str:
    return f"{fraction * 100:.1f} %".replace(".", ",")


def load_rows() -> tuple[list[str], list[tuple[str, str, str, str]]]:
    rows = []
    n_species = None
    for label, arch in MODELS:
        binary_df = pd.read_csv(RESULTS_DIR / f"{arch}_test_report_binary.csv", index_col=0)
        species_df = pd.read_csv(RESULTS_DIR / f"{arch}_test_report_species.csv", index_col=0)
        n_species = len(species_df) - 3  # abzgl. accuracy/macro avg/weighted avg

        binary_acc = binary_df.loc["accuracy", "f1-score"]
        species_acc = species_df.loc["accuracy", "f1-score"]
        macro_f1 = species_df.loc["macro avg", "f1-score"]
        rows.append((label, _fmt(binary_acc), _fmt(species_acc), _fmt(macro_f1)))

    headers = ["Modell", "Accuracy\nessbar/giftig", f"Accuracy\nArt ({n_species} Kl.)", "Macro-F1\nArt"]
    return headers, rows


def col_x(i: int) -> float:
    return COL0_W + (i - 1) * COL_W if i > 0 else 0.0


def col_w(i: int) -> float:
    return COL0_W if i == 0 else COL_W


def main() -> None:
    headers, rows = load_rows()

    n_rows = len(rows)
    total_h = HEADER_H + n_rows * ROW_H
    fig, ax = plt.subplots(figsize=(FIG_W, total_h))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # Header
    header_y = total_h - HEADER_H
    ax.add_patch(FancyBboxPatch(
        (0, header_y), FIG_W, HEADER_H,
        boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=0, facecolor=ACCENT, zorder=2,
    ))
    for i, label in enumerate(headers):
        cx = col_x(i) + col_w(i) / 2
        ax.text(cx, header_y + HEADER_H / 2, label, ha="center", va="center",
                fontsize=HEADER_FS, weight="bold", color=HEADER_TEXT, linespacing=1.25, zorder=3)

    # Datenzeilen
    row_y = header_y
    for model, binary_acc, species_acc, f1 in rows:
        row_y -= ROW_H
        ax.add_patch(plt.Rectangle((0, row_y), FIG_W, ROW_H, facecolor=ROW_FILL, edgecolor="none", zorder=1))
        ax.plot([0, FIG_W], [row_y, row_y], color="#d8dce0", linewidth=1, zorder=2)

        values = [model, binary_acc, species_acc, f1]
        for i, val in enumerate(values):
            cx = col_x(i) + col_w(i) / 2
            ha = "left" if i == 0 else "center"
            x = col_x(i) + 0.25 if i == 0 else cx
            ax.text(x, row_y + ROW_H / 2, val, ha=ha, va="center",
                    fontsize=CELL_FS, weight="normal", color=TEXT_COLOR, zorder=3)

    # Aeussere Umrandung
    ax.add_patch(FancyBboxPatch(
        (0, row_y), FIG_W, header_y + HEADER_H - row_y,
        boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=1.6, edgecolor="#c7ccd1", facecolor="none", zorder=4,
    ))
    # Vertikale Spaltenlinien
    for i in range(1, 4):
        x = col_x(i)
        ax.plot([x, x], [row_y, header_y], color="#d8dce0", linewidth=1, zorder=2)

    fig.tight_layout(pad=0.1)
    fig.savefig(OUT_PATH, dpi=220)
    plt.close(fig)
    print(f"Gespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
