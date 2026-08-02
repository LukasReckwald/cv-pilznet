"""Baut je Modell die Loss- und Accuracy-Kurven (Train vs. Val) aus den
results/<arch>_history.json, die train.py am Ende wegschreibt. Kein Argument
nötig, einfach durchlaufen lassen."""

from pathlib import Path

import json
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")

ARCH_LABELS = {"resnet50": "ResNet50", "mobilenet_v2": "MobileNetV2"}
TRAIN_COLOR = "#0072B2"
VAL_COLOR = "#E69F00"


def _plot(history: dict, epochs: list[int], key_prefix: str, ylabel: str, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(epochs, history[f"train_{key_prefix}"], color=TRAIN_COLOR, linewidth=2, marker="o", markersize=5, label="Train")
    ax.plot(epochs, history[f"val_{key_prefix}"], color=VAL_COLOR, linewidth=2, marker="o", markersize=5, label="Val")

    ax.set_title(title)
    ax.set_xlabel("Trainings-Epoche")
    ax.set_ylabel(ylabel)
    ax.set_xticks(epochs)
    ax.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Gespeichert: {out_path}")


def main():
    for arch, label in ARCH_LABELS.items():
        history_path = RESULTS_DIR / f"{arch}_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        epochs = list(range(1, len(history["train_loss"]) + 1))

        _plot(
            history, epochs, "loss", "Loss",
            f"{label} — Loss (Train vs. Val)",
            RESULTS_DIR / f"{arch}_loss_curve.png",
        )
        _plot(
            history, epochs, "acc", "Accuracy",
            f"{label} — Accuracy (Train vs. Val)",
            RESULTS_DIR / f"{arch}_acc_curve.png",
        )


if __name__ == "__main__":
    main()
