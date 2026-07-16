"""
Wertet einen trainierten Artenklassifikator aus: liefert sowohl die
169-Klassen-Confusion-Matrix (Artenebene) als auch eine daraus abgeleitete
essbar/giftig-Confusion-Matrix (binär, über class_mapping.CLASS_TO_LABEL).

Aufruf: python src/evaluate.py --checkpoint models/mobilenet_v2.pt --split test
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from build_labels import BINARY_TO_IDX, PROCESSED_DIR
from class_mapping import BINARY_CLASSES, CLASS_TO_LABEL
from dataset import MushroomDataset
from model_factory import build_model
from transforms import get_eval_transforms

RESULTS_DIR = Path("results")


def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_species, y_pred_species, y_true_binary = [], [], []
    for images, species_idx, binary_idx in loader:
        outputs = model(images.to(device))
        y_pred_species.append(outputs.argmax(dim=1).cpu().numpy())
        y_true_species.append(species_idx.numpy())
        y_true_binary.append(binary_idx.numpy())
    return (
        np.concatenate(y_true_species),
        np.concatenate(y_pred_species),
        np.concatenate(y_true_binary),
    )


def _plot_species_confusion(cm: np.ndarray, species_classes: list[str], out_path: Path) -> None:
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(22, 20))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(species_classes)))
    ax.set_yticks(range(len(species_classes)))
    ax.set_xticklabels(species_classes, rotation=90, fontsize=4)
    ax.set_yticklabels(species_classes, fontsize=4)
    ax.set_xlabel("Vorhergesagte Art")
    ax.set_ylabel("Tatsächliche Art")
    ax.set_title("Confusion Matrix – 169 Arten (zeilennormalisiert)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="Anteil an tatsächlicher Klasse")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_binary_confusion(cm: np.ndarray, out_path: Path) -> None:
    cm_pct = cm.astype(float) / cm.sum()

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(BINARY_CLASSES)))
    ax.set_yticks(range(len(BINARY_CLASSES)))
    ax.set_xticklabels(BINARY_CLASSES)
    ax.set_yticklabels(BINARY_CLASSES)
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Tatsächlich")
    ax.set_title("Confusion Matrix – essbar / giftig")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1%})",
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, ckpt = _load_checkpoint(args.checkpoint, device)
    species_classes: list[str] = ckpt["species_classes"]
    img_size: int = ckpt["img_size"]
    arch: str = ckpt["arch"]

    dataset = MushroomDataset(args.data_dir / f"{args.split}.csv", transform=get_eval_transforms(img_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    y_true_species, y_pred_species, y_true_binary = _predict(model, loader, device)

    # Vorhergesagte Art -> essbar/giftig über die zentrale Zuordnung ableiten (kein zweites Modell nötig)
    species_to_binary_idx = np.array([BINARY_TO_IDX[CLASS_TO_LABEL[s]] for s in species_classes])
    y_pred_binary = species_to_binary_idx[y_pred_species]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{arch}_{args.split}"

    species_labels = list(range(len(species_classes)))
    cm_species = confusion_matrix(y_true_species, y_pred_species, labels=species_labels)
    _plot_species_confusion(cm_species, species_classes, RESULTS_DIR / f"{prefix}_confusion_species.png")
    species_report = pd.DataFrame(
        classification_report(
            y_true_species, y_pred_species, labels=species_labels, target_names=species_classes,
            output_dict=True, zero_division=0,
        )
    ).transpose()
    species_report.to_csv(RESULTS_DIR / f"{prefix}_report_species.csv")

    binary_labels = list(range(len(BINARY_CLASSES)))
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary, labels=binary_labels)
    _plot_binary_confusion(cm_binary, RESULTS_DIR / f"{prefix}_confusion_binary.png")
    binary_report = pd.DataFrame(
        classification_report(
            y_true_binary, y_pred_binary, labels=binary_labels, target_names=BINARY_CLASSES,
            output_dict=True, zero_division=0,
        )
    ).transpose()
    binary_report.to_csv(RESULTS_DIR / f"{prefix}_report_binary.csv")

    print(f"\n== {arch} / {args.split} ==")
    print(f"Arten-Ebene  : Accuracy={accuracy_score(y_true_species, y_pred_species):.4f}  "
          f"Macro-F1={f1_score(y_true_species, y_pred_species, average='macro', zero_division=0):.4f}")
    print(f"Binaer-Ebene : Accuracy={accuracy_score(y_true_binary, y_pred_binary):.4f}  "
          f"Macro-F1={f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0):.4f}")
    print(f"Plots/Reports gespeichert unter {RESULTS_DIR}/{prefix}_*")


if __name__ == "__main__":
    main()
