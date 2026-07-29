"""
Lässt ein trainiertes Modell auf einem Bilderordner laufen (Top-K-Artvorhersagen
+ essbar/giftig-Einschaetzung). Je nachdem ob true_species aus --labels-csv im
Trainingsset vorkommt, wertet das Skript als Domain-Shift (Top-1/Top-5-Accuracy)
oder Out-of-Domain (nur Gattungstreffer) aus.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from class_mapping import BINARY_CLASSES, CLASS_TO_LABEL
from model_factory import build_model
from transforms import get_eval_transforms

RESULTS_DIR = Path("results")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def _load_image(path: Path, img_size: int) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Bild konnte nicht gelesen werden: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = get_eval_transforms(img_size)(image=image)["image"]
    return transformed


def _genus(species_name: str) -> str:
    return species_name.split(" ")[0]


@torch.no_grad()
def _predict_image(model, tensor: torch.Tensor, device, species_classes, species_to_binary_idx, top_k: int):
    logits = model(tensor.unsqueeze(0).to(device))
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = np.argsort(probs)[::-1][:top_k]
    top_species = [(species_classes[i], float(probs[i])) for i in top_idx]

    binary_probs = np.zeros(len(BINARY_CLASSES))
    for i, p in enumerate(probs):
        binary_probs[species_to_binary_idx[i]] += p

    return top_species, binary_probs


def _load_labels(labels_csv: Path) -> dict:
    df = pd.read_csv(labels_csv, dtype=str).fillna("")
    return {row["filename"]: row for _, row in df.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, ckpt = _load_checkpoint(args.checkpoint, device)
    species_classes: list[str] = ckpt["species_classes"]
    img_size: int = ckpt["img_size"]
    arch: str = ckpt["arch"]
    species_set = set(species_classes)

    binary_to_idx = {name: i for i, name in enumerate(BINARY_CLASSES)}
    species_to_binary_idx = [binary_to_idx[CLASS_TO_LABEL[s]] for s in species_classes]

    labels = _load_labels(args.labels_csv) if args.labels_csv else {}

    image_paths = sorted(
        p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"Keine Bilder in {args.images_dir} gefunden.")
    print(f"{len(image_paths)} Bilder in {args.images_dir} gefunden.\n")

    rows = []
    for path in image_paths:
        tensor = _load_image(path, img_size)
        top_species, binary_probs = _predict_image(
            model, tensor, device, species_classes, species_to_binary_idx, args.top_k
        )
        top_names = [name for name, _ in top_species]
        predicted_binary = BINARY_CLASSES[int(np.argmax(binary_probs))]

        label_row = labels.get(path.name)
        true_species = label_row["true_species"] if label_row is not None and label_row["true_species"] else None
        true_binary = label_row["true_binary"] if label_row is not None and label_row["true_binary"] else None

        if true_species is None:
            domain = "unlabeled"
        elif true_species in species_set:
            domain = "domain_shift"
        else:
            domain = "ood"

        row = {
            "filename": path.name,
            "domain": domain,
            "true_species": true_species or "",
            "true_binary": true_binary or "",
        }
        for rank, (name, prob) in enumerate(top_species, start=1):
            row[f"top{rank}_species"] = name
            row[f"top{rank}_confidence"] = prob
        for label, idx in binary_to_idx.items():
            row[f"prob_{label}"] = binary_probs[idx]
        row["predicted_binary"] = predicted_binary

        if domain == "domain_shift":
            row["top1_correct"] = true_species == top_names[0]
            row["top5_correct"] = true_species in top_names
        elif domain == "ood":
            row["true_genus_in_topk"] = _genus(true_species) in {_genus(n) for n in top_names}

        if true_binary:
            row["binary_correct"] = predicted_binary == true_binary

        rows.append(row)

        tag = {"domain_shift": "[Domain-Shift]", "ood": "[OOD]", "unlabeled": ""}[domain]
        print(f"{path.name} {tag}")
        if true_species:
            print(f"  Wahre Art: {true_species}" + (f" (wahr: {true_binary})" if true_binary else ""))
        for rank, (name, prob) in enumerate(top_species, start=1):
            label = CLASS_TO_LABEL[name]
            marker = " <-- richtig" if true_species == name else ""
            print(f"  {rank}. {name:<30s} {prob:6.1%}  ({label}){marker}")
        print(
            f"  -> essbar/giftig (aggregiert): "
            f"edible={binary_probs[binary_to_idx['edible']]:.1%}  "
            f"poisonous={binary_probs[binary_to_idx['poisonous']]:.1%}\n"
        )

    df = pd.DataFrame(rows)
    out_path = args.output or RESULTS_DIR / f"{arch}_folder_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Ergebnisse gespeichert unter {out_path}\n")

    if not labels:
        return

    print(f"== Zusammenfassung ({arch}) ==")
    ds = df[df["domain"] == "domain_shift"]
    if len(ds):
        print(
            f"Domain-Shift-Test ({len(ds)} Bilder): "
            f"Top-1-Accuracy={ds['top1_correct'].mean():.1%}  "
            f"Top-5-Accuracy={ds['top5_correct'].mean():.1%}"
        )
        if "binary_correct" in ds and ds["binary_correct"].notna().any():
            print(f"  Essbar/giftig-Accuracy: {ds['binary_correct'].mean():.1%}")

    ood = df[df["domain"] == "ood"]
    if len(ood):
        print(
            f"Out-of-Domain-Test ({len(ood)} Bilder): "
            f"richtige Gattung in Top-{args.top_k}={ood['true_genus_in_topk'].mean():.1%}"
        )
        if "binary_correct" in ood and ood["binary_correct"].notna().any():
            print(f"  Essbar/giftig-Accuracy: {ood['binary_correct'].mean():.1%}")


if __name__ == "__main__":
    main()
