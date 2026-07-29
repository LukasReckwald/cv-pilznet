"""
Ergänzt die Confusion-Matrix-Plots aus evaluate.py um die konkreten Zahlen:
welche Art wird wie oft mit welcher anderen verwechselt, als sortierte CSV.

python src/confusion_pairs.py --checkpoint models/resnet50.pt --top 30
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from build_labels import PROCESSED_DIR
from dataset import MushroomDataset
from model_factory import build_model
from transforms import get_eval_transforms

RESULTS_DIR = Path("results")


@torch.no_grad()
def _predict(model, loader, device):
    y_true, y_pred = [], []
    for images, species_idx, _ in loader:
        outputs = model(images.to(device))
        y_pred.append(outputs.argmax(dim=1).cpu().numpy())
        y_true.append(species_idx.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    species_classes: list[str] = ckpt["species_classes"]
    img_size: int = ckpt["img_size"]
    arch: str = ckpt["arch"]

    dataset = MushroomDataset(args.data_dir / f"{args.split}.csv", transform=get_eval_transforms(img_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Predicting on {len(dataset)} images ...")
    y_true, y_pred = _predict(model, loader, device)

    labels = list(range(len(species_classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    rows = []
    n = cm.shape[0]
    for i in range(n):
        row_sum = cm[i].sum()
        if row_sum == 0:
            continue
        for j in range(n):
            if i == j or cm[i, j] == 0:
                continue
            rows.append(
                {
                    "true_species": species_classes[i],
                    "predicted_species": species_classes[j],
                    "count": int(cm[i, j]),
                    "pct_of_true_class": cm[i, j] / row_sum,
                    "true_class_support": int(row_sum),
                }
            )

    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{arch}_{args.split}_confused_pairs.csv"
    df.to_csv(out_path, index=False)
    print(f"\nAlle Verwechslungspaare gespeichert unter {out_path} ({len(df)} Paare)")

    print(f"\nTop {args.top} Verwechslungen ({arch}, {args.split}):")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
