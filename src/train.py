"""
Trainiert einen 169-Klassen-Artenklassifikator per Transfer Learning.

Rezept: zuerst nur der neue Klassifikations-Head bei eingefrorenem Backbone
(--freeze-epochs), danach vollständiges Fine-Tuning mit kleinerer Lernrate.
Läuft auf CPU (langsam, für Smoke-Tests) und GPU (mit automatic mixed
precision, wenn CUDA verfügbar ist).

Aufruf: python -m src.train --arch mobilenet_v2 --epochs 15 --freeze-epochs 3
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.build_labels import PROCESSED_DIR, SPECIES_CLASSES
from src.dataset import MushroomDataset
from src.model_factory import build_model, set_backbone_trainable
from src.transforms import get_eval_transforms, get_train_transforms

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def _make_loader(csv_path: Path, transform, batch_size: int, num_workers: int, shuffle: bool, limit: int | None) -> DataLoader:
    dataset = MushroomDataset(csv_path, transform=transform)
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def _run_epoch(model, loader, device, criterion, optimizer, scaler, train: bool) -> tuple[float, float]:
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, species_idx, _binary_idx in loader:
            images, species_idx = images.to(device), species_idx.to(device)

            if train:
                optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, species_idx)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == species_idx).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["mobilenet_v2", "resnet50"], required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=5, help="Early-Stopping-Geduld in Epochen")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out", type=Path, default=None, help="Checkpoint-Pfad (Default: models/<arch>.pt)")
    parser.add_argument("--limit", type=int, default=None, help="Nur die ersten N Zeilen je Split nutzen (Smoke-Test)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or MODELS_DIR / f"{args.arch}.pt"

    train_loader = _make_loader(
        args.data_dir / "train.csv", get_train_transforms(args.img_size), args.batch_size, args.num_workers, True, args.limit
    )
    val_loader = _make_loader(
        args.data_dir / "val.csv", get_eval_transforms(args.img_size), args.batch_size, args.num_workers, False, args.limit
    )

    num_classes = len(SPECIES_CLASSES)
    model = build_model(args.arch, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    freezing = args.freeze_epochs > 0
    set_backbone_trainable(model, args.arch, trainable=not freezing)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        if freezing and epoch == args.freeze_epochs:
            print("Backbone wird entfroren, volles Fine-Tuning mit reduzierter Lernrate.")
            set_backbone_trainable(model, args.arch, trainable=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
            freezing = False

        start = time.time()
        train_loss, train_acc = _run_epoch(model, train_loader, device, criterion, optimizer, scaler, train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, device, criterion, optimizer, scaler, train=False)
        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch + 1}/{args.epochs} ({elapsed:.1f}s) "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": args.arch,
                    "num_classes": num_classes,
                    "species_classes": SPECIES_CLASSES,
                    "img_size": args.img_size,
                    "val_acc": val_acc,
                },
                out_path,
            )
            print(f"  -> neues bestes Modell gespeichert unter {out_path} (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early Stopping nach {epoch + 1} Epochen (keine Verbesserung seit {args.patience} Epochen).")
                break

    history_path = RESULTS_DIR / f"{args.arch}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Trainingshistorie gespeichert unter {history_path}")


if __name__ == "__main__":
    main()
