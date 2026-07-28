"""PyTorch Dataset für die von build_labels.py erzeugten Label-CSVs."""

import csv
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class MushroomDataset(Dataset):
    """Liest data/processed/{train,val,test}.csv, gibt (image, species_idx, binary_idx) zurück."""

    def __init__(self, csv_path, transform=None, limit=None):
        self.csv_path = Path(csv_path)
        self.transform = transform
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        if limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"{self.csv_path} enthält keine Zeilen. Erst 'python -m src.build_labels' ausführen.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = row["image_path"]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Bild konnte nicht gelesen werden: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        species_idx = int(row["species_idx"])
        binary_idx = int(row["binary_idx"])
        return image, species_idx, binary_idx
