"""
Führt die Kaggle-Split-CSVs mit der essbar/giftig-Zuordnung aus
class_mapping.py zusammen -> data/processed/{train,val,test}.csv.

Aufruf: python src/build_labels.py
"""

import csv
from collections import Counter
from pathlib import Path

from class_mapping import (
    BINARY_CLASSES,
    CLASS_TO_LABEL,
    get_label_for_class,
    validate_mapping,
)

RAW_DIR = Path("data/raw")
IMAGE_ROOT = RAW_DIR / "merged_dataset"
PROCESSED_DIR = Path("data/processed")

SPECIES_CLASSES = sorted(CLASS_TO_LABEL.keys())
SPECIES_TO_IDX = {name: i for i, name in enumerate(SPECIES_CLASSES)}
BINARY_TO_IDX = {name: i for i, name in enumerate(BINARY_CLASSES)}


def _resolve_image_path(raw_path):
    """Extrahiert (Art-Ordner, Dateiname) aus dem Kaggle-Pfad."""
    parts = raw_path.replace("\\", "/").split("/")
    return parts[-2], parts[-1]


def _existing_files_index():
    """Indiziert vorhandene Dateien je Art-Ordner (für schnelle Existenzchecks)."""
    index = {}
    for species_dir in IMAGE_ROOT.iterdir():
        if species_dir.is_dir():
            index[species_dir.name] = {p.name for p in species_dir.iterdir()}
    return index


def _process_split(name, existing_files):
    src_csv = RAW_DIR / f"{name}.csv"
    if not src_csv.exists():
        raise FileNotFoundError(
            f"{src_csv} nicht gefunden. Datensatz laut README herunterladen."
        )

    with src_csv.open(newline="", encoding="utf-8") as f:
        rows = [(row["image_path"], row["label"]) for row in csv.DictReader(f)]

    validate_mapping(sorted({species for _, species in rows}))

    out_csv = PROCESSED_DIR / f"{name}.csv"
    species_counter = Counter()
    binary_counter = Counter()
    missing = 0
    duplicates = 0
    seen_paths = set()
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "species", "species_idx", "binary_label", "binary_idx"])
        for raw_path, species in rows:
            species_dir, filename = _resolve_image_path(raw_path)
            if filename not in existing_files.get(species_dir, ()):
                missing += 1
                continue
            image_path = (IMAGE_ROOT / species_dir / filename).as_posix()
            if image_path in seen_paths:
                duplicates += 1
                continue
            seen_paths.add(image_path)
            binary_label = get_label_for_class(species)
            writer.writerow(
                [image_path, species, SPECIES_TO_IDX[species], binary_label, BINARY_TO_IDX[binary_label]]
            )
            species_counter[species] += 1
            binary_counter[binary_label] += 1

    if missing:
        print(f"  Warnung: {missing} Zeilen aus {src_csv.name} verweisen auf fehlende Bilddateien (übersprungen).")
    if duplicates:
        print(f"  {duplicates} doppelte Zeilen (gleiches Bild, Oversampling im Original) entfernt.")
    print(f"{name}: {sum(species_counter.values())} Zeilen -> {out_csv}")
    print(f"  binaer: {dict(binary_counter)}")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Klassen im Mapping: {len(SPECIES_CLASSES)} {dict(Counter(CLASS_TO_LABEL.values()))}")

    existing_files = _existing_files_index()
    for split in ("train", "val", "test"):
        _process_split(split, existing_files)


if __name__ == "__main__":
    main()
