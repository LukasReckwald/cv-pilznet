"""
Führt die Kaggle-Split-CSVs (data/raw/{train,val,test}.csv, Spalten
image_path,label) mit der essbar/giftig-Zuordnung aus class_mapping.py
zusammen und schreibt einheitliche, lokal nutzbare Label-Dateien nach
data/processed/{train,val,test}.csv.

Aufruf: python -m src.build_labels
"""

import argparse
import csv
from collections import Counter
from pathlib import Path, PurePosixPath

from src.class_mapping import (
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


def _resolve_image_path(raw_path: str) -> tuple[str, str]:
    """Kaggle-Pfad (/kaggle/working/merged_dataset/<Art>/<Datei>) auf
    (Art-Ordnername, Dateiname) reduzieren, unabhängig vom lokalen OS."""
    posix_path = PurePosixPath(raw_path.replace("\\", "/"))
    return posix_path.parts[-2], posix_path.parts[-1]


def _existing_files_index() -> dict[str, set[str]]:
    """Einmaliger Verzeichnis-Scan statt eines stat()-Aufrufs pro CSV-Zeile."""
    index: dict[str, set[str]] = {}
    for species_dir in IMAGE_ROOT.iterdir():
        if species_dir.is_dir():
            index[species_dir.name] = {p.name for p in species_dir.iterdir()}
    return index


def _process_split(name: str, existing_files: dict[str, set[str]]) -> None:
    src_csv = RAW_DIR / f"{name}.csv"
    if not src_csv.exists():
        raise FileNotFoundError(
            f"{src_csv} nicht gefunden. Datensatz laut README herunterladen."
        )

    with src_csv.open(newline="", encoding="utf-8") as f:
        rows = [(row["image_path"], row["label"]) for row in csv.DictReader(f)]

    validate_mapping(sorted({species for _, species in rows}))

    out_csv = PROCESSED_DIR / f"{name}.csv"
    species_counter: Counter = Counter()
    binary_counter: Counter = Counter()
    missing = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "species", "species_idx", "binary_label", "binary_idx"])
        for raw_path, species in rows:
            species_dir, filename = _resolve_image_path(raw_path)
            if filename not in existing_files.get(species_dir, ()):
                missing += 1
                continue
            binary_label = get_label_for_class(species)
            image_path = (IMAGE_ROOT / species_dir / filename).as_posix()
            writer.writerow(
                [image_path, species, SPECIES_TO_IDX[species], binary_label, BINARY_TO_IDX[binary_label]]
            )
            species_counter[species] += 1
            binary_counter[binary_label] += 1

    if missing:
        print(f"  Warnung: {missing} Zeilen aus {src_csv.name} verweisen auf fehlende Bilddateien (übersprungen).")
    print(f"{name}: {sum(species_counter.values())} Zeilen -> {out_csv}")
    print(f"  binaer: {dict(binary_counter)}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Klassen im Mapping: {len(SPECIES_CLASSES)} {dict(Counter(CLASS_TO_LABEL.values()))}")

    existing_files = _existing_files_index()
    for split in ("train", "val", "test"):
        _process_split(split, existing_files)


if __name__ == "__main__":
    main()
