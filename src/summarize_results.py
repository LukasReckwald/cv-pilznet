"""
Fasst die Kennzahlen aus der README (Ergebnistabelle, Trainingsverlauf,
Domain-Shift- und OOD-Test) für beide Modelle in einer CSV zusammen, plus die
Top-5-Verwechslungspaare je Modell in einer zweiten CSV.

python src/summarize_results.py
"""

import json
from pathlib import Path

import pandas as pd

from class_mapping import CLASS_TO_LABEL

RESULTS_DIR = Path("results")
MODELS = [("ResNet50", "resnet50"), ("MobileNetV2", "mobilenet_v2")]


def _genus(species_name: str) -> str:
    return species_name.split(" ")[0]


def _metrics_row(label: str, arch: str) -> dict:
    binary_df = pd.read_csv(RESULTS_DIR / f"{arch}_test_report_binary.csv", index_col=0)
    species_df = pd.read_csv(RESULTS_DIR / f"{arch}_test_report_species.csv", index_col=0)

    history = json.loads((RESULTS_DIR / f"{arch}_history.json").read_text(encoding="utf-8"))

    ds_df = pd.read_csv(RESULTS_DIR / f"{arch}_domain_shift_predictions.csv")
    ood_df = pd.read_csv(RESULTS_DIR / f"{arch}_ood_predictions.csv")
    ood_genus_top1 = (ood_df["true_species"].apply(_genus) == ood_df["top1_species"].apply(_genus)).mean()

    return {
        "model": label,
        "acc_binary": binary_df.loc["accuracy", "f1-score"],
        "acc_species": species_df.loc["accuracy", "f1-score"],
        "macro_f1_species": species_df.loc["macro avg", "f1-score"],
        "train_acc_epoch8": history["train_acc"][-1],
        "val_acc_epoch8": history["val_acc"][-1],
        "domain_shift_top1": ds_df["top1_correct"].mean(),
        "domain_shift_top5": ds_df["top5_correct"].mean(),
        "domain_shift_binary": ds_df["binary_correct"].mean(),
        "ood_genus_top1": ood_genus_top1,
        "ood_genus_top5": ood_df["true_genus_in_topk"].mean(),
        "ood_binary": ood_df["binary_correct"].mean(),
    }


def _top_confusions(label: str, arch: str, top: int = 5) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / f"{arch}_test_confused_pairs.csv")
    df = df.sort_values("pct_of_true_class", ascending=False).head(top).copy()
    df.insert(0, "rank", range(1, len(df) + 1))
    df.insert(0, "model", label)
    return df


def _boundary_confusions(label: str, arch: str) -> pd.DataFrame:
    """Verwechslungspaare, bei denen essbar/giftig-Label von wahrer und
    vorhergesagter Art auseinanderfallen (gefährlichste Fehlerklasse)."""
    df = pd.read_csv(RESULTS_DIR / f"{arch}_test_confused_pairs.csv")
    df["true_label"] = df["true_species"].map(CLASS_TO_LABEL)
    df["predicted_label"] = df["predicted_species"].map(CLASS_TO_LABEL)
    df = df[df["true_label"] != df["predicted_label"]]
    df = df.sort_values("pct_of_true_class", ascending=False).copy()
    df.insert(0, "model", label)
    return df[[
        "model", "true_species", "true_label", "predicted_species", "predicted_label",
        "count", "pct_of_true_class", "true_class_support",
    ]]


def main() -> None:
    metrics_rows = [_metrics_row(label, arch) for label, arch in MODELS]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = RESULTS_DIR / "summary_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Gespeichert: {metrics_path}")

    confusions_df = pd.concat([_top_confusions(label, arch) for label, arch in MODELS], ignore_index=True)
    confusions_path = RESULTS_DIR / "summary_top_confusions.csv"
    confusions_df.to_csv(confusions_path, index=False)
    print(f"Gespeichert: {confusions_path}")

    boundary_df = pd.concat([_boundary_confusions(label, arch) for label, arch in MODELS], ignore_index=True)
    boundary_path = RESULTS_DIR / "summary_boundary_confusions.csv"
    boundary_df.to_csv(boundary_path, index=False)
    print(f"Gespeichert: {boundary_path} ({len(boundary_df)} grenzüberschreitende Paare)")


if __name__ == "__main__":
    main()
