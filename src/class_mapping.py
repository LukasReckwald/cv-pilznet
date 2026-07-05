"""
Hier wird jede Pilzart-Klasse aus dem Kaggle-Datensatz (Ordnername)
einer von zwei Kategorien zugeordnet: "edible" (essbar) oder "poisonous"
(nicht essbar / giftig).
"""

from typing import Literal

CLASS_TO_LABEL: dict[str, Literal["edible", "poisonous"]] = {
    "Agaricus augustus": "edible",
    "Agaricus xanthodermus": "edible",
    "Amanita amerirubescens": "poisonous",
    "Amanita augusta": "poisonous",
    # weitere Klassen hier ergänzen ...
}

# Beide Zielklassen für die binäre Klassifikation (fix, nicht ändern)
BINARY_CLASSES = ["edible", "poisonous"]


def get_label_for_class(class_name: str) -> str:
    """
    Gibt 'edible' oder 'poisonous' für eine gegebene Pilzart zurück.
    Wirft einen Fehler, falls die Klasse noch nicht im Mapping ist -
    das ist Absicht, damit nichts versehentlich falsch zugeordnet wird.
    """
    if class_name not in CLASS_TO_LABEL:
        raise KeyError(
            f"Klasse '{class_name}' ist noch nicht in CLASS_TO_LABEL "
            f"eingetragen. Bitte in class_mapping.py ergänzen."
        )
    return CLASS_TO_LABEL[class_name]


def validate_mapping(found_classes: list[str]) -> None:
    """
    Prüft, ob für alle im Datensatz gefundenen Ordner/Klassen ein Mapping
    existiert. Gibt eine verständliche Fehlermeldung mit den fehlenden
    Klassen aus, statt nur irgendwo mitten im Training zu crashen.
    """
    missing = [c for c in found_classes if c not in CLASS_TO_LABEL]
    if missing:
        raise ValueError(
            "Folgende Klassen fehlen noch im Mapping (class_mapping.py):\n"
            + "\n".join(f"  - {m}" for m in missing)
        )
    print(f"OK: Alle {len(found_classes)} Klassen sind im Mapping enthalten.")


if __name__ == "__main__":
    # Kleiner Selbsttest
    test_classes = list(CLASS_TO_LABEL.keys())
    validate_mapping(test_classes)
    for c in test_classes[:3]:
        print(f"{c} -> {get_label_for_class(c)}")