# cv-pilznet

![meme](meme.jpg)

## Setup

### Environment

Das Projekt benutzt eine eigene Conda-Umgebung namens `pilznet`, die alles Nötige mitgringt (PyTorch, torchvision, scitkit-learn, albumentations, etc.) sowie die Kaggle-CLI für den Download der Daten.

```bash
conda env create -f enviroment.yml
conda activate pilznet
```

### Datensatz herunterladen

**Wichtig: Daten NICHT ins Git-Repo.** `data/raw/` ist deshalb in `.gitignore`. 

**Einmalige Einrichtung pro Person:**

1. Kaggle-Account erstellen/anmelden
2. API-Token erstellen
3. Den angezeigten Token-String kopieren
4. In PowerShell den Ordner anlegen und den Token **ohne BOM/UTF-16** speichern (wichtig — Notepad o.ä. kann sonst eine falsche Kodierung verwenden, die die CLI nicht lesen kann):

```powershell
mkdir $env:USERPROFILE\.kaggle -Force
Set-Content -Path "$env:USERPROFILE\.kaggle\access_token" -Value "DEIN_TOKEN_HIER" -NoNewline -Encoding ascii
```

**Testen:**

```bash
kaggle auth login
```

oder gleich

```bash
kaggle datasets download -d zlatan599/mushroom1 -p data\raw --unzip
```

Danach sollte es so aussehen

```
data/raw/
    merged_dataset/
        Agaricus
            img001.jpg
            img002.jpg
            ...
        Amanita/
            ...
    test.csv
    train.csv
    val.csv
```

## Pipeline

Ziel: ein 169-Klassen-Artenklassifikator (Transfer Learning, MobileNetV2/ResNet50). Die essbar/giftig-Confusion-Matrix wird nicht separat trainiert, sondern aus den Art-Vorhersagen über `src/class_mapping.py` abgeleitet.

### 1. Labels aufbereiten

Führt die Kaggle-Split-CSVs mit `src/class_mapping.py` (essbar/giftig je Art) zusammen:

```bash
python -m src.build_labels
```

Erzeugt `data/processed/{train,val,test}.csv` mit Spalten `image_path,species,species_idx,binary_label,binary_idx`. Der Original-Datensatz enthält bewusste Duplikate zum Oversampling der Minderheitsklassen — die werden hier pro Split auf eindeutige Bilder reduziert (spart beim Training unnötiges wiederholtes Laden/Dekodieren derselben Datei). Die Klassenbalance beim Training übernimmt stattdessen ein `WeightedRandomSampler` in `src/train.py`. Die val/test-Splits waren schon vorher duplikat- und überschneidungsfrei.

### 2. Training (Google Colab)

Lokal ist keine NVIDIA-GPU vorhanden, daher läuft das eigentliche Training in Colab: `notebooks/train_colab.ipynb` hochladen, GPU-Laufzeit aktivieren, Zellen der Reihe nach ausführen (Repo/Daten holen, `build_labels`, Training für beide Architekturen). Am Ende `models/*.pt` und `results/*_history.json` herunterladen und lokal in die entsprechenden Ordner legen.

Für einen schnellen lokalen CPU-Smoke-Test der Trainingsschleife (kein echtes Training):

```bash
python -m src.train --arch mobilenet_v2 --epochs 2 --freeze-epochs 1 --batch-size 4 --limit 40
```

### 3. Auswertung

Läuft auf CPU (für reine Inferenz ausreichend schnell) und erzeugt beide Confusion Matrices:

```bash
python -m src.evaluate --checkpoint models/mobilenet_v2.pt --split test
```

Ergebnisse landen in `results/`:
- `<arch>_<split>_confusion_species.png` — 169×169 Confusion Matrix (zeilennormalisiert) + `_report_species.csv` (Precision/Recall/F1 je Art)
- `<arch>_<split>_confusion_binary.png` — essbar/giftig Confusion Matrix + `_report_binary.csv`

Für den Architekturvergleich `evaluate` für beide Checkpoints laufen lassen und die ausgegebene Accuracy/Macro-F1-Zeile vergleichen.
