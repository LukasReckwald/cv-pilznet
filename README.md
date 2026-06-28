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
