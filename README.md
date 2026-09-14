# Numbers + Letters specialist

Public specialist pack: **labels, training scripts, Streamlit, and checkpoints** for all Drive numbers + A–Z.

Phrases stay on the main FSL-105 model. This specialist is numbers + letters only.

### Where this folder belongs

Clone or copy this repo into your [fslr-transformer-vs-iv3gru](https://github.com/remi-9/fslr-transformer-vs-iv3gru) checkout as:

`specialists/numbers_letters/`

Training and Streamlit import models/preprocessing from that parent repo. Raw Drive videos and built NPZs are **not** in git (see `.gitignore`) — download numbers locally, then follow the steps below.

## Layout

```
specialists/numbers_letters/
  drive_number_classes.txt  ← full Drive number inventory
  labels_reference.csv      ← 116 numbers + 26 letters
  prepare_dataset.py
  train.py
  streamlit_app.py
  run_streamlit.bat
  data/                     ← train/val NPZs + CSVs
  trained_models/           ← MediaPipeLSTM_best.pt
  raw/numbers/              ← put your Drive download here
  README.md
```

## 1. Download numbers from Drive (manual)

Download the Drive **numbers** folder yourself, then place it so class folders sit directly under:

`specialists/numbers_letters/raw/numbers/`

Expected layout:

```
raw/numbers/
  11/*.mp4
  12/*.mp4
  ...
  100/*.mp4
  1000/*.mp4
  ...
```

If Drive gives you an extra wrapper (e.g. `numbers/numbers/11/...`), move the inner class folders up so `11`, `12`, … are immediate children of `raw/numbers/`.

You can overwrite any partial automatic download in that folder.

ONE–TEN are not required in Drive; they are supplemented from FSL during `prepare_dataset.py`.

## 2. Build NPZs + CSVs

From the **repo root**:

```powershell
cd "C:\Users\Mark Vincent Perez\OneDrive\Desktop\videos\fslr-transformer-vs-iv3gru"
.\.venv\Scripts\python.exe specialists\numbers_letters\prepare_dataset.py --preprocess-drive
```

This scans every class under `raw/numbers/`, extracts Holistic features, merges alphabet NPZs + FSL ONE–TEN, and writes `data/` + `labels_reference.csv`.

## 3. Train (checkpoints stay in this folder)

```powershell
.\.venv\Scripts\python.exe specialists\numbers_letters\train.py --epochs 40 --batch-size 16
```

Output: `specialists/numbers_letters/trained_models/MediaPipeLSTM_best.pt`

## 4. Streamlit (this specialist only)

```powershell
.\specialists\numbers_letters\run_streamlit.bat
```

Or:

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m streamlit run specialists\numbers_letters\streamlit_app.py --server.port 8505
```

Open **http://localhost:8505** — NPZ or video upload for numbers/letters only.

Main Pansinayan app (port 8503) remains for FSL phrases.

## Label map (local)

| Local id | Label |
|----------|--------|
| 0–9 | ONE … TEN (FSL supplement) |
| 10–115 | Drive numbers 11–99, hundreds, thousands, million/billion/trillion |
| 116–141 | LETTER_A … LETTER_Z |

Exact list: `drive_number_classes.txt` + `labels_reference.csv`.  
Categories: `0=NUMBER`, `1=ALPHABET`.
