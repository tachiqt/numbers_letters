# Numbers + Letters (FSL specialist)

MediaPipe-LSTM specialist for **Filipino Sign Language numbers + LETTER_A–Z**.

Clone or copy this repo into [fslr-transformer-vs-iv3gru](https://github.com/remi-9/fslr-transformer-vs-iv3gru) as:

```text
specialists/numbers_letters/
```

Includes **training scripts**, **Streamlit UI**, and the **best checkpoint**.  
**Datasets are not stored in git** — build them locally with the steps below.

## Layout

```text
specialists/numbers_letters/
  prepare_dataset.py      # build train/val NPZs + labels
  train.py                # transfer-learn LSTM
  eval_static_letters.py  # eval on japorton letter NPZs
  streamlit_app.py        # specialist demo UI
  run_streamlit.bat
  labels_reference.csv    # 116 numbers + 26 letters = 142 glosses
  vocab_meta.txt
  data/                   # created by prepare_dataset.py (gitignored)
  trained_models/
    MediaPipeLSTM_best.pt
  raw/numbers/            # optional local number videos
```

## Vocab

| IDs | Classes |
|-----|---------|
| `0 … 115` | Numbers (ONE–TEN + Drive classes) |
| `116 … 141` | `LETTER_A` … `LETTER_Z` |
| Categories | `0 = NUMBER`, `1 = ALPHABET` |

---

## Datasets (README instructions)

You need **three** inputs. None of the raw media is committed here.

### 1) Numbers videos (Drive / FSL-105)

Folder of class subfolders (`11`, `12`, `TWENTY_ONE`, … or word names). Example path used during training:

```text
...\FSL-105\clips\numbers-...\numbers\
```

Put a copy under `specialists/numbers_letters/raw/numbers/` **or** pass `--numbers-root`.

### 2) Alphabet stills — [japorton/fsl-dataset](https://www.kaggle.com/datasets/japorton/fsl-dataset)

From the **parent** repo root:

```powershell
cd path\to\fslr-transformer-vs-iv3gru
.\.venv\Scripts\python.exe scripts\transfer_learning\download_alphabet_datasets.py
.\.venv\Scripts\python.exe scripts\transfer_learning\preprocess_alphabet.py `
  --max-per-class 300 --max-videos-per-class 100
```

Optional J/Z motion videos (same download script):  
[signnteam/asl-sign-language-alphabet-videos-j-z](https://www.kaggle.com/datasets/signnteam/asl-sign-language-alphabet-videos-j-z)

Requires Kaggle CLI auth once (`kaggle auth login` or `%USERPROFILE%\.kaggle\kaggle.json`).

Outputs Holistic NPZs under:

```text
data/processed/alphabet_train/
data/processed/alphabet_val/
```

### 3) ONE–TEN supplement

If Drive numbers start at 11+, `prepare_dataset.py` pulls ONE–TEN from parent FSL-105 processed NPZs when available.

### Build the specialist pack

```powershell
cd path\to\fslr-transformer-vs-iv3gru

# First time (slow Holistic on ~850 number clips):
.\.venv\Scripts\python.exe specialists\numbers_letters\prepare_dataset.py `
  --numbers-root "PATH\to\numbers" `
  --preprocess-drive --max-per-class 250

# Later rebuilds after re-preprocessing alphabet (reuse cached number NPZs):
.\.venv\Scripts\python.exe specialists\numbers_letters\prepare_dataset.py `
  --numbers-root "PATH\to\numbers" `
  --reuse-drive-npz --max-per-class 250
```

Writes `data/train`, `data/val`, `train.csv`, `val.csv`, and refreshes `labels_reference.csv`.

---

## Train

Needs pretrained FSL-105 MediaPipe-LSTM:

```text
trained_models/mediapipe_lstm/FSL105_classification/MediaPipeLSTM_best.pt
```

```powershell
.\.venv\Scripts\python.exe specialists\numbers_letters\train.py --epochs 400 --batch-size 16
```

Output: `specialists/numbers_letters/trained_models/MediaPipeLSTM_best.pt`

## Streamlit UI

```powershell
.\specialists\numbers_letters\run_streamlit.bat
```

Or:

```powershell
.\.venv\Scripts\python.exe -m streamlit run specialists\numbers_letters\streamlit_app.py --server.port 8505
```

Open http://localhost:8505 — upload Holistic NPZ or video (video is preprocessed then predicted).

You can also select **Numbers + Letters** in the main parent Streamlit app (port 8501) if that model entry is enabled.

## Eval static letters

After alphabet NPZs exist in the parent repo:

```powershell
.\.venv\Scripts\python.exe specialists\numbers_letters\eval_static_letters.py --split val
```

## Notes

- This pack is numbers + letters only; greetings/phrases stay on other specialists.
- Letter stills (japorton) and number clips remain under their original data licenses / local Drive access.
