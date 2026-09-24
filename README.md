# Numbers + Letters (FSL specialist)

Training pack for **Filipino Sign Language numbers + LETTER_A–Z** (MediaPipe Holistic LSTM).

This is **dataset + training only** — not the full HandSpeak / Streamlit app.

Parent training code lives in [fslr-transformer-vs-iv3gru](https://github.com/remi-9/fslr-transformer-vs-iv3gru). Place this folder at:

`specialists/numbers_letters/`

## What’s included

| Path | Contents |
|------|----------|
| `data/train/`, `data/val/` | Holistic NPZ `[T,178]` + CSVs (Drive numbers + japorton alphabet + FSL ONE–TEN) |
| `labels_reference.csv` | 116 numbers + 26 letters = **142** glosses |
| `prepare_dataset.py` | Build / refresh the specialist pack |
| `train.py` | Transfer-learn MediaPipe-LSTM from FSL-105 |
| `eval_static_letters.py` | Eval on japorton static letter NPZs |
| `trained_models/MediaPipeLSTM_best.pt` | Best checkpoint (~59% val gloss) |

**Not included:** Streamlit UI, API, mobile app, raw Drive/Kaggle video downloads.

## Vocab

- Gloss IDs `0…115` — numbers (ONE–TEN + Drive classes)
- Gloss IDs `116…141` — `LETTER_A` … `LETTER_Z`
- Categories: `0=NUMBER`, `1=ALPHABET`

## Train (from parent repo root)

```powershell
cd path\to\fslr-transformer-vs-iv3gru
# optional: rebuild from local numbers videos + alphabet NPZs
.\.venv\Scripts\python.exe specialists\numbers_letters\prepare_dataset.py `
  --numbers-root "PATH\to\numbers" --reuse-drive-npz --max-per-class 250

.\.venv\Scripts\python.exe specialists\numbers_letters\train.py --epochs 400 --batch-size 16
```

Requires a pretrained FSL-105 MediaPipe-LSTM at:

`trained_models/mediapipe_lstm/FSL105_classification/MediaPipeLSTM_best.pt`

## Eval static letters

Alphabet Holistic NPZs must exist under parent `data/processed/alphabet_val/` (from japorton via `scripts/transfer_learning/`).

```powershell
.\.venv\Scripts\python.exe specialists\numbers_letters\eval_static_letters.py --split val
```

## Data sources

1. **Numbers** — local Drive / FSL-105 number clips (not in git; use `--numbers-root`)
2. **Letters** — [japorton/fsl-dataset](https://www.kaggle.com/datasets/japorton/fsl-dataset) (+ optional J/Z videos)
3. **ONE–TEN** — FSL-105 processed NPZs when Drive starts at 11+

## License / note

Research / demo pack. Letter stills are japorton; number clips remain with your local Drive rights.
