"""Evaluate numbers+letters specialist on japorton static alphabet images.

Uses Holistic NPZs already extracted from Collated A–Z (data/processed/alphabet_*),
remapped to specialist gloss IDs (LETTER_A=116 … LETTER_Z=141).

Usage (repo root):
  .\\.venv\\Scripts\\python.exe specialists\\numbers_letters\\eval_static_letters.py
  .\\.venv\\Scripts\\python.exe specialists\\numbers_letters\\eval_static_letters.py --split both --with-guards
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.prediction.predict import ModelPredictor  # noqa: E402
from evaluation.prediction.static_jz_guard import apply_numbers_letters_guards  # noqa: E402

# Alphabet preprocess used FSL-105 letter IDs 105–130; specialist uses 116–141
ALPHA_TO_NL = {105 + i: 116 + i for i in range(26)}


def load_gloss_map(path: Path) -> dict[int, str]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["gloss_id"])] = row["label"]
    return out


def iter_rows(split: str):
    csv_path = ROOT / "data" / "processed" / f"alphabet_{split}.csv"
    npz_dir = ROOT / "data" / "processed" / f"alphabet_{split}"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = int(row["gloss"])
            if g not in ALPHA_TO_NL:
                continue
            stem = row["file"]
            npz = npz_dir / f"{stem}.npz"
            if not npz.exists():
                continue
            yield npz, ALPHA_TO_NL[g], stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "both"], default="val")
    ap.add_argument("--ckpt", type=Path, default=SPEC / "trained_models" / "MediaPipeLSTM_best.pt")
    ap.add_argument("--labels", type=Path, default=SPEC / "labels_reference.csv")
    ap.add_argument("--with-guards", action="store_true", help="Apply A/J/Z handshape guards")
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--out",
        type=Path,
        default=SPEC / "trained_models" / "static_letter_eval.csv",
    )
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"Missing checkpoint: {args.ckpt}")

    gloss_map = load_gloss_map(args.labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.ckpt.name} on {device} …")
    predictor = ModelPredictor(
        model_type="mediapipe_lstm_isolated",
        checkpoint_path=str(args.ckpt),
        device=device,
    )

    splits = ["train", "val"] if args.split == "both" else [args.split]
    per_class: dict[str, Counter] = defaultdict(Counter)
    rows_out = []
    seen_per: dict[int, int] = defaultdict(int)
    total = correct = 0
    letter_correct = letter_total = 0
    number_as_pred = 0

    import tempfile

    for split in splits:
        for npz_path, true_id, stem in iter_rows(split):
            if args.max_per_class and seen_per[true_id] >= args.max_per_class:
                continue
            seen_per[true_id] += 1
            true_name = gloss_map.get(true_id, f"id_{true_id}")

            # Predict via temp path (API expects file)
            raw = predictor.predict_from_npz_simple(str(npz_path))
            if args.with_guards:
                data = np.load(npz_path)
                X = data["X"] if "X" in data.files else None
                raw = apply_numbers_letters_guards(raw, X, gloss_map)

            pred_id = int(raw["gloss_prediction"])
            pred_name = gloss_map.get(pred_id, f"id_{pred_id}")
            conf = float(raw["gloss_probability"])
            ok = pred_id == true_id
            total += 1
            if ok:
                correct += 1
            if true_name.startswith("LETTER_"):
                letter_total += 1
                if ok:
                    letter_correct += 1
            if not pred_name.startswith("LETTER_"):
                number_as_pred += 1

            per_class[true_name]["n"] += 1
            if ok:
                per_class[true_name]["ok"] += 1
            else:
                per_class[true_name][f"as:{pred_name}"] += 1

            rows_out.append(
                {
                    "split": split,
                    "file": stem,
                    "true": true_name,
                    "pred": pred_name,
                    "conf": f"{conf:.4f}",
                    "correct": int(ok),
                    "guards": int(args.with_guards),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["split", "file", "true", "pred", "conf", "correct", "guards"]
        )
        w.writeheader()
        w.writerows(rows_out)

    print("\n========== Static letter eval ==========")
    print(f"checkpoint: {args.ckpt}")
    print(f"guards:     {args.with_guards}")
    print(f"samples:    {total}")
    print(f"overall:    {correct}/{total} = {100.0 * correct / max(total,1):.1f}%")
    print(
        f"letters:    {letter_correct}/{letter_total} = "
        f"{100.0 * letter_correct / max(letter_total,1):.1f}%"
    )
    print(f"pred=NUMBER (not LETTER_*): {number_as_pred}/{total}")
    print("\nPer-letter accuracy:")
    for letter in [f"LETTER_{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]:
        c = per_class.get(letter)
        if not c or c["n"] == 0:
            continue
        acc = 100.0 * c["ok"] / c["n"]
        # top confusion
        confusions = sorted(
            ((k[3:], v) for k, v in c.items() if k.startswith("as:")),
            key=lambda x: -x[1],
        )[:3]
        conf_s = ", ".join(f"{n}×{v}" for n, v in confusions) if confusions else "-"
        print(f"  {letter:10s}  {c['ok']:3d}/{c['n']:3d}  {acc:5.1f}%  confusions: {conf_s}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
