"""
Train numbers+letters specialist MediaPipe-LSTM.

- Backbone transferred from FSL-105 classification checkpoint
- Heads replaced for this domain only (36 glosses, 2 categories)
- Does NOT mix phrase classes — avoids GOOD AFTERNOON vs LETTER_Z bleed

Usage (from repo root, after prepare_dataset.py):

  .\\.venv\\Scripts\\python.exe specialists\\numbers_letters\\train.py --epochs 400 --batch-size 16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SPEC = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mediapipe_lstm import MediaPipeLSTM  # noqa: E402


def read_vocab_meta() -> tuple[int, int]:
    """num_gloss, num_cat from prepare_dataset vocab_meta.txt (fallback 36/2)."""
    meta = SPEC / "vocab_meta.txt"
    num_gloss, num_cat = 36, 2
    if meta.exists():
        for line in meta.read_text(encoding="utf-8").splitlines():
            if line.startswith("num_gloss="):
                num_gloss = int(line.split("=", 1)[1])
            elif line.startswith("num_cat="):
                num_cat = int(line.split("=", 1)[1])
    # Also trust labels_reference.csv if present
    labels = SPEC / "labels_reference.csv"
    if labels.exists():
        import csv

        with labels.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            num_gloss = max(int(r["gloss_id"]) for r in rows) + 1
            num_cat = max(int(r["cat_id"]) for r in rows) + 1
    return num_gloss, num_cat


def transfer_backbone(
    pretrained: Path,
    num_gloss: int,
    num_cat: int,
    hidden1: int = 256,
    hidden2: int = 128,
    dropout: float = 0.3,
) -> MediaPipeLSTM:
    """Load FSL-105 weights into backbone; fresh heads for specialist vocab."""
    ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    old_gloss = state["gloss_head.weight"].shape[0]
    old_cat = state["category_head.weight"].shape[0]

    donor = MediaPipeLSTM(
        num_gloss=old_gloss,
        num_cat=old_cat,
        input_dim=178,
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=dropout,
    )
    donor.load_state_dict(state, strict=True)

    model = MediaPipeLSTM(
        num_gloss=num_gloss,
        num_cat=num_cat,
        input_dim=178,
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=dropout,
    )
    donor_sd = donor.state_dict()
    model_sd = model.state_dict()
    transferred = {}
    for k, v in donor_sd.items():
        if k.startswith("gloss_head") or k.startswith("category_head"):
            continue
        if k in model_sd and model_sd[k].shape == v.shape:
            transferred[k] = v
    model_sd.update(transferred)
    model.load_state_dict(model_sd)
    print(
        f"Transferred backbone from {pretrained.name} "
        f"(skipped heads {old_gloss}/{old_cat} → {num_gloss}/{num_cat}); "
        f"{len(transferred)} tensors copied"
    )
    return model


def freeze_backbone(model: MediaPipeLSTM) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("gloss_head") or name.startswith("category_head")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"Frozen backbone; trainable {n_train}/{n_all}")


def unfreeze_all(model: MediaPipeLSTM) -> None:
    for p in model.parameters():
        p.requires_grad = True
    print("Unfroze all parameters")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=ROOT
        / "trained_models/mediapipe_lstm/FSL105_classification/MediaPipeLSTM_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SPEC / "trained_models",
        help="Checkpoints stay inside specialists/numbers_letters/trained_models",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1.5e-4)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()

    num_gloss, num_cat = read_vocab_meta()
    print(f"Specialist vocab: num_gloss={num_gloss} num_cat={num_cat}")

    train_csv = SPEC / "data" / "train.csv"
    val_csv = SPEC / "data" / "val.csv"
    train_dir = SPEC / "data" / "train"
    val_dir = SPEC / "data" / "val"
    if not train_csv.exists() or not val_csv.exists():
        raise SystemExit("Missing data CSVs. Run prepare_dataset.py first.")

    if not args.pretrained.exists():
        raise SystemExit(f"Missing pretrained: {args.pretrained}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Ship specialist labels next to checkpoints
    labels_src = SPEC / "labels_reference.csv"
    if labels_src.exists():
        import shutil

        shutil.copy2(labels_src, args.output_dir / "labels_reference.csv")

    model = transfer_backbone(
        args.pretrained,
        num_gloss=num_gloss,
        num_cat=num_cat,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    unfreeze_all(model)
    # Optimizer must be None: training.train builds a single param-group AdamW.
    boot_path = args.output_dir / "MediaPipeLSTM_numbers_letters_init.pt"
    torch.save(
        {
            "epoch": 0,
            "model": model.state_dict(),
            "optimizer": None,
            "scaler": None,
            "scheduler": None,
            "best_metric": -1.0,
            "args": {
                "num_gloss": num_gloss,
                "num_cat": num_cat,
                "specialist": "numbers_letters",
                "transfer_from": str(args.pretrained),
            },
        },
        boot_path,
    )
    print(f"Wrote init checkpoint: {boot_path}")
    if args.init_only:
        return

    py = ROOT / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)

    cmd = [
        str(py),
        "-m",
        "training.train",
        "--model",
        "mediapipe_lstm_isolated",
        "--training-mode",
        "classification",
        "--keypoints-train",
        str(train_dir),
        "--keypoints-val",
        str(val_dir),
        "--labels-train-csv",
        str(train_csv),
        "--labels-val-csv",
        str(val_csv),
        "--num-gloss",
        str(num_gloss),
        "--num-cat",
        str(num_cat),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--alpha",
        "0.85",
        "--beta",
        "0.15",
        "--hidden1",
        str(args.hidden1),
        "--hidden2",
        str(args.hidden2),
        "--dropout",
        str(args.dropout),
        "--scheduler",
        "warmup_cosine",
        "--warmup-epochs",
        "5",
        "--grad-clip",
        "1.0",
        "--output-dir",
        str(args.output_dir),
        "--resume",
        str(boot_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    print("\n=== Running ===\n", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
    print(f"\nSpecialist training complete → {args.output_dir}")


if __name__ == "__main__":
    main()
