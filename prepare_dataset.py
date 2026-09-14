"""
Build numbers+letters labels and dataset from:
  - Drive numbers folder (all class subfolders under raw/numbers)
  - Existing alphabet NPZs (LETTER_A–Z)
  - Optional FSL ONE–TEN if Drive is missing some of 1–10

Local id layout (dynamic):
  0 .. N-1   = number classes (sorted by numeric value when possible)
  N .. N+25  = LETTER_A .. LETTER_Z

Usage:
  .\\.venv\\Scripts\\python.exe specialists\\numbers_letters\\prepare_dataset.py
  .\\.venv\\Scripts\\python.exe specialists\\numbers_letters\\prepare_dataset.py --preprocess-drive
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORD_TO_INT = {
    "ZERO": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
    "TEN": 10,
    "ELEVEN": 11,
    "TWELVE": 12,
    "THIRTEEN": 13,
    "FOURTEEN": 14,
    "FIFTEEN": 15,
    "SIXTEEN": 16,
    "SEVENTEEN": 17,
    "EIGHTEEN": 18,
    "NINETEEN": 19,
    "TWENTY": 20,
    "THIRTY": 30,
    "FORTY": 40,
    "FIFTY": 50,
    "SIXTY": 60,
    "SEVENTY": 70,
    "EIGHTY": 80,
    "NINETY": 90,
    "HUNDRED": 100,
    "ONEHUNDRED": 100,
    "ONE_HUNDRED": 100,
    "TWO_HUNDRED": 200,
    "THREE_HUNDRED": 300,
    "FOUR_HUNDRED": 400,
    "FIVE_HUNDRED": 500,
    "SIX_HUNDRED": 600,
    "SEVEN_HUNDRED": 700,
    "EIGHT_HUNDRED": 800,
    "NINE_HUNDRED": 900,
    "ONE_THOUSAND": 1000,
    "TWO_THOUSAND": 2000,
    "THREE_THOUSAND": 3000,
    "FOUR_THOUSAND": 4000,
    "FIVE_THOUSAND": 5000,
    "ONE_MILLION": 1_000_000,
    "ONE_BILLION": 1_000_000_000,
    "ONE_TRILLION": 1_000_000_000_000,
}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _norm_label(name: str) -> str:
    s = name.strip().replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s.upper().strip("_")


def _label_sort_key(label: str) -> tuple:
    """Sort number folders by numeric value when possible."""
    lab = _norm_label(label)
    if lab.isdigit():
        return (0, int(lab), lab)
    # TWENTY_ONE / TWENTYONE
    parts = lab.split("_")
    if lab in WORD_TO_INT:
        return (0, WORD_TO_INT[lab], lab)
    if len(parts) == 2 and parts[0] in WORD_TO_INT and parts[1] in WORD_TO_INT:
        tens, ones = WORD_TO_INT[parts[0]], WORD_TO_INT[parts[1]]
        if tens >= 20 and ones < 10:
            return (0, tens + ones, lab)
    # compact TWENTYONE
    for tens_w, tens_v in [
        ("TWENTY", 20),
        ("THIRTY", 30),
        ("FORTY", 40),
        ("FIFTY", 50),
        ("SIXTY", 60),
        ("SEVENTY", 70),
        ("EIGHTY", 80),
        ("NINETY", 90),
    ]:
        if lab.startswith(tens_w) and lab != tens_w:
            rest = lab[len(tens_w) :].lstrip("_")
            if rest in WORD_TO_INT and WORD_TO_INT[rest] < 10:
                return (0, tens_v + WORD_TO_INT[rest], lab)
    m = re.fullmatch(r"(?:NUMBER[_-]?)?(\d{1,18})", lab)
    if m:
        return (0, int(m.group(1)), lab)
    return (1, 10**18, lab)


def find_numbers_root() -> Path | None:
    candidates = [
        SPEC / "raw" / "numbers",
        SPEC / "raw" / "drive_download" / "numbers",
    ]
    # Also search one level under drive_download
    dd = SPEC / "raw" / "drive_download"
    if dd.exists():
        for p in dd.rglob("*"):
            if p.is_dir() and p.name.lower() == "numbers":
                candidates.append(p)
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    return None


def load_expected_drive_classes() -> list[str]:
    """All Drive number folders (even before download finishes)."""
    path = SPEC / "drive_number_classes.txt"
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def discover_number_classes(numbers_root: Path) -> dict[str, list[Path]]:
    """Map canonical label -> video paths. Keeps empty class folders."""
    buckets: dict[str, list[Path]] = defaultdict(list)
    # Prefer immediate subdirs as classes
    subdirs = [d for d in numbers_root.iterdir() if d.is_dir()]
    if subdirs:
        for d in sorted(subdirs, key=lambda p: _label_sort_key(p.name)):
            label = canonicalize_number_label(d.name)
            # Always register the class (even if download failed for every file)
            _ = buckets[label]
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.stat().st_size > 0:
                    buckets[label].append(p)
        return dict(buckets)

    # Flat videos: try to parse label from filename
    for p in numbers_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        if p.stat().st_size <= 0:
            continue
        stem = _norm_label(p.stem)
        # e.g. SRC1_NUMBER_21_001 or twenty_one / folder-style 1000000
        m = re.search(r"(?:NUMBER[_-]?)?(\d{1,18})", stem)
        if m:
            label = _number_label_from_int(int(m.group(1)))
        else:
            label = canonicalize_number_label(stem)
        buckets[label].append(p)
    return dict(buckets)


def _number_label_from_int(n: int) -> str:
    """Prefer word labels for common values; else NUMBER_N."""
    preferred = {
        0: "ZERO",
        1: "ONE",
        2: "TWO",
        3: "THREE",
        4: "FOUR",
        5: "FIVE",
        6: "SIX",
        7: "SEVEN",
        8: "EIGHT",
        9: "NINE",
        10: "TEN",
        11: "ELEVEN",
        12: "TWELVE",
        13: "THIRTEEN",
        14: "FOURTEEN",
        15: "FIFTEEN",
        16: "SIXTEEN",
        17: "SEVENTEEN",
        18: "EIGHTEEN",
        19: "NINETEEN",
        20: "TWENTY",
        30: "THIRTY",
        40: "FORTY",
        50: "FIFTY",
        60: "SIXTY",
        70: "SEVENTY",
        80: "EIGHTY",
        90: "NINETY",
        100: "ONE_HUNDRED",
        200: "TWO_HUNDRED",
        300: "THREE_HUNDRED",
        400: "FOUR_HUNDRED",
        500: "FIVE_HUNDRED",
        600: "SIX_HUNDRED",
        700: "SEVEN_HUNDRED",
        800: "EIGHT_HUNDRED",
        900: "NINE_HUNDRED",
        1000: "ONE_THOUSAND",
        2000: "TWO_THOUSAND",
        3000: "THREE_THOUSAND",
        4000: "FOUR_THOUSAND",
        5000: "FIVE_THOUSAND",
        1_000_000: "ONE_MILLION",
        1_000_000_000: "ONE_BILLION",
        1_000_000_000_000: "ONE_TRILLION",
    }
    if n in preferred:
        return preferred[n]
    if 21 <= n <= 99:
        tens = (n // 10) * 10
        ones = n % 10
        if ones == 0:
            return preferred[tens]
        return f"{preferred[tens]}_{preferred[ones]}"
    return f"NUMBER_{n}"


def canonicalize_number_label(name: str) -> str:
    """Map folder/file class name to a single canonical gloss label."""
    lab = _norm_label(name)
    if lab.isdigit():
        return _number_label_from_int(int(lab))
    m = re.fullmatch(r"NUMBER_(\d{1,18})", lab)
    if m:
        return _number_label_from_int(int(m.group(1)))
    # Already a word form we know, or keep as-is
    if lab in WORD_TO_INT or lab.startswith(("LETTER_",)):
        return lab
    # TWENTY_ONE style already fine
    return lab


def write_labels(number_labels: list[str], path: Path) -> tuple[dict[str, int], int]:
    """Write labels_reference.csv; return label->id and letter_offset."""
    rows = []
    label_to_id: dict[str, int] = {}
    for i, lab in enumerate(number_labels):
        canon = canonicalize_number_label(lab)
        label_to_id[canon] = i
        label_to_id[lab] = i
        # Also map raw digit form when known
        if canon in WORD_TO_INT:
            label_to_id[str(WORD_TO_INT[canon])] = i
        m = re.fullmatch(r"NUMBER_(\d{1,18})", canon)
        if m:
            label_to_id[m.group(1)] = i
        rows.append(
            {
                "gloss_id": str(i),
                "label": canon,
                "cat_id": "0",
                "category": "NUMBER",
            }
        )

    letter_offset = len(number_labels)
    for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        gid = letter_offset + i
        rows.append(
            {
                "gloss_id": str(gid),
                "label": f"LETTER_{ch}",
                "cat_id": "1",
                "category": "ALPHABET",
            }
        )
        label_to_id[f"LETTER_{ch}"] = gid

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gloss_id", "label", "cat_id", "category"])
        w.writeheader()
        w.writerows(rows)
    print(f"[labels] {path} → {len(number_labels)} numbers + 26 letters = {len(rows)} glosses")
    return label_to_id, letter_offset


def preprocess_videos(
    videos_by_label: dict[str, list[Path]],
    out_npz_root: Path,
    max_per_class: int,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Holistic extract → NPZ; return train/val CSV rows with local gloss ids later filled."""
    from preprocessing.extractors.keypoints_features import (
        close_models,
        create_models,
        extract_keypoints_from_frame,
        interpolate_gaps,
    )
    import cv2
    import numpy as np

    random.seed(seed)
    out_train = out_npz_root / "train"
    out_val = out_npz_root / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    models = create_models(seg_model=1, detection_conf=0.35, tracking_conf=0.35)
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    def video_to_X(path: Path, max_frames: int = 90) -> np.ndarray | None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(src_fps / 30.0)))
        frames = []
        masks = []
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue
            idx += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (256, 256))
            vec, mask = extract_keypoints_from_frame(rgb, models, conf_thresh=0.35)
            frames.append(vec)
            masks.append(mask)
            if len(frames) >= max_frames:
                break
        cap.release()
        if len(frames) < 5:
            return None
        X = np.stack(frames, axis=0).astype(np.float32)
        M = np.stack(masks, axis=0)
        X, M = interpolate_gaps(X, M, max_gap=5)
        if X.shape[0] < 30:
            pad = np.repeat(X[-1:], 30 - X.shape[0], axis=0)
            X = np.concatenate([X, pad], axis=0)
        return X.astype(np.float32)

    try:
        for label, paths in sorted(videos_by_label.items(), key=lambda kv: _label_sort_key(kv[0])):
            random.shuffle(paths)
            paths = paths[:max_per_class]
            n_val = max(1, int(len(paths) * val_ratio)) if len(paths) >= 5 else max(0, len(paths) // 5)
            kept = 0
            for i, p in enumerate(paths):
                X = video_to_X(p)
                if X is None:
                    continue
                split = "val" if i < n_val else "train"
                out_dir = out_val if split == "val" else out_train
                stem = f"num_{_norm_label(label)}_{i:04d}"
                np.savez_compressed(out_dir / f"{stem}.npz", X=X)
                row = {
                    "file": stem,
                    "gloss_label": _norm_label(label),
                    "cat": "0",
                    "occluded": "0",
                    "signer": "DRIVE",
                    "duration": str(float(X.shape[0]) / 30.0),
                }
                (val_rows if split == "val" else train_rows).append(row)
                kept += 1
                if kept % 10 == 0:
                    print(f"  [{label}] {kept}/{len(paths)}")
            print(f"[drive numbers] {label}: kept {kept}/{len(paths)}")
    finally:
        close_models(models)

    return train_rows, val_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "gloss", "cat", "occluded", "signer", "duration"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])
    print(f"[csv] {path} ({len(rows)} rows)")


def _link(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)
    return True


def add_alphabet(
    train_rows: list[dict],
    val_rows: list[dict],
    letter_offset: int,
    max_per_class: int,
) -> None:
    """Append LETTER NPZs remapped to local ids starting at letter_offset."""
    mapping = {105 + i: letter_offset + i for i in range(26)}
    counts_tr: dict[int, int] = defaultdict(int)
    counts_va: dict[int, int] = defaultdict(int)

    def _add(src_csv: Path, src_dir: Path, out_dir: Path, rows: list[dict], counts: dict[int, int]):
        if not src_csv.exists():
            return
        with src_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                g = int(row["gloss"])
                if g not in mapping:
                    continue
                local = mapping[g]
                if counts[local] >= max_per_class:
                    continue
                stem = row["file"]
                src = src_dir / f"{stem}.npz"
                out_stem = f"nl_{local:03d}_{stem}"
                if not _link(src, out_dir / f"{out_stem}.npz"):
                    continue
                rows.append(
                    {
                        "file": out_stem,
                        "gloss": str(local),
                        "cat": "1",
                        "occluded": row.get("occluded", "0"),
                        "signer": row.get("signer", "ALPH"),
                        "duration": row.get("duration", "1.0"),
                    }
                )
                counts[local] += 1

    _add(
        ROOT / "data/processed/alphabet_train.csv",
        ROOT / "data/processed/alphabet_train",
        SPEC / "data" / "train",
        train_rows,
        counts_tr,
    )
    _add(
        ROOT / "data/processed/alphabet_val.csv",
        ROOT / "data/processed/alphabet_val",
        SPEC / "data" / "val",
        val_rows,
        counts_va,
    )
    print(f"[alphabet] added train letters={sum(counts_tr.values())} val={sum(counts_va.values())}")


def add_fsl_one_ten_missing(
    number_labels: list[str],
    label_to_id: dict[str, int],
    train_rows: list[dict],
    val_rows: list[dict],
    max_per_class: int,
) -> None:
    """Fill ONE–TEN from FSL105 if Drive did not provide them."""
    needed = []
    for word, n in [
        ("ONE", 1),
        ("TWO", 2),
        ("THREE", 3),
        ("FOUR", 4),
        ("FIVE", 5),
        ("SIX", 6),
        ("SEVEN", 7),
        ("EIGHT", 8),
        ("NINE", 9),
        ("TEN", 10),
    ]:
        if word in label_to_id or str(n) in label_to_id:
            # already have class; still ok to add more samples
            pass
        else:
            continue
        local = label_to_id.get(word) or label_to_id.get(str(n))
        if local is None:
            continue
        needed.append((20 + (n - 1), local, word))  # global FSL gloss 20=ONE

    if not needed:
        # Map by word if labels use words
        for i, word in enumerate(
            ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"]
        ):
            if word in label_to_id:
                needed.append((20 + i, label_to_id[word], word))

    counts = defaultdict(int)

    def _add(src_csv, src_dir, out_dir, rows):
        if not src_csv.exists():
            return
        global_to_local = {g: loc for g, loc, _ in needed}
        with src_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                g = int(row["gloss"])
                if g not in global_to_local:
                    continue
                local = global_to_local[g]
                if counts[local] >= max_per_class:
                    continue
                stem = row["file"]
                out_stem = f"nl_{local:03d}_{stem}"
                if not _link(src_dir / f"{stem}.npz", out_dir / f"{out_stem}.npz"):
                    continue
                rows.append(
                    {
                        "file": out_stem,
                        "gloss": str(local),
                        "cat": "0",
                        "occluded": row.get("occluded", "0"),
                        "signer": row.get("signer", "FSL"),
                        "duration": row.get("duration", "1.0"),
                    }
                )
                counts[local] += 1

    _add(
        ROOT / "data/processed/FSL105_train.csv",
        ROOT / "data/processed/FSL105_train",
        SPEC / "data" / "train",
        train_rows,
    )
    _add(
        ROOT / "data/processed/FSL105_val.csv",
        ROOT / "data/processed/FSL105_val",
        SPEC / "data" / "val",
        val_rows,
    )
    print(f"[fsl one-ten] supplemental samples={sum(counts.values())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=200)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preprocess-drive",
        action="store_true",
        help="Run Holistic preprocess on Drive number videos (slow)",
    )
    parser.add_argument(
        "--skip-drive-preprocess",
        action="store_true",
        help="Only rebuild labels/CSV from already-preprocessed drive NPZs if present",
    )
    args = parser.parse_args()

    numbers_root = find_numbers_root()
    videos_by_label: dict[str, list[Path]] = {}
    if numbers_root:
        print(f"[scan] numbers root: {numbers_root}")
        videos_by_label = discover_number_classes(numbers_root)
        print(f"[scan] found {len(videos_by_label)} number classes on disk:")
        for lab, paths in sorted(videos_by_label.items(), key=lambda kv: _label_sort_key(kv[0])):
            print(f"  {lab}: {len(paths)} videos")
    else:
        print(
            "[warn] No Drive numbers folder yet. "
            "Place under specialists/numbers_letters/raw/numbers/ "
            "or wait for download. Falling back to FSL ONE–TEN + Drive manifest."
        )

    # Always include the full Drive inventory (even if videos not downloaded yet)
    for raw in load_expected_drive_classes():
        lab = canonicalize_number_label(raw)
        videos_by_label.setdefault(lab, [])

    # Ensure ONE–TEN exist (FSL supplement) even if Drive starts at 11
    for w in ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"]:
        videos_by_label.setdefault(w, [])

    # Canonical sorted number label list (dedupe by numeric value)
    by_value: dict[int, str] = {}
    extras: list[str] = []
    for lab in videos_by_label:
        key = _label_sort_key(lab)
        if key[0] == 0:
            # Prefer word form already stored
            by_value.setdefault(key[1], canonicalize_number_label(lab))
        else:
            extras.append(canonicalize_number_label(lab))
    number_labels = [by_value[k] for k in sorted(by_value)] + sorted(set(extras))

    print(f"[labels] including {len(number_labels)} number glosses (Drive + ONE–TEN)")

    label_to_id, letter_offset = write_labels(number_labels, SPEC / "labels_reference.csv")
    # Persist meta for train.py
    meta_path = SPEC / "vocab_meta.txt"
    meta_path.write_text(
        f"num_numbers={len(number_labels)}\nnum_letters=26\nnum_gloss={letter_offset + 26}\nnum_cat=2\n",
        encoding="utf-8",
    )

    train_dir = SPEC / "data" / "train"
    val_dir = SPEC / "data" / "val"
    # Clean previous specialist pack (keep raw)
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    val_rows: list[dict] = []

    drive_has_videos = any(len(v) > 0 for v in videos_by_label.values())
    if drive_has_videos and args.preprocess_drive and not args.skip_drive_preprocess:
        print("\n[preprocess] Drive number videos → Holistic NPZ…")
        d_train, d_val = preprocess_videos(
            videos_by_label,
            SPEC / "data",
            max_per_class=args.max_per_class,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        for r in d_train:
            lab = r.pop("gloss_label")
            # resolve id
            gid = label_to_id.get(lab) or label_to_id.get(_norm_label(lab))
            if gid is None and lab.isdigit():
                gid = label_to_id.get(_number_label_from_int(int(lab)))
            if gid is None:
                continue
            r["gloss"] = str(gid)
            train_rows.append(r)
        for r in d_val:
            lab = r.pop("gloss_label")
            gid = label_to_id.get(lab) or label_to_id.get(_norm_label(lab))
            if gid is None and lab.isdigit():
                gid = label_to_id.get(_number_label_from_int(int(lab)))
            if gid is None:
                continue
            r["gloss"] = str(gid)
            val_rows.append(r)
    elif drive_has_videos and not args.preprocess_drive:
        print(
            "\n[note] Drive videos found but --preprocess-drive not set. "
            "Labels updated; re-run with --preprocess-drive to extract NPZs."
        )

    # Always add alphabet
    add_alphabet(train_rows, val_rows, letter_offset, args.max_per_class)
    # Supplement ONE–TEN from FSL if useful
    add_fsl_one_ten_missing(number_labels, label_to_id, train_rows, val_rows, args.max_per_class)

    _write_csv(SPEC / "data" / "train.csv", train_rows)
    _write_csv(SPEC / "data" / "val.csv", val_rows)
    print(f"\nDone. Train={len(train_rows)} Val={len(val_rows)} glosses={letter_offset + 26}")
    print("Next: train with specialists/numbers_letters/train.py (reads vocab_meta.txt)")


if __name__ == "__main__":
    main()
