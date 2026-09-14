"""
Streamlit demo for the Numbers + Letters specialist only.

Run from repo root:

  .\\.venv\\Scripts\\python.exe -m streamlit run specialists/numbers_letters/streamlit_app.py --server.port 8505

Or from this folder (with PYTHONPATH=repo root):

  streamlit run streamlit_app.py --server.port 8505
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch

SPEC = Path(__file__).resolve().parent
ROOT = SPEC.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.prediction.predict import ModelPredictor  # noqa: E402
from streamlit_app.components.data_processing import process_videos_unified  # noqa: E402

CKPT = SPEC / "trained_models" / "MediaPipeLSTM_best.pt"
CKPT_FALLBACKS = [
    CKPT,
    SPEC / "trained_models" / "MediaPipeLSTM_last.pt",
    SPEC / "trained_models" / "MediaPipeLSTM_numbers_letters_init.pt",
]
LABELS = SPEC / "labels_reference.csv"


def load_labels() -> tuple[dict[int, str], dict[int, str]]:
    gloss, cat = {}, {}
    with LABELS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gloss[int(row["gloss_id"])] = row["label"]
            cat[int(row["cat_id"])] = row["category"]
    return gloss, cat


@st.cache_resource
def get_predictor():
    ckpt = next((p for p in CKPT_FALLBACKS if p.exists()), None)
    if ckpt is None:
        return None, None
    predictor = ModelPredictor(
        model_type="mediapipe_lstm_isolated",
        checkpoint_path=str(ckpt),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    # ModelPredictor sizes heads from checkpoint — verify
    return predictor, ckpt


def predict_npz(predictor: ModelPredictor, npz_path: Path, gloss_map, cat_map) -> dict:
    raw = predictor.predict_from_npz_simple(str(npz_path))
    gid = int(raw["gloss_prediction"])
    cid = int(raw["category_prediction"])
    top5 = [
        (gloss_map.get(int(i), f"id_{i}"), float(p))
        for i, p in raw.get("gloss_top5", [])
    ]
    return {
        "gloss": gloss_map.get(gid, f"id_{gid}"),
        "gloss_id": gid,
        "gloss_p": float(raw["gloss_probability"]),
        "category": cat_map.get(cid, f"id_{cid}"),
        "category_p": float(raw["category_probability"]),
        "top5": top5,
    }


def main() -> None:
    st.set_page_config(page_title="Numbers + Letters Specialist", layout="wide")
    st.title("Numbers + Letters Specialist")
    st.caption(
        "Isolated MediaPipe-LSTM for ONE–TEN and LETTER_A–Z only. "
        "Phrase signs stay on the main FSL-105 app."
    )

    gloss_map, cat_map = load_labels()
    predictor, ckpt = get_predictor()

    with st.sidebar:
        st.header("Model")
        if ckpt is None:
            st.error(
                "No checkpoint in `specialists/numbers_letters/trained_models/`. "
                "Run `prepare_dataset.py` then `train.py` first."
            )
        else:
            st.success(f"Loaded: `{ckpt.name}`")
            st.code(str(ckpt), language="text")
        st.markdown(f"**Vocab:** {len(gloss_map)} glosses (Drive numbers + letters)")
        st.markdown(f"**Labels:** `{LABELS.name}`")

    if predictor is None:
        st.stop()

    tab_npz, tab_video = st.tabs(["NPZ upload", "Video upload"])

    with tab_npz:
        up = st.file_uploader("Upload Holistic NPZ (`X` [T,178])", type=["npz"], key="npz")
        if up is not None:
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                tmp.write(up.getvalue())
                tmp_path = Path(tmp.name)
            try:
                result = predict_npz(predictor, tmp_path, gloss_map, cat_map)
                st.subheader(f"{result['gloss']}  ({result['gloss_p']:.1%})")
                st.write(
                    f"Category: **{result['category']}** ({result['category_p']:.1%})"
                )
                st.markdown("**Top-5**")
                for name, p in result["top5"]:
                    st.write(f"- {name}: {p:.1%}")
            finally:
                tmp_path.unlink(missing_ok=True)

    with tab_video:
        vid = st.file_uploader(
            "Upload video (mp4 / mov / webm / avi)",
            type=["mp4", "mov", "webm", "avi"],
            key="video",
        )
        if vid is not None and st.button("Preprocess + Predict", type="primary"):
            with st.spinner("Extracting Holistic keypoints…"):
                processed = process_videos_unified(
                    [vid],
                    target_fps=30,
                    out_size=256,
                    write_keypoints=True,
                    write_iv3_features=False,
                    occ_detailed=False,
                )
            stem = Path(vid.name).stem
            npz_data = processed.get(stem)
            if not npz_data or "X" not in npz_data:
                st.error("Preprocessing failed — no keypoints extracted.")
            else:
                with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                    np.savez_compressed(tmp, X=np.asarray(npz_data["X"]))
                    tmp_path = Path(tmp.name)
                try:
                    result = predict_npz(predictor, tmp_path, gloss_map, cat_map)
                    st.subheader(f"{result['gloss']}  ({result['gloss_p']:.1%})")
                    st.write(
                        f"Category: **{result['category']}** ({result['category_p']:.1%})"
                    )
                    st.markdown("**Top-5**")
                    for name, p in result["top5"]:
                        st.write(f"- {name}: {p:.1%}")
                    st.download_button(
                        "Download NPZ",
                        data=tmp_path.read_bytes(),
                        file_name=f"{stem}_keypoints.npz",
                        mime="application/octet-stream",
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
