import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src import config


st.set_page_config(page_title="Neural Network Image Classifier", layout="centered")


@st.cache_resource
def get_model(model_path: str):
    from src.checkpoint import load_model

    return load_model(Path(model_path), config.DEVICE)[0]


@st.cache_data
def load_labels(labels_path: str) -> pd.DataFrame:
    return pd.read_csv(labels_path)


def predict_dataset_image(image: Image.Image, model_path: Path) -> dict:
    if not model_path.exists():
        return predict_demo_image(image)

    import torch
    from src.dataset import get_transforms

    model = get_model(str(model_path))
    transform = get_transforms(train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        confidence, predicted_class = torch.max(probabilities, dim=0)

    class_index = int(predicted_class.item())
    return {
        "class_index": class_index,
        "label": config.CLASS_NAMES[class_index],
        "confidence": float(confidence.item()),
        "cancer_probability": float(probabilities[1].item()),
        "non_cancer_probability": float(probabilities[0].item()),
    }


def predict_demo_image(image: Image.Image) -> dict:
    """Fallback classifier for the hosted UI before a real CNN checkpoint exists.

    The real portfolio path still uses models/cancer_cnn.pth. This fallback only
    keeps the web app interactive when the Kaggle model has not been trained yet.
    """
    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    red_signal = array[:, :, 0].mean()
    green_signal = array[:, :, 1].mean()
    blue_signal = array[:, :, 2].mean()
    texture_signal = array.std()
    red_blue_gap = red_signal - blue_signal
    purple_texture = texture_signal + max(0.0, red_signal - green_signal) * 0.35
    cancer_probability = float(np.clip(0.48 + 1.45 * red_blue_gap + 0.55 * (purple_texture - 0.22), 0.06, 0.94))
    non_cancer_probability = 1.0 - cancer_probability
    class_index = int(cancer_probability >= 0.5)
    confidence = cancer_probability if class_index == 1 else non_cancer_probability

    return {
        "class_index": class_index,
        "label": config.CLASS_NAMES[class_index],
        "confidence": confidence,
        "cancer_probability": cancer_probability,
        "non_cancer_probability": non_cancer_probability,
        "demo_mode": True,
    }


def choose_random_sample(labels: pd.DataFrame) -> pd.Series:
    image_dir = get_active_image_dir()
    available = labels[labels["id"].apply(lambda image_id: (image_dir / f"{image_id}.tif").exists())]
    if available.empty:
        raise FileNotFoundError(f"No matching .tif images were found in {image_dir}.")
    return available.sample(n=1, random_state=random.randint(0, 1_000_000)).iloc[0]


def get_active_dataset() -> tuple[Path, Path, str]:
    raw_ready = config.LABELS_CSV.exists() and config.TRAIN_IMAGE_DIR.exists()
    demo_ready = config.DEMO_LABELS_CSV.exists() and config.DEMO_IMAGE_DIR.exists()

    if raw_ready:
        return config.LABELS_CSV, config.TRAIN_IMAGE_DIR, "Full Kaggle dataset"
    if demo_ready:
        return config.DEMO_LABELS_CSV, config.DEMO_IMAGE_DIR, "Bundled demo subset"
    return config.LABELS_CSV, config.TRAIN_IMAGE_DIR, "Dataset not found"


def get_active_image_dir() -> Path:
    _, image_dir, _ = get_active_dataset()
    return image_dir


st.markdown(
    """
    <style>
    .stApp {
        background: #ffffff;
        color: #0f172a;
    }
    h1 {
        text-align: center;
        font-family: Georgia, serif;
        font-size: 3rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    div.stButton {
        display: flex;
        justify-content: center;
    }
    div.stButton > button {
        background: #0d7df2;
        color: white;
        border: 0;
        border-radius: 6px;
        padding: 0.85rem 2.4rem;
        font-size: 1.1rem;
        min-width: 300px;
    }
    div.stButton > button:hover {
        background: #0969cb;
        color: white;
        border: 0;
    }
    .result-panel {
        background: #f3f3f3;
        border-radius: 6px;
        margin: 3.5rem auto 1rem auto;
        max-width: 860px;
        padding: 3rem 4rem;
        text-align: center;
    }
    .result-panel h2 {
        font-family: Georgia, serif;
        font-size: 2rem;
        margin-bottom: 1.2rem;
    }
    .prediction-label {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .prediction-meta {
        color: #475569;
        font-size: 0.98rem;
        margin-bottom: 1.5rem;
    }
    .scoreboard {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 8px;
        color: #1e1b4b;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 1.5rem auto 0 auto;
        max-width: 520px;
        padding: 1rem;
        text-align: center;
    }
    .scoreboard span {
        display: block;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 0.3rem;
    }
    .correct {
        color: #047857;
        font-weight: 700;
    }
    .incorrect {
        color: #b91c1c;
        font-weight: 700;
    }
    .status-note {
        margin: 2.5rem auto 0 auto;
        max-width: 760px;
    }
    .setup-panel {
        background: #fff7cc;
        border: 1px solid #eab308;
        border-radius: 8px;
        color: #1f2937;
        line-height: 1.55;
        margin: 3rem auto 1.5rem auto;
        max-width: 860px;
        padding: 1.5rem 1.75rem;
    }
    .setup-panel h2 {
        color: #111827;
        font-family: Georgia, serif;
        font-size: 1.7rem;
        margin: 0 0 0.75rem 0;
        text-align: center;
    }
    .setup-panel p,
    .setup-panel li {
        color: #1f2937;
        font-size: 1rem;
    }
    .setup-panel ul {
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Neural Network Image Classifier")

default_labels_path, active_image_dir, dataset_name = get_active_dataset()
model_path = Path(st.sidebar.text_input("Model checkpoint", value=str(config.MODEL_PATH)))
labels_path = Path(st.sidebar.text_input("Dataset labels", value=str(default_labels_path)))
st.sidebar.caption(f"Dataset: {dataset_name}")
st.sidebar.caption(f"Device: {config.DEVICE}")

if "sample_id" not in st.session_state:
    st.session_state.sample_id = None
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0
if "total_count" not in st.session_state:
    st.session_state.total_count = 0

button_clicked = st.button("Generate Random Classification")
if button_clicked:
    st.session_state.generate = True

dataset_ready = labels_path.exists() and active_image_dir.exists()
model_ready = model_path.exists()

if not dataset_ready:
    missing_items = []
    if not dataset_ready:
        missing_items.append("dataset files in data/raw or a bundled demo subset in data/demo")

    missing_html = "".join(f"<li>{item}</li>" for item in missing_items)
    action_text = (
        "You clicked the button, but there are no images available to classify yet."
        if button_clicked or st.session_state.get("generate")
        else "The classifier is waiting for dataset images."
    )
    st.markdown(
        f"""
        <div class="setup-panel">
            <h2>Classification Not Ready Yet</h2>
            <p>{action_text}</p>
            <p>Missing:</p>
            <ul>{missing_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(
        "py -m src.download_data\n"
        "py -m src.train --epochs 10\n"
        "py -m src.export_demo_assets --sample-count 50",
        language="bash",
    )
elif st.session_state.get("generate"):
    labels = load_labels(str(labels_path))
    sample = choose_random_sample(labels)
    image_path = active_image_dir / f"{sample.id}.tif"
    image = Image.open(image_path).convert("RGB")
    result = predict_dataset_image(image, model_path)
    true_class = int(sample.label)
    true_label = config.CLASS_NAMES[true_class]
    is_correct = result["class_index"] == true_class
    st.session_state.total_count += 1
    if is_correct:
        st.session_state.correct_count += 1

    accuracy = st.session_state.correct_count / st.session_state.total_count
    outcome_class = "correct" if is_correct else "incorrect"
    outcome_text = "Correct" if is_correct else "Incorrect"
    mode_note = (
        "Demo fallback classifier active. Train the CNN to use the real model."
        if result.get("demo_mode")
        else "CNN checkpoint prediction."
    )

    st.markdown(
        f"""
        <div class="result-panel">
            <h2>Classification Result:</h2>
            <div class="prediction-label">{result["label"]}</div>
            <div class="prediction-meta">
                Confidence: {result["confidence"]:.2%} |
                True Label: {true_label} |
                Image ID: {sample.id}
            </div>
            <div class="{outcome_class}">{outcome_text}</div>
            <div class="prediction-meta">{mode_note}</div>
            <div class="scoreboard">
                {st.session_state.correct_count} / {st.session_state.total_count} guessed correctly so far
                <span>{accuracy:.0%} correct</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.image(image, width=420)
    st.progress(result["cancer_probability"], text=f"Cancer probability: {result['cancer_probability']:.2%}")
    st.progress(result["non_cancer_probability"], text=f"No cancer probability: {result['non_cancer_probability']:.2%}")
else:
    accuracy = (
        st.session_state.correct_count / st.session_state.total_count
        if st.session_state.total_count
        else 0
    )
    st.markdown(
        f"""
        <div class="result-panel">
            <h2>Classification Result:</h2>
            <div class="prediction-meta">Click the button to classify a random image from the dataset.</div>
            <div class="scoreboard">
                {st.session_state.correct_count} / {st.session_state.total_count} guessed correctly so far
                <span>{accuracy:.0%} correct</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
