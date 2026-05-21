from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DEMO_DATA_DIR = DATA_DIR / "demo"
TRAIN_IMAGE_DIR = RAW_DATA_DIR / "train"
TEST_IMAGE_DIR = RAW_DATA_DIR / "test"
LABELS_CSV = RAW_DATA_DIR / "train_labels.csv"
DEMO_IMAGE_DIR = DEMO_DATA_DIR / "images"
DEMO_LABELS_CSV = DEMO_DATA_DIR / "labels.csv"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

IMAGE_SIZE = 96
BATCH_SIZE = 64
NUM_WORKERS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 10
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

CLASS_NAMES = ["No Cancer Detected", "Cancer Detected"]
MODEL_PATH = MODELS_DIR / "cancer_cnn.pth"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
