from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src import config


def plot_training_history(history: dict, output_dir: Path = config.FIGURES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=180)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path: Path = config.FIGURES_DIR / "confusion_matrix.png") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=config.CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 6))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.xticks(rotation=25, ha="right")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.700, 0.540, 0.690]).view(3, 1, 1)
    std = torch.tensor([0.235, 0.270, 0.215]).view(3, 1, 1)
    image = image_tensor.cpu() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


def plot_sample_predictions(samples: list[dict], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        return

    columns = min(4, len(samples))
    rows = int(np.ceil(len(samples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, sample in zip(axes, samples):
        ax.imshow(denormalize(sample["image"]))
        ax.axis("off")
        ax.set_title(
            f"Pred: {config.CLASS_NAMES[sample['pred']]}\n"
            f"True: {config.CLASS_NAMES[sample['label']]}\n"
            f"Conf: {sample['confidence']:.1%}",
            fontsize=9,
        )

    for ax in axes[len(samples) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
