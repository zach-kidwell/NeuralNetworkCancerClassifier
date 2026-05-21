from __future__ import annotations

import argparse

import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from tqdm import tqdm

from src import config
from src.checkpoint import load_model
from src.dataset import build_dataloaders
from src.utils import ensure_dirs, save_json
from src.visualize import plot_confusion_matrix, plot_sample_predictions, set_plot_style


def collect_predictions(model, loader, device: str = config.DEVICE):
    y_true, y_pred, y_conf = [], [], []
    correct_samples, incorrect_samples = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            y_conf.extend(confidences.cpu().tolist())

            for image, label, prediction, confidence in zip(
                images.cpu(), labels.cpu(), predictions.cpu(), confidences.cpu()
            ):
                sample = {
                    "image": image,
                    "label": int(label),
                    "pred": int(prediction),
                    "confidence": float(confidence),
                }
                if label == prediction and len(correct_samples) < 8:
                    correct_samples.append(sample)
                elif label != prediction and len(incorrect_samples) < 8:
                    incorrect_samples.append(sample)

    return y_true, y_pred, y_conf, correct_samples, incorrect_samples


def evaluate(args) -> dict:
    set_plot_style()
    ensure_dirs(config.OUTPUTS_DIR, config.FIGURES_DIR, config.PREDICTIONS_DIR)

    _, _, test_loader = build_dataloaders(batch_size=args.batch_size)
    model, checkpoint = load_model(args.model_path, config.DEVICE)

    y_true, y_pred, _, correct_samples, incorrect_samples = collect_predictions(model, test_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0),
    }

    save_json(metrics, config.METRICS_PATH)
    plot_confusion_matrix(y_true, y_pred)
    plot_sample_predictions(
        correct_samples,
        config.PREDICTIONS_DIR / "sample_predictions.png",
        "Sample Predictions with Confidence",
    )
    plot_sample_predictions(
        incorrect_samples,
        config.PREDICTIONS_DIR / "incorrect_predictions.png",
        "Incorrect Predictions",
    )

    print(metrics["classification_report"])
    print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained cancer image classifier.")
    parser.add_argument("--model-path", type=str, default=str(config.MODEL_PATH))
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
