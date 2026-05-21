from __future__ import annotations

import argparse

import torch
from torch import nn, optim
from tqdm import tqdm

from src import config
from src.checkpoint import save_checkpoint
from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import ensure_dirs, save_json, set_seed
from src.visualize import plot_training_history, set_plot_style


def run_epoch(model, loader, criterion, optimizer=None, device: str = config.DEVICE) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def train(args) -> dict:
    set_seed(config.RANDOM_SEED)
    set_plot_style()
    ensure_dirs(config.MODELS_DIR, config.OUTPUTS_DIR, config.FIGURES_DIR)

    train_loader, val_loader, _ = build_dataloaders(batch_size=args.batch_size)
    model = build_model().to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val_loss = float("inf")

    print(f"Training on {config.DEVICE}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, None, config.DEVICE)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, config.MODEL_PATH)

    save_json(history, config.OUTPUTS_DIR / "training_history.json")
    plot_training_history(history)
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN for histopathologic cancer detection.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
