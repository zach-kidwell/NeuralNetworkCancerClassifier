from pathlib import Path

import torch

from src.model import build_model


def save_checkpoint(model, optimizer, epoch: int, val_loss: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def load_model(path: Path, device: str):
    model = build_model().to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
