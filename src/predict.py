from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src import config
from src.checkpoint import load_model
from src.dataset import get_transforms


def predict_image(image_path: str | Path, model_path: str | Path = config.MODEL_PATH) -> dict:
    model, _ = load_model(Path(model_path), config.DEVICE)
    transform = get_transforms(train=False)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        confidence, predicted_class = torch.max(probabilities, dim=0)

    return {
        "class_index": int(predicted_class.item()),
        "label": config.CLASS_NAMES[int(predicted_class.item())],
        "confidence": float(confidence.item()),
        "probabilities": {
            config.CLASS_NAMES[index]: float(probabilities[index].item()) for index in range(len(config.CLASS_NAMES))
        },
    }
