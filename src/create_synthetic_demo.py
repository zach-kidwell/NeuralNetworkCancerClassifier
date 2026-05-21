from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from src import config
from src.utils import ensure_dirs


def _make_tile(label: int, seed: int, difficulty: str) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)

    palettes = {
        "benign": ((150, 18), (155, 18), (215, 16)),
        "benign_dense": ((172, 24), (140, 18), (195, 22)),
        "cancer": ((220, 20), (100, 16), (145, 22)),
        "cancer_pale": ((198, 18), (122, 18), (165, 20)),
        "questionable": ((188, 22), (132, 18), (178, 22)),
    }
    red, green, blue = palettes[difficulty]
    base[:, :, 0] = rng.normal(*red, (config.IMAGE_SIZE, config.IMAGE_SIZE)).clip(0, 255)
    base[:, :, 1] = rng.normal(*green, (config.IMAGE_SIZE, config.IMAGE_SIZE)).clip(0, 255)
    base[:, :, 2] = rng.normal(*blue, (config.IMAGE_SIZE, config.IMAGE_SIZE)).clip(0, 255)

    image = Image.fromarray(base, mode="RGB").filter(ImageFilter.GaussianBlur(radius=0.4))
    draw = ImageDraw.Draw(image, "RGBA")

    cell_counts = {
        "benign": 10,
        "benign_dense": 18,
        "cancer": 34,
        "cancer_pale": 28,
        "questionable": 24,
    }
    cell_count = cell_counts[difficulty]
    for _ in range(cell_count):
        x = int(rng.integers(4, config.IMAGE_SIZE - 12))
        y = int(rng.integers(4, config.IMAGE_SIZE - 12))
        radius = int(rng.integers(3, 10 if label == 1 else 8))
        if difficulty.startswith("cancer"):
            fill = (95, 20, 115, 145)
        elif difficulty == "questionable":
            fill = (100, 45, 135, 115)
        else:
            fill = (85, 80, 155, 80)
        draw.ellipse((x, y, x + radius * 2, y + radius * 2), fill=fill)

    return image


def create_synthetic_demo(sample_count: int = 24, output_dir: Path = config.DEMO_DATA_DIR) -> None:
    """Create a tiny bundled demo dataset for UI testing and hosting previews."""
    image_dir = output_dir / "images"
    ensure_dirs(image_dir)

    rows = []
    patterns = [
        ("benign", 0),
        ("benign_dense", 0),
        ("questionable", 0),
        ("questionable", 1),
        ("cancer_pale", 1),
        ("cancer", 1),
    ]

    for index in range(sample_count):
        difficulty, label = patterns[index % len(patterns)]
        image_id = f"demo_{index:03d}"
        image = _make_tile(label, seed=config.RANDOM_SEED + index, difficulty=difficulty)
        image.save(image_dir / f"{image_id}.tif")
        rows.append({"id": image_id, "label": label, "difficulty": difficulty})

    pd.DataFrame(rows).to_csv(output_dir / "labels.csv", index=False)
    print(f"Created {sample_count} synthetic demo images in {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a tiny synthetic demo dataset for the Streamlit app.")
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=config.DEMO_DATA_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_synthetic_demo(args.sample_count, args.output_dir)
