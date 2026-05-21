from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.utils import ensure_dirs


def export_demo_assets(
    sample_count: int,
    output_dir: Path = config.DEMO_DATA_DIR,
    labels_csv: Path = config.LABELS_CSV,
    image_dir: Path = config.TRAIN_IMAGE_DIR,
) -> None:
    """Copy a small stratified image subset for a hosted app demo.

    The full Kaggle dataset is too large for a normal GitHub/Render workflow.
    This creates a compact subset that can be committed with the app after the
    user has downloaded the dataset locally.
    """
    labels = pd.read_csv(labels_csv)
    labels = labels[labels["id"].apply(lambda image_id: (image_dir / f"{image_id}.tif").exists())]

    if labels.empty:
        raise FileNotFoundError("No labeled images were found. Download the Kaggle dataset first.")

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2.")

    sample_size = min(sample_count, len(labels))
    _, sample = train_test_split(
        labels,
        test_size=sample_size,
        stratify=labels["label"],
        random_state=config.RANDOM_SEED,
    )

    demo_image_dir = output_dir / "images"
    ensure_dirs(demo_image_dir)

    rows = []
    for row in sample.itertuples(index=False):
        source = image_dir / f"{row.id}.tif"
        destination = demo_image_dir / f"{row.id}.tif"
        shutil.copy2(source, destination)
        rows.append({"id": row.id, "label": int(row.label)})

    pd.DataFrame(rows).to_csv(output_dir / "labels.csv", index=False)
    print(f"Exported {len(rows)} demo images to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export a small dataset subset for hosted app demos.")
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=config.DEMO_DATA_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_demo_assets(args.sample_count, args.output_dir)
