from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

from src import config
from src.utils import ensure_dirs


def download_dataset(output_dir: Path = config.RAW_DATA_DIR) -> None:
    """Download the Kaggle dataset with the Kaggle CLI.

    This requires a Kaggle account and an API token at ~/.kaggle/kaggle.json.
    """
    ensure_dirs(output_dir)
    zip_path = output_dir / "histopathologic-cancer-detection.zip"

    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "histopathologic-cancer-detection",
            "-p",
            str(output_dir),
        ],
        check=True,
    )

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(output_dir)

    print(f"Dataset is ready in {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download the Kaggle Histopathologic Cancer Detection dataset.")
    parser.add_argument("--output-dir", type=Path, default=config.RAW_DATA_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_dataset(args.output_dir)
