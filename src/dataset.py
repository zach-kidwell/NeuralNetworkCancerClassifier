from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class HistopathologyDataset(Dataset):
    """Dataset for Kaggle histopathology tiles.

    Kaggle labels each 96x96 tile as 1 when the center region contains tumor
    tissue and 0 otherwise. The image file name is the sample id plus ".tif".
    """

    def __init__(self, frame: pd.DataFrame, image_dir: Path, transform: Optional[transforms.Compose] = None):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.image_dir / f"{row.id}.tif"
        image = Image.open(image_path).convert("RGB")
        label = int(row.label)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(train: bool = False) -> transforms.Compose:
    """Create preprocessing and augmentation pipelines.

    CNNs learn spatial patterns. Random flips and rotations expose the model to
    realistic tissue orientations without changing the class label.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.700, 0.540, 0.690], std=[0.235, 0.270, 0.215]),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.700, 0.540, 0.690], std=[0.235, 0.270, 0.215]),
        ]
    )


def create_splits(labels_csv: Path = config.LABELS_CSV) -> DataSplits:
    labels = pd.read_csv(labels_csv)

    train_val, test = train_test_split(
        labels,
        test_size=config.TEST_SPLIT,
        stratify=labels["label"],
        random_state=config.RANDOM_SEED,
    )

    validation_fraction = config.VALIDATION_SPLIT / (1 - config.TEST_SPLIT)
    train, val = train_test_split(
        train_val,
        test_size=validation_fraction,
        stratify=train_val["label"],
        random_state=config.RANDOM_SEED,
    )

    return DataSplits(train=train, val=val, test=test)


def build_dataloaders(batch_size: int = config.BATCH_SIZE) -> tuple[DataLoader, DataLoader, DataLoader]:
    splits = create_splits()

    train_dataset = HistopathologyDataset(splits.train, config.TRAIN_IMAGE_DIR, get_transforms(train=True))
    val_dataset = HistopathologyDataset(splits.val, config.TRAIN_IMAGE_DIR, get_transforms(train=False))
    test_dataset = HistopathologyDataset(splits.test, config.TRAIN_IMAGE_DIR, get_transforms(train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)

    return train_loader, val_loader, test_loader
