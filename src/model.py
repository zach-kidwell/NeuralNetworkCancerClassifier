import torch
from torch import nn


class CancerCNN(nn.Module):
    """A compact CNN for 96x96 histopathology tiles.

    Convolution blocks learn local tissue textures, pooling reduces spatial
    size, and dropout helps reduce overfitting on highly similar image patches.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            self._conv_block(3, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.45),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model() -> CancerCNN:
    return CancerCNN(num_classes=2)
