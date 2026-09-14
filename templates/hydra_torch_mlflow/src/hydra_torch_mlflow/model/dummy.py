import logging

import torch
from torch import nn

from .base import BaseModel

log = logging.getLogger(__name__)


class DummyModel(BaseModel):
    """Small conv classifier, for smoke-testing the pipeline."""

    def __init__(self, in_channels: int = 3, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log.info(f"DummyModel forwarding input of shape {tuple(x.shape)}")
        x = self.features(x).flatten(1)
        return self.classifier(x)
