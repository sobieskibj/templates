import logging

import torch

from .base import BaseDataset

log = logging.getLogger(__name__)


class DummyDataset(BaseDataset):
    """Dataset returning random tensors, for smoke-testing the pipeline."""

    def __init__(self, n_samples: int = 8, channels: int = 3, size: int = 16):
        self.n_samples = n_samples
        self.channels = channels
        self.size = size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        log.info(f"DummyDataset returning sample {idx}")
        return torch.randn(self.channels, self.size, self.size)
