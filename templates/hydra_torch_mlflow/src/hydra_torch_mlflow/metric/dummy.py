import logging

import torch

from .base import BaseMetric

log = logging.getLogger(__name__)


class DummyMetric(BaseMetric):
    """Running mean of the model outputs, for smoke-testing the pipeline."""

    def __init__(self):
        super().__init__()
        self.total = 0.0
        self.count = 0

    def forward(self, outputs: torch.Tensor):
        with torch.no_grad():
            self.total += outputs.float().mean().item()
            self.count += 1

    def compute_and_log(self) -> float:
        value = self.total / max(self.count, 1)
        log.info(f"DummyMetric computed value: {value}")
        return value
