import logging

import torch

from .base import BaseMetric

log = logging.getLogger(__name__)


class TestMetric(BaseMetric):
    def __init__(self):
        super(TestMetric, self).__init__()

    def __str__(self):
        return "TestMetric"

    @torch.no_grad()
    def forward(self, x_0, x_0_hat):
        log.info("Forwarding through TestMetric")

    def compute_and_log(self):
        log.info("Computing and logging TestMetric")
