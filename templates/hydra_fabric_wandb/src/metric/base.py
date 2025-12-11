import abc

import torch


class BaseMetric(abc.ABC, torch.nn.Module):
    """Base class for metrics."""

    def __init__(self):
        super(BaseMetric, self).__init__()
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    @abc.abstractmethod
    def compute_and_log(self):
        """Compute metric from intermediate states and log results."""
        pass
