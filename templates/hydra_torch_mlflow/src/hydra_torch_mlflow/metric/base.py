from abc import ABC, abstractmethod

from torch import nn


class BaseMetric(ABC, nn.Module):
    """Base class for metrics.

    Metrics accumulate state per batch in forward() and aggregate it in compute_and_log().
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Update internal state from a batch."""
        pass

    @abstractmethod
    def compute_and_log(self):
        """Compute the final value from the accumulated state."""
        pass
