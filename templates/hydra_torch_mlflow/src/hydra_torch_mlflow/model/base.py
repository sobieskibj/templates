from abc import ABC, abstractmethod

from torch import nn


class BaseModel(ABC, nn.Module):
    """Base class for models."""

    @abstractmethod
    def forward(self, x):
        """Run the model on a batch."""
        pass
