from torch import nn
from abc import ABC, abstractmethod


class BaseLogger(ABC, nn.Module):
    @abstractmethod
    def log(self, *args, **kwargs):
        pass
