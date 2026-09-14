from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Base class for datasets."""

    @abstractmethod
    def __getitem__(self, idx: int):
        """Return a single sample."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""
        pass
