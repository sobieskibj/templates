from dataclasses import dataclass


@dataclass
class DummyDatasetConfig:
    _target_: str = "hydra_torch_mlflow.dataset.dummy.DummyDataset"

    n_samples: int = 8
    channels: int = 3
    size: int = 16
