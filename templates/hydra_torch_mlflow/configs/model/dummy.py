from dataclasses import dataclass


@dataclass
class DummyModelConfig:
    _target_: str = "hydra_torch_mlflow.model.dummy.DummyModel"

    in_channels: int = 3
    n_classes: int = 2
