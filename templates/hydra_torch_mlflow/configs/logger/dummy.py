from dataclasses import dataclass


@dataclass
class DummyLoggerConfig:
    _target_: str = "hydra_torch_mlflow.logger.dummy.DummyLogger"
