from dataclasses import dataclass


@dataclass
class DummyMetricConfig:
    _target_: str = "hydra_torch_mlflow.metric.dummy.DummyMetric"
