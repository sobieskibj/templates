from dataclasses import dataclass, field
from typing import Any


@dataclass
class MLFlowLoggerConfig:
    _target_: str = "hydra_torch_mlflow.logger.mlflow.MLFlowLogger"
    _recursive_: bool = False  # Prevents Hydra from evaluating nested _target_ keys in kwargs
    # Resolves to the hydra run dir, which the root config points at exp.log_dir
    run_path: str = "${hydra:runtime.output_dir}"
    # Filled at runtime: the experiment passes the whole root config, which the
    # logger persists as config.yaml and logs as MLflow params
    root_config: Any = None

    log_images: bool = True
    exclude_images: list[str] = field(default_factory=list)

    log_data: bool = True
    exclude_data: list[str] = field(default_factory=list)

    log_metrics: bool = True
    exclude_metrics: list[str] = field(default_factory=list)

    log_metadata: bool = True
    exclude_metadata: list[str] = field(default_factory=list)
