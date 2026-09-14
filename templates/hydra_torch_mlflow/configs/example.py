from dataclasses import dataclass, field
from typing import Any

from hydra.conf import HydraConf, RunDir


@dataclass
class ExampleConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            # Environment variables parameterize the selected variants (DATA=... MODEL=...).
            # The second entry after the comma is the fallback when the variable is unset.
            {"dataset": "${oc.env:DATA,dummy}"},
            {"model": "${oc.env:MODEL,dummy}"},
            {"metric": "dummy"},
            {"logger": "mlflow"},
        ]
    )

    # Placeholders for instantiated components
    dataset: Any = None
    model: Any = None
    metric: Any = None
    logger: Any = None

    # Kwargs passed to the DataLoader built in the experiment
    dataloader: dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
        }
    )

    exp: dict[str, Any] = field(
        default_factory=lambda: {
            "seed": 42,
            # One directory per run: <OUTPUTS_DIR>/<date>/<time>/<short uuid>
            "log_dir": "${oc.env:OUTPUTS_DIR,./outputs}/${now:%Y-%m-%d}/${now:%H-%M-%S}/${short_uuid:}",
            "run_func": {"_target_": "hydra_torch_mlflow.experiment.example.run"},
        }
    )

    # Point hydra's run dir at exp.log_dir so hydra's own artifacts and the
    # logger's artifacts (run_path = ${hydra:runtime.output_dir}) share one dir.
    hydra: HydraConf = field(default_factory=lambda: HydraConf(run=RunDir(dir="${exp.log_dir}")))
