import logging
import random

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.example import ExampleConfig
from hydra_torch_mlflow.dataset.base import BaseDataset
from hydra_torch_mlflow.logger.base import BaseLogger
from hydra_torch_mlflow.metric.base import BaseMetric
from hydra_torch_mlflow.model.base import BaseModel
from hydra_torch_mlflow.utils.device import get_devices

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_components(config: ExampleConfig):
    """Instantiate all experiment components from their configs."""
    dataset: BaseDataset = instantiate(config.dataset)
    model: BaseModel = instantiate(config.model)
    metric: BaseMetric = instantiate(config.metric)
    # The logger receives the whole root config and persists it as config.yaml
    logger: BaseLogger = instantiate(config.logger, root_config=config, _recursive_=False)
    return dataset, model, metric, logger


def run(config: ExampleConfig):
    """Generic experiment entry point: dataset -> model -> metric -> logger."""
    seed_everything(config.exp.seed)
    _, device_gpu = get_devices()
    log.info(f"Running on device: {device_gpu}")

    dataset, model, metric, logger = make_components(config)
    model = model.to(device_gpu)

    dataloader = DataLoader(dataset, **config.dataloader)

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing"):
        batch = batch.to(device_gpu)
        with torch.no_grad():
            outputs = model(batch)

        metric(outputs)

        logger.log(
            {
                "metadata/batch": {"index": batch_idx, "shape": list(batch.shape)},
                "metrics/accuracy": metric.compute_and_log(),
                "data/outputs": outputs,
            },
            id=str(batch_idx),
        )

    logger.log({"metrics/dummy_mean": metric.compute_and_log()}, id="summary")
    log.info("Run finished successfully")
