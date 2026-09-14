import logging

import utils
from hydra.utils import instantiate
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def get_fabric(config):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


def get_components(config, fabric):
    network = fabric.setup(instantiate(config.network))
    # init metrics
    metrics = [fabric.setup(instantiate(m)) for m in config.metric.values()]
    return network, metrics


def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataloader))


def run(config: DictConfig):
    utils.hydra.preprocess_config(config)
    utils.wandb.setup_wandb(config)

    log.info("Launching Fabric")
    fabric = get_fabric(config)

    log.info("Building components")
    network, metrics = get_components(config, fabric)

    log.info("Initializing dataloader")
    dataloader = get_dataloader(config, fabric)

    with fabric.init_tensor():
        for batch_idx, batch in enumerate(dataloader):
            
            log_dict = ...

            for metric in metrics:
                metric(log_dict)

            logger.log(log_dict)

        for metric in metrics:
            metric.compute_and_log(fabric, logger)
