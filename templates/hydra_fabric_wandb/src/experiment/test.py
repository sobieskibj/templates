
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

import utils

import logging
log = logging.getLogger(__name__)

def get_fabric(config):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric

def get_components(config, fabric):
    network = fabric.setup(instantiate(config.network))
    return network

def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))

def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    log.info(f'Launching Fabric')
    fabric = get_fabric(config)

    log.info(f'Building components')
    network = get_components(config, fabric)

    log.info(f'Initializing dataloader')
    dataloader = get_dataloader(config, fabric)

    with fabric.init_tensor():
        for batch_idx, batch in enumerate(dataloader):
            network.action()