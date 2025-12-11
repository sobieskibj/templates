import omegaconf
from hydra.utils import instantiate


def setup_wandb(config):
    """Sets up W&B run based on config."""
    group, name = config.exp.log_dir.parts[-2:]
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    instantiate(config.wandb)(
        config=wandb_config,
        dir=config.exp.log_dir,
        group=group,
        name=name,
    )
