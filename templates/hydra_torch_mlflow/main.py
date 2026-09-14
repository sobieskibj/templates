import hydra
from omegaconf import DictConfig
from configs.registry import auto_register

auto_register()


@hydra.main(version_base=None)
def main(config: DictConfig):
    hydra.utils.call(config.exp.run_func, config)


if __name__ == "__main__":
    main()
