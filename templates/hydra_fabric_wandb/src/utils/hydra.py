import logging
from pathlib import Path

import hydra

log = logging.getLogger(__name__)


def preprocess_config(config):
    """Sets config.exp.log_dir to logging directory and symlinks it to CWD."""

    # get logging directory
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # make symlink in current working directory if its not the same
    date_subdir = log_dir.relative_to(log_dir.parents[1])
    log_cwd = Path.cwd() / "outputs" / date_subdir

    if not log_cwd == log_dir and not log_cwd.exists():
        log_cwd.parent.mkdir(exist_ok=True, parents=True)

        # sometimes, the file already exists and we skip this case here
        try:
            log_cwd.symlink_to(log_dir, target_is_directory=True)

        except FileExistsError:
            log.info("Attempting to symlink to existing directory.")

    # save in config
    config.exp.log_dir = log_dir
