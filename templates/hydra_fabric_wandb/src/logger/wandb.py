import csv
import logging
from pathlib import Path

import torch
import wandb
from torchvision.utils import make_grid

from utils.helpers import min_max_scale

from .base import BaseLogger

log = logging.getLogger(__name__)


class WandBLogger(BaseLogger):
    def __init__(
        self,
        config,
        dir,
        group,
        name,
        log_images,
        exclude_images,
        log_data,
        exclude_data,
        log_metrics,
        exclude_metrics,
        log_metadata,
        exclude_metadata,
        **kwargs,
    ):
        super(WandBLogger, self).__init__()
        wandb.init(config=config, dir=dir, group=group, name=name, **kwargs)
        self.counter = 0
        self.log_images = log_images
        self.exclude_images = exclude_images
        self.log_data = log_data
        self.exclude_data = exclude_data
        self.log_metrics = log_metrics
        self.exclude_metrics = exclude_metrics
        self.log_metadata = log_metadata
        self.exclude_metadata = exclude_metadata

    def _log_images(self, log_dict):
        # exclude keys from self.exclude_images
        images_dict = {
            k: v
            for k, v in log_dict.items()
            if not any([k in e for e in self.exclude_images])
        }
        # extract image entries
        images_dict = {
            k.replace("images/", "", 1): wandb.Image(make_grid(min_max_scale(v)))
            for k, v in images_dict.items()
            if k.startswith("images/")
        }
        if len(images_dict) > 0:
            # log to wandb
            wandb.log(images_dict)

    def _log_data(self, log_dict):
        # exclude keys from self.exclude_data
        data_dict = {
            k: v
            for k, v in log_dict.items()
            if not any([k in e for e in self.exclude_data])
        }
        # extract data entries
        data_dict = {
            Path(wandb.run.dir) / (k + f"_{self.counter}.pt"): v
            for k, v in data_dict.items()
            if k.startswith("data/")
        }
        if len(data_dict) > 0:
            # make subdirectories for each entry if needed
            [p.parent.mkdir(parents=True, exist_ok=True) for p in data_dict.keys()]
            # save each entry to .pt file
            [torch.save(v, k) for k, v in data_dict.items()]
            # increase counter
            self.counter += 1

    def _log_metrics(self, log_dict):
        # exclude keys from self.exclude_metrics
        metrics_dict = {
            k: v
            for k, v in log_dict.items()
            if not any([k in e for e in self.exclude_metrics])
        }
        # extract metric entries
        metrics_dict = {
            k.replace("metrics/", "", 1): v
            for k, v in metrics_dict.items()
            if k.startswith("metrics/")
        }
        wandb.log(metrics_dict)

    def _log_metadata(self, log_dict):
        # exclude keys logic
        metadata_dict = {
            k: v
            for k, v in log_dict.items()
            if not any([k in e for e in self.exclude_metadata])
        }

        # extract metadata entries
        metadata_dict = {
            (Path(wandb.run.dir) / k).with_suffix(".csv"): v
            for k, v in metadata_dict.items()
            if k.startswith("metadata/")
        }

        for filepath, value in metadata_dict.items():
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # 1. Handle Tensor conversion
            if isinstance(value, torch.Tensor):
                value = value.numpy(force=True)

            # 2. Open file
            with open(filepath, "a", newline="") as f:
                writer = csv.writer(f)

                # 3. Handle shapes (Scalar vs Batch)
                if hasattr(value, "ndim") and value.ndim > 0:
                    # Case: (32, 2) Tensor/Array -> Write 32 rows with 2 cols each
                    writer.writerows(value)
                else:
                    # Case: Scalar (float/int) -> Wrap in list to write 1 row
                    writer.writerow([value])

    def log(self, log_dict):
        if self.log_images:
            self._log_images(log_dict)
        if self.log_data:
            self._log_data(log_dict)
        if self.log_metrics:
            self._log_metrics(log_dict)
        if self.log_metadata:
            self._log_metadata(log_dict)
