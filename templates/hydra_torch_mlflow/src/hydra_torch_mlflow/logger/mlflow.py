import atexit
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from .base import BaseLogger

log = logging.getLogger(__name__)


def to_json_serializable(val: Any) -> Any:
    """Recursively converts tensors, numpy types, and custom objects to JSON-serializable primitives."""
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if hasattr(val, "as_tensor"):
        val = val.as_tensor().detach().cpu().numpy()
    elif isinstance(val, torch.Tensor):
        val = val.detach().cpu().numpy()

    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.generic, np.number)):
        return val.item()
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, dict):
        return {str(k): to_json_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [to_json_serializable(v) for v in val]
    if hasattr(val, "__dict__"):
        return {str(k): to_json_serializable(v) for k, v in vars(val).items()}
    return str(val)


class MLFlowLogger(BaseLogger):
    def __init__(
        self,
        run_path: Union[str, Path],
        root_config: Optional[Union[DictConfig, Dict[str, Any]]] = None,
        log_images: bool = True,
        exclude_images: Optional[List[str]] = None,
        log_data: bool = True,
        exclude_data: Optional[List[str]] = None,
        log_metrics: bool = True,
        exclude_metrics: Optional[List[str]] = None,
        log_metadata: bool = True,
        exclude_metadata: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()

        # Directory resolution
        self.run_path = Path(run_path).resolve()
        self.logs_dir = self.run_path / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.log_images = log_images
        self.exclude_images = exclude_images or []
        self.log_data = log_data
        self.exclude_data = exclude_data or []
        self.log_metrics = log_metrics
        self.exclude_metrics = exclude_metrics or []
        self.log_metadata = log_metadata
        self.exclude_metadata = exclude_metadata or []

        self.counters: Dict[str, int] = defaultdict(int)
        # Metrics are run-level series: one global step per log() call that
        # contains metrics, independent of the per-sample counters.
        self._metric_step = 0

        self.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("TRACKING_URI")
        self.experiment_name = os.environ.get("EXPERIMENT_NAME")

        # Without a tracking server, skip MLflow entirely: the resolved config,
        # per-sample artifacts and metrics are all persisted under run_path only.
        self.mlflow_enabled = bool(self.tracking_uri)
        if not self.mlflow_enabled:
            log.info("No TRACKING_URI set: MLflow tracking disabled, logging locally only")

        self.run = None
        if self.mlflow_enabled:
            mlflow.set_tracking_uri(self.tracking_uri)
            self._init_mlflow_experiment()

            tags = kwargs.pop("tags", {})
            if mlflow.active_run() is None:
                self.run = mlflow.start_run(tags=tags, **kwargs)
            else:
                self.run = mlflow.active_run()
                if tags:
                    mlflow.set_tags(tags)

        atexit.register(self.close)

        if root_config is not None:
            self._log_config(root_config)
        else:
            log.warning(
                "root_config not provided - resolved config will not be persisted or uploaded"
            )

    def reset(self, id: Optional[Union[str, int]] = None):
        """Zero out the step counter for a specific sample ID, or all counters
        (including the run-level metric step) when no ID is given."""
        if id is not None:
            self.counters[str(id)] = 0
        else:
            self.counters.clear()
            self._metric_step = 0

    def _init_mlflow_experiment(self):
        if not self.experiment_name:
            log.warning(
                "EXPERIMENT_NAME environment variable not set. Using default MLflow experiment."
            )
            return

        try:
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(self.experiment_name)
            if exp is None:
                exp_id = client.create_experiment(name=self.experiment_name)
                mlflow.set_experiment(experiment_id=exp_id)
            else:
                mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            log.warning(f"Failed to set MLflow experiment '{self.experiment_name}': {e}")

    def _log_config(self, config: Union[DictConfig, Dict[str, Any]]):
        # 1. Parse configuration once, shared by the local and remote paths.
        if isinstance(config, DictConfig):
            yaml_text = OmegaConf.to_yaml(config, resolve=True)
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            import yaml

            yaml_text = yaml.dump(config, default_flow_style=False)
            config_dict = dict(config)

        # 2. Local copy (source of truth); a failure here must not block the upload.
        try:
            (self.run_path / "config.yaml").write_text(yaml_text, encoding="utf-8")
        except Exception as e:
            log.error(f"Failed to write local config.yaml: {e}")

        if not self.mlflow_enabled:
            return

        # 3. The full resolved config as an artifact (viewable/downloadable in the
        #    UI, byte-identical to the local file). This is the authoritative copy
        #    on the server, so a failure here is a hard error.
        try:
            mlflow.log_text(yaml_text, artifact_file="config.yaml")
        except Exception as e:
            log.error(f"Failed to upload config.yaml to MLflow - full config NOT on server: {e}")

        # 4. Flattened, searchable parameters - a best-effort mirror only (values
        #    capped at 500 chars); a failure here must never affect the artifact.
        try:
            flat_params = {}

            def _flatten(d: dict, prefix: str = ""):
                for k, v in d.items():
                    if prefix == "" and k == "logger":
                        continue
                    key = f"{prefix}.{k}" if prefix else str(k)
                    if isinstance(v, dict):
                        _flatten(v, key)
                    else:
                        val_str = str(v)
                        flat_params[key] = val_str[:497] + "..." if len(val_str) > 500 else val_str

            if isinstance(config_dict, dict):
                _flatten(config_dict)
                mlflow.log_params(flat_params)
        except Exception as e:
            log.warning(f"Failed to log params to MLflow (config.yaml artifact unaffected): {e}")

    def _process_visual_value(self, val: Any) -> Optional[np.ndarray]:
        if hasattr(val, "as_tensor"):
            val = val.as_tensor().detach().cpu().numpy()
        elif isinstance(val, torch.Tensor):
            val = val.detach().cpu().numpy()

        if isinstance(val, np.ndarray):
            val = np.squeeze(val)
            if val.ndim == 3 and val.shape[0] in (1, 3, 4):  # CHW -> HWC
                val = np.transpose(val, (1, 2, 0))
            if val.ndim == 3 and val.shape[-1] not in (1, 3, 4):  # 3D volume center slice
                val = val[val.shape[0] // 2]
            if val.dtype in (np.float32, np.float64):
                val_min, val_max = val.min(), val.max()
                if val_max > val_min:
                    val = (val - val_min) / (val_max - val_min) * 255.0
                val = val.astype(np.uint8)
            return val
        if isinstance(val, Image.Image):
            return np.array(val)
        return None

    def _log_images(self, log_dict: Dict[str, Any], sample_id: str, step: int):
        images_dict = {
            k: v for k, v in log_dict.items() if not any(e in k for e in self.exclude_images)
        }
        images_dict = {
            k.replace("images/", "", 1): self._process_visual_value(v)
            for k, v in images_dict.items()
            if k.startswith("images/")
        }

        for rel_key, img_array in images_dict.items():
            if img_array is None:
                continue

            save_path = self.logs_dir / sample_id / "images" / f"{rel_key}_{step}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img_array).save(save_path)

            if self.mlflow_enabled:
                try:
                    mlflow.log_image(img_array, f"{sample_id}/{rel_key}_{step}.png")
                except Exception as e:
                    log.warning(f"Remote image upload failed for {sample_id}/{rel_key}_{step}: {e}")

    def _log_data(self, log_dict: Dict[str, Any], sample_id: str, step: int):
        data_dict = {
            k: v for k, v in log_dict.items() if not any(e in k for e in self.exclude_data)
        }
        data_dict = {
            k.replace("data/", "", 1): v for k, v in data_dict.items() if k.startswith("data/")
        }

        if len(data_dict) > 0:
            target_dir = self.logs_dir / sample_id / "data"
            target_dir.mkdir(parents=True, exist_ok=True)

            for rel_path, tensor_val in data_dict.items():
                save_path = target_dir / f"{rel_path}_{step}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if hasattr(tensor_val, "as_tensor"):
                    tensor_val = tensor_val.as_tensor().detach().cpu()
                elif isinstance(tensor_val, torch.Tensor):
                    tensor_val = tensor_val.detach().cpu()

                torch.save(tensor_val, save_path)

    def _log_metrics(self, log_dict: Dict[str, Any]):
        """Log metric series: one run-level CSV per key, appended step by step.

        Metrics are observations, not per-sample snapshots: the step is a
        run-level counter (one per log() call), and each key forms a single
        MLflow metric series regardless of the sample IDs that log it.
        """
        metrics_dict = {
            k: v for k, v in log_dict.items() if not any(e in k for e in self.exclude_metrics)
        }
        metrics_dict = {
            k.replace("metrics/", "", 1): v
            for k, v in metrics_dict.items()
            if k.startswith("metrics/")
        }

        if len(metrics_dict) == 0:
            return

        step = self._metric_step

        # Persist locally: one CSV per key under logs/metrics/, appended per step
        target_dir = self.logs_dir / "metrics"
        target_dir.mkdir(parents=True, exist_ok=True)

        for rel_key, value in metrics_dict.items():
            save_path = target_dir / f"{rel_key}.csv"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            is_new = not save_path.exists()
            with open(save_path, "a", newline="", encoding="utf-8") as f:
                if is_new:
                    f.write("step,value\n")
                serialized = to_json_serializable(value)
                cell = (
                    serialized
                    if isinstance(serialized, (int, float, bool))
                    else json.dumps(serialized)
                )
                f.write(f"{step},{cell}\n")

        # Log a single MLflow series per key (server mode only). MLflow values
        # must be numeric; non-numeric entries are skipped there (the CSV keeps them).
        if self.mlflow_enabled:
            mlflow_metrics = {}
            for rel_key, value in metrics_dict.items():
                try:
                    mlflow_metrics[rel_key] = float(
                        value.item() if hasattr(value, "item") else value
                    )
                except (TypeError, ValueError):
                    log.warning(f"Non-numeric metric '{rel_key}' skipped in MLflow: {value!r}")
            if mlflow_metrics:
                try:
                    mlflow.log_metrics(mlflow_metrics, step=step)
                except Exception as e:
                    log.warning(f"Failed to log metrics to MLflow: {e}")

        self._metric_step += 1

    def _log_metadata(self, log_dict: Dict[str, Any], sample_id: str, step: int):
        metadata_dict = {
            k: v for k, v in log_dict.items() if not any(e in k for e in self.exclude_metadata)
        }
        metadata_dict = {
            k.replace("metadata/", "", 1): v
            for k, v in metadata_dict.items()
            if k.startswith("metadata/")
        }

        target_dir = self.logs_dir / sample_id / "metadata"
        target_dir.mkdir(parents=True, exist_ok=True)

        for rel_key, value in metadata_dict.items():
            save_path = target_dir / f"{rel_key}_{step}.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            serialized = to_json_serializable(value)
            if isinstance(serialized, dict):
                payload = {"step": step, **serialized}
            else:
                payload = {"step": step, "value": serialized}

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            if self.mlflow_enabled:
                try:
                    mlflow.log_dict(payload, f"{sample_id}/metadata/{rel_key}_{step}.json")
                except Exception as e:
                    log.debug(f"Remote artifact upload skipped for metadata {rel_key}_{step}: {e}")

    def log(self, log_dict: Dict[str, Any], id: Optional[Union[str, int]] = None, **kwargs):
        sample_id = str(id if id is not None else kwargs.get("sample_id", "default"))
        current_step = self.counters[sample_id]

        if self.log_images:
            self._log_images(log_dict, sample_id=sample_id, step=current_step)
        if self.log_data:
            self._log_data(log_dict, sample_id=sample_id, step=current_step)
        if self.log_metrics:
            self._log_metrics(log_dict)
        if self.log_metadata:
            self._log_metadata(log_dict, sample_id=sample_id, step=current_step)

        # Monotonically advance step counter for this sample
        self.counters[sample_id] += 1

    def close(self):
        if self.mlflow_enabled and mlflow.active_run() is not None:
            mlflow.end_run()
