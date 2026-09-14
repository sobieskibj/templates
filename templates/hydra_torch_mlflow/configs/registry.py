import importlib
import inspect
import logging
import pkgutil
import uuid
from dataclasses import is_dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import configs

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "short_uuid", lambda: uuid.uuid4().hex[:8], use_cache=True, replace=True
)


def auto_register() -> None:
    cs = ConfigStore.instance()

    for _, module_name, is_pkg in pkgutil.walk_packages(configs.__path__, configs.__name__ + "."):
        if is_pkg or module_name == "configs.registry":
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            log.error(f"Skipping {module_name} due to import error: {e}")
            continue

        parts = module_name.split(".")

        # Root-level configs (e.g., configs.sample -> config_name="sample")
        if len(parts) == 2:
            group = None
            name = parts[-1]

        # Grouped configs (e.g., configs.dataset.cifar10 -> group="dataset", name="cifar10")
        elif len(parts) >= 3:
            # Allows infinite nesting: configs.network.attention.transformer -> group="network/attention"
            group = "/".join(parts[1:-1])
            name = parts[-1]

        # Check if the module explicitly defines __all__
        has_all = hasattr(module, "__all__")

        for obj_name, obj in inspect.getmembers(module, inspect.isclass):
            if not is_dataclass(obj):
                continue

            if has_all:
                # If __all__ is defined, only register objects explicitly listed in it
                if obj_name not in module.__all__:
                    continue
            else:
                # Fallback: ignore imported dataclasses if __all__ is missing
                if getattr(obj, "__module__", None) != module_name:
                    continue

            cs.store(group=group, name=name, node=obj)
