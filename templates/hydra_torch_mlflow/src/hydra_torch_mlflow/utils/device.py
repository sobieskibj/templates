import logging

import torch

log = logging.getLogger(__name__)


def get_devices() -> tuple[torch.device, torch.device]:
    """Return the (cpu, accelerator) devices.

    Falls back to the CPU device when no accelerator is available.
    """
    device_cpu = torch.cpu.current_device()
    device_gpu = None
    if torch.cuda.is_available():
        device_gpu = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device_gpu = torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device_gpu = torch.device("xpu")
    if device_gpu is None:
        log.warning("No GPU device detected, using CPU!")
        device_gpu = device_cpu
    return device_cpu, device_gpu
