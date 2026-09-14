import logging
from .base import BaseLogger

log = logging.getLogger(__name__)


class DummyLogger(BaseLogger):
    def __init__(self, **kwargs):
        # Accept and ignore constructor kwargs (e.g. root_config) so any logger
        # can be instantiated uniformly by the experiment
        pass

    def log(self, *args, **kwargs):
        log.info(
            f"DummyLogger recording logs with args of length: {len(args)}, kwargs of length: {len(kwargs)}"
        )
