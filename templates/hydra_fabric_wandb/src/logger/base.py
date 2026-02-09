import abc


class BaseLogger(abc.ABC):
    """Base class for logger."""

    def __init__(self):
        super(BaseLogger, self).__init__()

    @abc.abstractmethod
    def log(self):
        """Log the inputs."""
        pass
