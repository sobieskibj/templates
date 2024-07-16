from torch import nn

import logging
log = logging.getLogger(__name__)

class TestNetwork(nn.Module):

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def action(self):
        log.info(self.text)