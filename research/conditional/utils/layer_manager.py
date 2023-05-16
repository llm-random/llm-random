import re
from typing import List

import torch

from lizrd.support.logging import get_current_logger


class LayerManager:
    """
    This class is used to manage the feedforward layers of a model.
    It is used to log everything from weights and activations to gradients and your mum's phone number. [citation needed][this an unfiltered Codex suggestion I had to leave this in im sorry]
    """

    def __init__(self, model):
        self.model = model
        self._layers = []
        self._register_layers()
        self.logger = get_current_logger()

    def _register_layers(self):
        for name, layer in self.model.named_modules():
            if name.endswith("feedforward"):
                pattern = r"block_(\d+)"
                match = re.search(pattern, name)
                if match:
                    block_name = match.group()
                else:
                    raise Exception(
                        f"The expected pattern {pattern} was not found in name: {name}. The naming convention for layers is not as expected."
                    )
                self._layers.append((block_name, layer))

    def log(self, step, verbosity_level):
        infos = []
        for block_name, layer in self._layers:
            info: List[str, torch.Tensor, dict] = layer.log(step, verbosity_level)
            infos.append(info)
        for name, data, additional_info in infos:
            type = additional_info.pop("type")
            self.logger.report_generic_info(
                type=type, title=name, iteration=step, **additional_info
            )
