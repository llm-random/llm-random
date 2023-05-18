import re
from typing import Dict, Union

import plotly.graph_objs as go

from lizrd.core import nn
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

    def prepare_for_logging(self):
        for block_name, layer in self._layers:
            layer.prepare_for_logging()

    def log(self, step, verbosity_level):
        infos = []
        for block_name, layer in self._layers:
            info: Dict[str, Union[float, go.Figure]] = layer.log(verbosity_level)
            infos.append(info)
        for name, data in infos:
            self.logger.report_generic_info(title=name, iteration=step, data=data)


class LoggingLayer(nn.Module):
    def __init__(self):
        super(LoggingLayer, self).__init__()
        self.logging_switch = False
        self.cached_data = {}

    def report_stats(self):
        assert self.logging_switch
        self.logging_switch = False
        data = self.cached_data
        self.cached_data = {}
        return data

    def prepare_for_logging(self):
        self.logging_switch = True

    def cache(self, key, value):
        if self.logging_switch:
            self.cached_data[key] = value.detach().cpu()

    def log(self, verbosity_level):
        if verbosity_level == 1:
            return self.log_light()
        elif verbosity_level == 2:
            return self.log_light() + self.heavy()

    def log_light(self):
        return []

    def log_heavy(self):
        return []
