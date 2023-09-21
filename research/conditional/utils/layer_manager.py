import re
import time
from contextlib import contextmanager
from typing import Union

import torch

from lizrd.core import nn
from lizrd.support.logging import get_current_logger


class LayerManager:
    """
    This class is used to manage the feedforward layers of a model.
    It is used to log everything from weights and activations to gradients and your mum's phone number. [citation needed][this an unfiltered Codex suggestion I had to leave this in im sorry]
    """

    def __init__(self, model, logging_interval_light, logging_interval_heavy):
        self._layers = []
        self._register_layers(model)
        self.logger = get_current_logger()
        self.logging_interval_light = logging_interval_light
        self.logging_interval_heavy = logging_interval_heavy

    def _register_layers(self, model):
        for name, layer in model.named_modules():
            if name.endswith("feedforward"):
                pattern = r"block_(\d+)"
                match = re.search(pattern, name)
                if match:
                    block_name = match.group()
                else:
                    raise Exception(
                        f"The expected pattern {pattern} was not found in name: {name}. The naming convention of model layers is not as expected."
                    )
                self._layers.append((block_name, layer))

    def prepare_for_logging(self, step):
        if (
            step % self.logging_interval_light == 0
            or step % self.logging_interval_heavy == 0
        ):
            for block_name, layer in self._layers:
                if hasattr(layer, "prepare_for_logging"):
                    layer.prepare_for_logging()

    def log(self, step):
        if step == 0:
            return
        verbosity_levels = []
        for level, freq in enumerate(
            [self.logging_interval_light, self.logging_interval_heavy], start=1
        ):
            if step % freq == 0:
                verbosity_levels.append(level)
        for verbosity_level in verbosity_levels:
            for block_name, layer in self._layers:
                if hasattr(layer, "log"):
                    info = layer.log(verbosity_level)
                    for name, data in info.items():
                        logging_name = block_name + "/" + name
                        self.logger.report_generic_info(
                            title=logging_name, iteration=step, data=data
                        )


class LoggingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # info about position in model
        self.layer_type: Union[str, None] = None
        self.block_number: Union[int, None] = None

        # whether to log
        self.logging_switch = False

        # caches for logging and propagation
        self.logging_cache = {}
        self.forward_pass_cache: Union[dict, None] = None

    def report_stats(self):
        assert self.logging_switch
        self.logging_switch = False
        data = self.logging_cache
        self.logging_cache = {}
        return data

    def prepare_for_logging(self):
        self.logging_switch = True

    def update_cache_for_logging(self, key, value):
        if self.logging_switch:
            if type(value) == dict:
                if key in self.logging_cache:
                    self.logging_cache[key].update(value)
                else:
                    self.logging_cache[key] = value
            else:
                self.logging_cache[key] = value.clone().detach().cpu()

    def _combine_to_dict_key(self, key, layer_type, block_number):
        return f"block_{block_number}_{layer_type}_{key}"

    def update_forward_pass_cache(self, key, value):
        combined_key = self._combine_to_dict_key(
            key, self.layer_type, self.block_number
        )
        self.forward_pass_cache[combined_key] = value

    def get_from_forward_pass_cache(self, key, block_number, layer_type):
        combined_key = self._combine_to_dict_key(key, layer_type, block_number)
        return self.forward_pass_cache[combined_key]

    def log(self, verbosity_level):
        if verbosity_level == 0:
            return []
        elif verbosity_level == 1:
            return self.log_light()
        elif verbosity_level == 2:
            return self.log_heavy()
        else:
            raise Exception("Invalid verbosity level")

    def log_light(self):
        return {}

    def log_heavy(self):
        return {}


@contextmanager
def measure_time(layer: LoggingLayer, instruction_name: str):
    """
    This simple context manager is used to measure the time of a block of code.
    Args:
        layer: The LoggingLayer object that will be used to cache the time.
        instruction_name: The name of the instruction that is being measured.
    """
    if layer.logging_switch:
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.time()
    yield
    if layer.logging_switch:
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            layer.update_cache_for_logging(
                "time", {instruction_name: start.elapsed_time(end)}
            )
        else:
            end = time.time()
            layer.update_cache_for_logging("time", {instruction_name: end - start})
