import re
import time
from contextlib import contextmanager
from typing import Union
from plotly import express as px

import torch

import torch.nn as nn
from lizrd.support.logging import get_current_logger
from functools import wraps


def get_registered_name(name):
    pattern = r"block_(\d+)"
    match = re.search(pattern, name)
    short_name = name.split("block.")[-1]
    if match:
        block_name = match.group()
        return f"{block_name}/{short_name}"
    return f"block_UNKNOWN/{name}"


class LayerManager:
    """
    This class is used to manage the feedforward layers of a model.
    It is used to log everything from weights and activations to gradients and your mum's phone number. [citation needed][this an unfiltered Codex suggestion I had to leave this in im sorry]
    """

    def __init__(
        self,
        model,
        logging_interval_light,
        logging_interval_heavy,
        steps_until_start_temperature_learn,
    ):
        self._layers = []
        self._logable_layers = []
        self._register_layers(model)
        self.logger = get_current_logger()
        self.logging_interval_light = logging_interval_light
        self.logging_interval_heavy = logging_interval_heavy
        self.steps_until_start_temperature_learn = steps_until_start_temperature_learn

    def _register_layers(self, model):
        """
        Iterates over all submodules and finds the ones that are of interest.
        Currently, those are only the feedforward and residual blocks.
        During model creation in LLM [llm.py], the feedforward layers are expected to be named "feedforward" and the residual layers "residual" (hardcoded in the repo as of 14.11.2023).
        """
        for name, layer in model.named_modules():
            suffix = name.split(".")[-1]
            registered_name = get_registered_name(name)
            if suffix in [
                "residual_feedforward",
                "residual_attention",
                "feedforward",
            ]:
                self._layers.append((registered_name, layer))
            if hasattr(layer, "log"):
                self._logable_layers.append((registered_name, layer))

    def prepare_for_logging(self, step):
        if (
            self.logging_interval_light > 0
            and step % self.logging_interval_light == 0
            or self.logging_interval_heavy > 0
            and step % self.logging_interval_heavy == 0
        ):
            for block_name, layer in self._logable_layers:
                if hasattr(layer, "prepare_for_logging"):
                    layer.prepare_for_logging()

    def log(self, step):
        verbosity_levels = []
        if self.logging_interval_heavy > 0 and step % self.logging_interval_heavy == 0:
            verbosity_levels = [2, 1, 0]
        elif (
            self.logging_interval_light > 0 and step % self.logging_interval_light == 0
        ):
            verbosity_levels = [1, 0]

        should_clean_up = len(verbosity_levels) > 0

        for verbosity_level in verbosity_levels:
            for block_name, layer in self._logable_layers:
                if isinstance(layer, LoggingLayer) or (
                    isinstance(layer, torch.distributed.fsdp.FullyShardedDataParallel)
                    and isinstance(layer._fsdp_wrapped_module, LoggingLayer)
                ):
                    info = layer.log(verbosity_level)
                    for name, data in info.items():
                        logging_name = block_name + "/" + name
                        self.logger.report_generic_info(
                            title=logging_name, iteration=step, data=data
                        )
        if should_clean_up:
            for _, layer in self._logable_layers:
                if isinstance(layer, LoggingLayer):
                    layer.clean_up_after_logging()

    def manage_learnable_temperature(self, step):
        is_learning_temperature = step >= self.steps_until_start_temperature_learn
        for block_name, layer in self._layers:
            for name, param in layer.named_parameters():
                if name in ["temperature_merge", "temperature_emit"]:
                    param.requires_grad = is_learning_temperature


class MeasuringLayer(nn.Module):
    def __init__(self, layer, name, parent):
        super().__init__()
        self.l = layer
        self.name = name
        self.parent = [parent]

    def forward(self, *args, **kwargs):
        with measure_time(self.parent[0], self.name):
            return self.l(*args, **kwargs)


def time_measured(name):
    def _decorator(func):
        @wraps(func)
        def _decorator_wrapper(self, *args, **kwargs):
            with measure_time(self, name):
                return func(self, *args, **kwargs)

        return _decorator_wrapper

    return _decorator


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

    def clean_up_after_logging(self):
        assert self.logging_switch
        self.logging_switch = False
        self.logging_cache = {}

    def prepare_for_logging(self):
        self.logging_switch = True

    def update_cache_for_logging(self, key, value):
        if self.logging_switch:
            if isinstance(value, dict):
                if key in self.logging_cache:
                    self.logging_cache[key].update(value)
                else:
                    self.logging_cache[key] = value
            elif isinstance(value, torch.Tensor):
                self.logging_cache[key] = value.clone().detach().cpu()
            elif isinstance(value, float) or isinstance(value, int):
                self.logging_cache[key] = value
            else:
                raise NotImplementedError

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
            return self.log_time()
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

    def log_time(self):
        log = {}
        if "time" in self.logging_cache:
            instr_names = list(self.logging_cache["time"].keys())
            instr_times = list(self.logging_cache["time"].values())
            times_fig = px.bar(x=instr_names, y=instr_times)
            log["time"] = times_fig
        return log

    def measure(self, module, name, exists=True):
        if not exists:
            return nn.Identity()
        return MeasuringLayer(module, name, self)


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
