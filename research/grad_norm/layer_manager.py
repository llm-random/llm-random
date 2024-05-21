import re

import torch

from lizrd.core.misc import LoggingLayer
from lizrd.support.logging import get_current_logger


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
        elif self.logging_interval_light > 0 and step % self.logging_interval_light == 0:
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
                        self.logger.report_generic_info(title=logging_name, iteration=step, data=data)
        if should_clean_up:
            for _, layer in self._logable_layers:
                if isinstance(layer, LoggingLayer):
                    layer.clean_up_after_logging()
