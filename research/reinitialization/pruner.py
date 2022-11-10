import copy
from typing import TYPE_CHECKING, OrderedDict

from logging import getLogger
from torch import nn
from torch import Tensor
import torch

# to avoid cycle import while using hints
if TYPE_CHECKING:
    from research.reinitialization.linears import RandomPruneLayer


class Pruner:
    def __init__(self, n_steps_prune: int, prob: float):
        self.n_steps_prune = n_steps_prune
        self.prob = prob
        self.current_step = 0
        self.layers = []

    def register(self, layer: "RandomPruneLayer"):
        self.layers.append(layer)

    def step(self):
        if self.current_step % self.n_steps_prune == 0:
            print("Pruning step")
            for layer in self.layers:
                layer.prune(self.prob)
        self.current_step += 1

def are_state_dicts_the_same(model_state_dict_1, model_state_dict_2):
    logger = getLogger()
    import torch
    # if br:
    #     breakpoint()

    if len(model_state_dict_1) != len(model_state_dict_2):
        logger.info(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            logger.info(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            logger.info(f"Tensor mismatch: {v_1} vs {v_2}")
            return False
    return True

def generate_random_string(length: int) -> str:
    import random
    import string

    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))

class LTHPruner:
    def __init__(self):
        self.layers = []

    def register_model(self, model: nn.Module):
        # self.initial_state_dict = copy.deepcopy(model.state_dict())
        self.model_path = f'/tmp/{generate_random_string(10)}.pt'
        torch.save(model.state_dict(), self.model_path)
        self.model = model
        # self.device = model.de

    def register(self, layer):
        self.layers.append(layer)

    def prune(self, percentage: float):
        for layer in self.layers:
            layer.prune(percentage)

    def reinitialize(self):
        masks = copy.deepcopy([layer.mask for layer in self.layers])
        model_state_dict = torch.load(self.model_path)
        assert not are_state_dicts_the_same(self.model.state_dict(), model_state_dict)
        self.model.load_state_dict(model_state_dict)
        assert are_state_dicts_the_same(self.model.state_dict(), model_state_dict)
        for layer, mask in zip(self.layers, masks):
            layer.mask = mask