import random
import numpy as np
import torch
from attr import define


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.


def get_ith_chunk(tensor, chunks, i):
    list_of_chunks = torch.chunk(tensor, chunks, dim=0)
    return list_of_chunks[i]


def generate_shuffler_unshuffler(batch_size, seqlen, mix_whole_batch=False):
    if mix_whole_batch:
        shuffled_indices = torch.randperm(batch_size * seqlen)
    else:
        shuffled_indices = torch.randperm(seqlen)
    unshuffled_indices = shuffled_indices.argsort()

    if mix_whole_batch:

        def shuffle_tensor(tensor):
            reshape = tensor.reshape(-1, *tensor.shape[2:])
            shuffled_tensor = reshape[shuffled_indices]
            return shuffled_tensor.reshape(tensor.shape)

        def unshuffle_tensor(shuffled_tensor):
            reshape = shuffled_tensor.reshape(-1, *shuffled_tensor.shape[2:])
            return reshape[unshuffled_indices].reshape(shuffled_tensor.shape)

    else:

        def shuffle_tensor(tensor):
            return tensor[:, shuffled_indices, :]

        def unshuffle_tensor(shuffled_tensor):
            return shuffled_tensor[:, unshuffled_indices, :]

    return shuffle_tensor, unshuffle_tensor


def stable_softmax_temperature(x, temperature, dim=-1):
    x = x / temperature
    x = x - x.max(dim=dim, keepdim=True)[0]
    x = torch.exp(x)
    x = x / x.sum(dim=dim, keepdim=True)
    return x


def entropy(x):
    ent = -torch.sum(x * torch.log(x + 1e-8), dim=-1)
    return ent


@define
class TemperatureScheduler:
    model: torch.nn.Module
    steps_until_anneal: int
    total_steps: int
    previous_multiplier: float = 1.0
    current_multiplier: float = 1.0

    def step(self, current_step):
        if current_step > self.steps_until_anneal:
            # Anneal linearly from step N to end
            fraction = 1 - (
                (current_step - self.steps_until_anneal)
                / (self.total_steps - self.steps_until_anneal)
            )
            self.previous_multiplier = self.current_multiplier
            self.current_multiplier = fraction
            self.update_model_temperature()

    def update_model_temperature(self):
        for module in self.model.modules():
            if hasattr(module, "temperature"):
                # contmoe
                module.temperature *= self.current_multiplier / self.previous_multiplier
            elif hasattr(module, "temperature_merge"):
                # learnable temperature
                module.temperature_merge *= (
                    self.current_multiplier / self.previous_multiplier
                )
                module.temperature_emit *= (
                    self.current_multiplier / self.previous_multiplier
                )
