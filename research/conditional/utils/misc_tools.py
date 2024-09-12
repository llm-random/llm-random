from contextlib import contextmanager
from datetime import datetime
import subprocess
import os

import torch


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


def entropy(x, dim):
    ent = -torch.sum(x * torch.log(x + 1e-8), dim=dim)
    return ent


# context manager
@contextmanager
def temp_modify_attr(
    layers: list[torch.nn.Module], attribute_name, new_attribute_value
):
    """
    modify the attribute of a list of layers to a new value, and then restore the original value
    """
    original_attribute_values = []
    for layer in layers:
        original_attribute_values.append(getattr(layer, attribute_name))
        setattr(layer, attribute_name, new_attribute_value)
    yield
    for layer, original_attribute_value in zip(layers, original_attribute_values):
        setattr(layer, attribute_name, original_attribute_value)

def get_termination_timestamp_slurm() -> int:
    # try:
    #     print("1+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")#dev
    #     env=os.environ.copy()
    #     print(env)#dev
    #     # result = subprocess.run(
    #     #     ['scontrol', 'show', 'job', env['SLURM_JOBID']],
    #     #     capture_output=True, text=True, check=True, 
    #     # )
    #     #dev squeue -h -j $SLURM_JOB_ID -O EndTime
    #     result = subprocess.run(
    #         ['sacct', '-h', '-j', env['SLURM_JOBID'], '-o', 'EndTime'],
    #         capture_output=True, text=True, check=True, 
    #     )
    #     print("2+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")#dev
    #     print(result)#dev
    # except Exception as e:
    #     print(f"Error executing scontrol: {e}")
    #     raise e

    # stdout = result.stdout
    # print(stdout)#dev
    # param_k = "EndTime="
    # for param in [e for e in stdout.replace('\n', ' ').split(' ') if e!='']:
    #     if param_k in param:
    #         val = param.replace(param_k, '')
    #         date_format = '%Y-%m-%dT%H:%M:%S'
    #         date_obj = datetime.strptime(val, date_format)
    env=os.environ.copy()
    return int(env["SLURM_JOB_START_TIME"])
