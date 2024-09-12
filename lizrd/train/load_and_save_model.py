import pathlib
from typing import Union, Optional
import os

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

from lizrd.support.misc import generate_random_string


def get_checkpoint_from_path(load_weights_path: str, repeater_mode: bool) -> str:
    assert os.path.exists(load_weights_path), f"Path {load_weights_path} does not exist"
    if repeater_mode:
        load_weights_path = pathlib.Path(load_weights_path)

        if load_weights_path.is_dir():
            load_weights_path = load_weights_path/"repeater_save.pt" #dev highest number TODO

        if load_weights_path.exists():
            print(f"Repeater file ({load_weights_path}) does not exist, starting new training.")
            return None

    print(f"Loading checkpoint from {load_weights_path}...")
    checkpoint = torch.load(str(load_weights_path))
    print(f"Checkpoint loaded")
    return checkpoint


def load_model_weights(model: torch.nn.Module, checkpoint: dict[str, torch.Tensor]):
    print(f"Loading model weights...")
    model.load_state_dict(checkpoint["model"], strict=False)
    print(f"Loaded model weights")


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint: dict[str, torch.Tensor],
    model: Union[torch.nn.Module, FSDP],
    rank: int,
):
    print(f"Loading optimizer state...")
    if isinstance(model, FSDP):
        full_osd = None
        if rank == 0:
            full_osd = checkpoint["optimizer"]
        sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
        optimizer.load_state_dict(sharded_osd)
    else:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loaded optimizer state")


def load_scaler_state(
    scaler: torch.cuda.amp.GradScaler,
    checkpoint: dict[str, torch.Tensor],
):
    scaler.load_state_dict(checkpoint["scaler"])


def prepare_save_weights_path(path_to_dir: Optional[str], is_repeater:bool=False) -> Optional[str]:
    if path_to_dir is None:
        return None
    # we need a random dir because we can be running a whole grid from the same directory
    if not is_repeater:
        random_dirname = f"{generate_random_string(10)}"
        save_weights_path = os.path.join(path_to_dir, random_dirname)
    save_weights_path = os.path.abspath(save_weights_path)
    os.makedirs(save_weights_path, exist_ok=True)
    return save_weights_path


def save_checkpoint(
    model: Union[torch.nn.Module, FSDP],
    optimizer,
    scaler,
    path: str,
    rank: int,
    step: int,
):
    if rank == 0 or rank is None:
        print(f"Saving weights...")
    if isinstance(model, FSDP):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state_dict = model.state_dict()
        optimizer_state_dict = FSDP.full_optim_state_dict(model, optimizer)
    else:
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

    if rank == 0 or rank is None:
        full_path = os.path.join(path, f"{step}.pt")
        checkpoint = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "step": step,
        }
        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()

        torch.save(checkpoint, f=full_path)
        print(f"Weights saved to {full_path} (step {step})")
