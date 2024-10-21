import pathlib
import re
from typing import Union, Optional
import os

from neptune import Run
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

from lizrd.support.logging import JointLogger, NeptuneLogger
from lizrd.support.misc import generate_random_string


def get_latest_checkpoint(dir_path) -> pathlib.Path:
    """Returns the latest checkpoint by the heighest number in its name that depicts training step number"""
    dir_path = pathlib.Path(dir_path)
    all_checkpoints_paths = dir_path.glob(r"[0-9]*.pt")
    latest_checkpoint = None
    latest_checkpoint_step = 0
    for checkpoint_path in all_checkpoints_paths:
        c_step = int(re.findall(r"\d+", checkpoint_path.name)[0])
        if c_step >= latest_checkpoint_step:
            latest_checkpoint = checkpoint_path
            latest_checkpoint_step = c_step
    return latest_checkpoint


def get_checkpoint_from_path(load_weights_path: str, repeater_mode: bool) -> str:
    assert os.path.exists(load_weights_path), f"Path {load_weights_path} does not exist"
    if repeater_mode:
        load_weights_path = pathlib.Path(load_weights_path)

        if load_weights_path.is_dir():
            latest_model = get_latest_checkpoint(load_weights_path)
            if not latest_model:
                print(
                    f"No model yet saved in ({load_weights_path}), starting new training."
                )
                return None
            load_weights_path = load_weights_path / latest_model.name

    print(f"Loading checkpoint from {load_weights_path}...")
    checkpoint = torch.load(load_weights_path)
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


def prepare_save_weights_path(
    path_to_dir: Optional[str], is_repeater: bool = False
) -> Optional[str]:
    if path_to_dir is None:
        if is_repeater:
            raise Exception(
                "Please specify checkpoint directory when using repeater mode"
            )
        return None
    # we need a random dir because we can be running a whole grid from the same directory
    if not is_repeater:
        random_dirname = f"{generate_random_string(10)}"
        path_to_dir = os.path.join(path_to_dir, random_dirname)
    save_weights_path = os.path.abspath(path_to_dir)
    os.makedirs(save_weights_path, exist_ok=True)
    return save_weights_path


def save_checkpoint(
    model: Union[torch.nn.Module, FSDP],
    optimizer,
    scaler,
    path: str,
    rank: int,
    step: int,
    batch_size,
    cutoff,
    joint_loggers: Optional[JointLogger] = None,
):
    if isinstance(model, FSDP):
        # for some reason, setting the model to training mode and
        # running a forward pass is necessary to be able to save it
        # in FSDP. God help us.
        model.train()
        with torch.no_grad():
            _ = model(torch.zeros((batch_size, cutoff), dtype=torch.int))
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
        neptune_logger: Run = [
            l
            for l in joint_loggers.loggers
            if isinstance(l, NeptuneLogger)  # dev TODO do it for other loggers
        ]
        if len(neptune_logger) == 1:
            neptune_logger = neptune_logger[0].instance_logger
            logger_metadata = {"run_id": neptune_logger._sys_id}
        else:
            print(f"No Neptune logger, no saving.")
            logger_metadata = None

        full_path = os.path.join(path, f"{step}.pt")
        checkpoint = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "step": step,
            "logger": logger_metadata,
        }
        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()

        torch.save(checkpoint, f=full_path)
        print(f"Weights saved to {full_path} (step {step})")
