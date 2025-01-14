import pathlib
import re
from typing import Union, Optional
import os

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict

from lizrd.support.logging import AbstractLogger, NeptuneLogger
from lizrd.support.misc import generate_random_string
from research.conditional.utils.misc_tools import get_slurm_job_id


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


def get_checkpoint_from_path(load_weights_path: str) -> str:
    assert os.path.exists(load_weights_path), f"Path {load_weights_path} does not exist"

    # TODO: modify loading to make it work with multinode
    # as in this example: https://github.com/wz337/pytorch/blob/main/torch/distributed/checkpoint/examples/fsdp_checkpoint_example.py

    print(f"Loading checkpoint from {load_weights_path}...")
    if os.path.isdir(load_weights_path):
        checkpoint = torch.load(os.path.join(load_weights_path, "metadata.pt"))
    else:
        checkpoint = torch.load(load_weights_path)
    print(f"Checkpoint loaded")
    return checkpoint


def load_sharded_checkpoint(model, optimizer, checkpoint_path):
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model.state_dict(),
        }

        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(checkpoint_path),
        )
        model.load_state_dict(state_dict["model"])

        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optimizer",
            storage_reader=dist_cp.FileSystemReader(checkpoint_path),
        )

        flattened_osd = FSDP.optim_state_dict_to_load(
            model, optimizer, optim_state["optimizer"]
        )
        optimizer.load_state_dict(flattened_osd)


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


def prepare_save_weights_path(path_to_dir: Optional[str]) -> Optional[str]:
    # we need a random dir because we can be running a whole grid from the same directory
    slurm_job_id = get_slurm_job_id()
    if slurm_job_id:
        random_dirname = slurm_job_id
    else:
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
    global_rank: int,
    step: int,
    cutoff,
    loggers: list[AbstractLogger],
    batch_size: int = 1,
    args_override: Optional[dict] = None,
    save_sharded: bool = False,
):
    if isinstance(model, FSDP):
        # for some reason, setting the model to training mode and
        # running a forward pass is necessary to be able to save it
        # in FSDP. God help us.
        model.train()
        with torch.no_grad():
            _ = model(torch.zeros((batch_size, cutoff), dtype=torch.int))

    is_saving_process = global_rank == 0 or global_rank is None

    if is_saving_process:
        neptune_loggers = [
            l
            for l in loggers
            if isinstance(l, NeptuneLogger)  # dev TODO do it for other loggers
        ]
        if len(neptune_loggers) >= 1:
            ids = [nl.instance_logger._sys_id for nl in neptune_loggers]
            logger_metadata = {"run_id": ids}
        else:
            logger_metadata = {"run_id": None}

        print(f"Saving weights...")
        metadata = {
            "step": step,
            "logger": logger_metadata,
            "args_override": args_override,
        }
        if scaler is not None:
            metadata["scaler"] = scaler.state_dict()
    else:
        metadata = None

    if save_sharded:
        full_path = os.path.join(path, str(step))
        state_dict = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
        }

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(full_path),
        )
        if is_saving_process:
            torch.save(metadata, os.path.join(full_path, "metadata.pt"))

    else:
        full_path = os.path.join(path, f"{step}.pt")

        if isinstance(model, FSDP):
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                model_state_dict = model.state_dict()
            optimizer_state_dict = FSDP.full_optim_state_dict(model, optimizer)
        else:
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()

        if is_saving_process:
            checkpoint = {
                "model": model_state_dict,
                "optimizer": optimizer_state_dict,
                **metadata,
            }  # dev TODO add accumulated training variables for proper logging, f.e. loss_interval/100 - loss accumulated over 100 training steps

            torch.save(checkpoint, f=full_path)

    if is_saving_process:
        for logger in neptune_loggers:
            logger.report_text(
                title=f"job/saved_checkpoint",
                value=str(full_path),
                iteration=step,
            )
        print(f"Weights saved to {full_path} (step {step})")
        return os.path.abspath(full_path)
