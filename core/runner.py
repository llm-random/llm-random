import argparse
import os
import random
import socket
from typing import Optional
import torch.multiprocessing as mp
import torch
from lizrd.support.misc import set_seed
from core.add_arguments import add_default_parser_arguments
from core.builder import Builder
from core.training import BaseTrainer
from torch.distributed import init_process_group, destroy_process_group

def handle_args(args=None, runner_params=None):
    parser = argparse.ArgumentParser()
    add_default_parser_arguments(parser)
    if runner_params is not None:
        args, extra = parser.parse_known_args(runner_params)
        if len(extra):
            print("Unknown args:", extra)
    elif args is None:
            args = parser.parse_args()

    if args.data_seed < 0:
        args.data_seed = random.randint(0, 10000000)
    return args


def init_distributed(rank, port, world_size):
    if rank is not None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def runner(
    rank: Optional[int],
    data_seeds: Optional[list[int]] = None,
    port: str = "29500",
    args: Optional[argparse.Namespace] = None,
    runner_params: Optional[list] = None,
    device: str = None
):
    """
    args: is used to pass parsed arguments to the main function when we run this file as a script.
    runner_params: is used in the 'grid' to pass the arguments to the main function,
        so we run the experiment as a local backend (and in the same process).
    """
    args = handle_args(args, runner_params)
    init_distributed(rank, port, args.n_gpus)
    set_seed(args.torch_seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = Builder(args, device, data_seeds, rank)
    trainer = BaseTrainer(
        **builder.get_train_artefacts(),
        dataset_type=args.dataset_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    trainer.train(args.n_steps)
    if rank is not None:
        destroy_process_group()
    return trainer.metric_holder