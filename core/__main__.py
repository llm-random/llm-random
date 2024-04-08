"""
In order to take advantage of this file, you need first copy this file to your research project.
It is designed in the way you should just inherit over 'Runner' and 'BaseTrainer' classes, and add parser and then replace these classes in
this file with your own implementations.
"""

import argparse
import os
import random
import socket
from typing import Optional
import torch.multiprocessing as mp
import torch

from core.runner import handle_args, runner
from lizrd.support.misc import set_seed
from core.add_arguments import add_default_parser_arguments
from core.builder import Builder
from core.training import BaseTrainer
from torch.distributed import init_process_group, destroy_process_group


def find_free_port(address: str = "") -> str:
    """Helper function to find a free port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((address, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
    return str(port)


if __name__ == "__main__":
    args = handle_args()
    if args.fsdp_enabled:
        random.seed(args.data_seed)
        data_seeds = [random.randint(0, 10000000) for _ in range(args.n_gpus)]

        port = find_free_port()
        mp.spawn(
            runner,
            args=[data_seeds, port, args],
            nprocs=args.n_gpus,
        )
    else:
        _ = runner(None, args=args)




