"""
In order to take advantage of this file, you need first copy this file to your research project.
It is designed in the way you should just inherit over 'Runner' and 'BaseTrainer' classes, and add parser and then replace these classes in
this file with your own implementations.
"""

import argparse
import random
from typing import Optional

import torch

from lizrd.support.misc import set_seed
from core.add_arguments import add_default_parser_arguments
from core.builder import Builder
from core.training import BaseTrainer


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


def main(
    args: Optional[argparse.Namespace] = None,
    runner_params: Optional[list] = None,
):
    """
    args: is used to pass parsed arguments to the main function when we run this file as a script.
    runner_params: is used in the 'grid' to pass the arguments to the main function,
        so we run the experiment as a local backend (and in the same process).
    """
    args = handle_args(args, runner_params)
    set_seed(args.torch_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = Builder(args, device)
    trainer = BaseTrainer(
        **builder.get_train_artefacts(),
        dataset_type=args.dataset_type
    )
    trainer.train(args.n_steps)
    return trainer.metric_holder


if __name__ == "__main__":
    args = handle_args()
    _ = main(args=args)



