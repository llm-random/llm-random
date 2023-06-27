import argparse
import os
import random
from typing import Optional
import socket

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from lizrd.core import misc
from lizrd.support.logging import get_logger
from lizrd.train.train_utils import (
    get_model,
    get_processed_dataset,
)
from research.conditional.utils.conditional_trainer import ConditionalTrainer
from research.conditional.utils.misc_utils import introduce_parser_arguments
from research.conditional.utils.model_utils import get_ff_layer, get_attention_layer

parser = argparse.ArgumentParser()
introduce_parser_arguments(parser)
args = parser.parse_args()


def main(rank: Optional[int], data_seeds: Optional[list[int]] = None):
    if rank is not None:
        os.environ["MASTER_ADDR"] = "localhost"

        # find free port and assign it
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("",0))
        port = s.getsockname()[1]
        os.environ["MASTER_PORT"] = port

        init_process_group("nccl", rank=rank, world_size=args.n_gpus)
        torch.cuda.set_device(rank)

    VOCAB_SIZE = 30522 if args.model_type == "bert" else 50257
    DEVICE = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    distributed = True if rank is not None else False
    train_dataloader = get_processed_dataset(
        max_total_length=args.cutoff,
        mask_percent=args.mask_percent,
        device=DEVICE,
        num_workers=args.num_workers,
        batch_size=args.batch_size // args.n_gpus if distributed else args.batch_size,
        seed=args.data_seed if data_seeds is None else data_seeds[rank],
        model_type=args.model_type,
        distributed=distributed,
    )

    ff_layer_fun = get_ff_layer(args)
    attention_layer_fun = get_attention_layer(args)

    model = get_model(
        max_length=args.cutoff,
        vocab_size=VOCAB_SIZE,
        ff_layer_fun=ff_layer_fun,
        attention_layer_fun=attention_layer_fun,
        dm=args.dmodel,
        n_blocks=args.n_blocks,
        device=DEVICE,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # make model distributed if necessary
    if rank is not None:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    logger = get_logger(args, model, VOCAB_SIZE) if rank is None or rank == 0 else None

    trainer = ConditionalTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        batch_size=args.batch_size,
        vocab_size=VOCAB_SIZE,
        mask_percent=args.mask_percent,
        mixed_precision=args.mixed_precision,
        logger=logger,
        hack_name=args.hack_name,
        model_type=args.model_type,
        logging_interval_loss=args.logging_interval_loss,
        logging_interval_light=args.logging_interval_light,
        logging_interval_heavy=args.logging_interval_heavy,
        n_gpus=args.n_gpus,
    )
    trainer.train(args.n_steps)

    if rank is not None:
        destroy_process_group()

if __name__ == "__main__":
    misc.print_available_gpus()
    if args.n_gpus == 1:
        main(None)
    else:
        random.seed(args.data_seed)
        data_seeds = [random.randint(0, 10000000) for _ in range(args.n_gpus)]
        mp.spawn(
            main,
            args=[
                data_seeds,
            ],
            nprocs=args.n_gpus,
        )
