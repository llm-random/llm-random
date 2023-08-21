import argparse
import os
import random
from typing import Optional
import socket

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Tokenizer

from lizrd.core import misc
from lizrd.datasets.wikibookdata import get_processed_dataset
from lizrd.support.logging import get_current_logger, get_logger
from lizrd.train.train_utils import (
    get_model,
)
from research.conditional.utils.conditional_trainer import ConditionalTrainer
from research.conditional.utils.argparse import introduce_parser_arguments
from research.conditional.utils.misc_tools import set_seed
from research.conditional.utils.model_utils import (
    get_ff_layer,
    get_attention_layer,
    get_residual_layer,
)


def log_batch(dataset_wrapper):
    # In case of GPT, log an example sequence for a possible inspection

    print("Logging example batch...")
    batch = dataset_wrapper.get_batch()

    t = GPT2Tokenizer.from_pretrained(
        "gpt2", additional_special_tokens=["<sequence_sep>"]
    )
    num_to_log = 5
    for i in range(min(num_to_log, len(batch.tokens))):
        get_current_logger().report_text(
            title=f"example_sequence/seq{i}/input_text",
            value=t.decode(batch.tokens[i]),
            iteration=0,
        )
        get_current_logger().report_text(
            title=f"example_sequence/seq{i}/target_text",
            value=t.decode(batch.target_tokens[i]),
            iteration=0,
        )
    del batch, t
    print("Logged example batch.")


def main(
    rank: Optional[int],
    data_seeds: Optional[list[int]] = None,
    port: str = "29500",
    args: Optional[argparse.Namespace] = None,
    runner_params: Optional[list] = None,
):
    if runner_params is not None:
        parser = argparse.ArgumentParser()
        introduce_parser_arguments(parser)
        args, extra = parser.parse_known_args(runner_params)
        if len(extra):
            print("Unknown args:", extra)

    if rank is not None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port

        init_process_group("nccl", rank=rank, world_size=args.n_gpus)
        torch.cuda.set_device(rank)

    if args.deterministic_experiment:
        set_seed(args.torch_seed)
    # vocab size for gpt is 50257 + 1 for sequence_sep
    VOCAB_SIZE = 30522 if args.model_type == "bert" else 50258
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_distributed = True if rank is not None else False
    ff_layer_fun = get_ff_layer(args)
    attention_layer_fun = get_attention_layer(args)
    residual_fn = get_residual_layer(args)
    if args.model_parallelism_fragmentation is not None:
        args.model_parallelism_fragmentation = [
            int(s) for s in args.model_parallelism_fragmentation.split(",")
        ]
    model = get_model(
        max_length=args.cutoff,
        vocab_size=VOCAB_SIZE,
        ff_layer_fun=ff_layer_fun,
        attention_layer_fun=attention_layer_fun,
        dm=args.dmodel,
        n_blocks=args.n_blocks,
        device=DEVICE
        if rank is None
        else torch.device(
            "cpu"
        ),  # in case DDP is enabled, we want to keep model on CPU and move it to proper GPU later
        gradient_checkpointing=args.gradient_checkpointing,
        model_fragmentation=args.model_parallelism_fragmentation,
        residual_fn=residual_fn,
    )

    # make model data_distributed if necessary
    if rank is not None:
        print(f"Moving model to cuda:{rank}")
        model = model.to(f"cuda:{rank}")
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    train_dataloader = get_processed_dataset(
        max_total_length=args.cutoff,
        mask_percent=args.mask_percent,
        device=DEVICE,
        num_workers=args.num_workers,
        batch_size=args.batch_size // args.n_gpus
        if data_distributed
        else args.batch_size,
        seed=args.data_seed if data_seeds is None else data_seeds[rank],
        model_type=args.model_type,
        dataset_type=args.dataset_type,
    )

    logger = get_logger(args, model, VOCAB_SIZE) if rank is None or rank == 0 else None

    if args.model_type == "gpt" and (rank is None or rank == 0):
        log_batch(train_dataloader)

    trainer = ConditionalTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
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
        save_weights_path=args.save_weights_path,
        save_weights_interval=args.save_weights_interval,
        load_weights_path=args.load_weights_path,
        gradient_clipping=args.grad_clip,
        loss_checkpoint_chungs=args.loss_checkpoint_chungs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_decay=args.lr_decay,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_interval=args.lr_decay_interval,
        log_gradients_and_weights=args.log_gradients_and_weights,
        max_sequence_length=args.cutoff,
    )
    trainer.train(args.n_steps)

    if rank is not None:
        destroy_process_group()


if __name__ == "__main__":
    misc.print_available_gpus()
    parser = argparse.ArgumentParser()
    introduce_parser_arguments(parser)
    args = parser.parse_args()

    if args.data_distributed == False:
        main(None, args=args)
    else:
        random.seed(args.data_seed)
        data_seeds = [random.randint(0, 10000000) for _ in range(args.n_gpus)]

        # find free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = str(s.getsockname()[1])

        mp.spawn(
            main,
            args=[data_seeds, port],
            nprocs=args.n_gpus,
        )
