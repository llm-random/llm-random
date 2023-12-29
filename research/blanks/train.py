import argparse
import os
import random
from typing import Callable, List, Optional
import socket

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from lizrd.core import misc
from lizrd.support.logging import get_current_logger, get_logger
from lizrd.support.misc import generate_random_string
from research.datasets import DataloaderWrapper
from .datasets import get_processed_dataset
from .model import (
    get_attention_layer,
    get_model,
    get_ff_layer,
)
from lizrd.text import tokenizers
from .tokenizers import BlankTokenizer

from lizrd.train.scheduler import get_scheduler
from .trainer import BlankTrainer
from .argparse import introduce_parser_arguments


def log_batch(
    wrapper: DataloaderWrapper,
    tokenizer_maker: Callable[[], tokenizers.AbstractTokenizer],
):
    # In case of GPT, log an example sequence for a possible inspection

    print("Logging example batch...")
    batch = wrapper.get_batch()
    hf_tokenizer = tokenizer_maker().tokenizer

    num_to_log = 5
    for i in range(min(num_to_log, len(batch.input_ids))):
        get_current_logger().report_text(
            title=f"example_sequence/seq{i}/input_text",
            value=hf_tokenizer.decode(batch.input_ids[i]),
            iteration=0,
        )
        get_current_logger().report_text(
            title=f"example_sequence/seq{i}/target_text",
            value=hf_tokenizer.decode(batch.target_ids[i]),
            iteration=0,
        )

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

    VOCAB_SIZE = BlankTokenizer.VOCAB_SIZE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BLANKS_IDS: List[int] = (
        BlankTokenizer.BLANK_IDS[: args.n_blanks]
        if args.blanks_use_multiple_embeddings
        else BlankTokenizer.BLANK_IDS[:1] * args.n_blanks
    )

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    data_distributed = True if rank is not None else False
    ff_layer_fun = get_ff_layer(args)
    attention_layer_fun = get_attention_layer(args)

    if args.model_parallelism_fragmentation is not None:
        args.model_parallelism_fragmentation = [
            int(s) for s in args.model_parallelism_fragmentation.split(",")
        ]
    if args.save_weights_path is not None:
        assert (
            "." not in args.save_weights_path
        ), f"Do not add .pt or .pth to save_weights_path! It is added automatically, along with step number."
        random_string = generate_random_string(10)
        args.save_weights_path = os.path.join(args.save_weights_path, random_string)
        args.save_weights_path = os.path.abspath(args.save_weights_path)
        os.makedirs(args.save_weights_path, exist_ok=True)

    sequence_length_with_blanks = args.cutoff + (
        args.n_blanks if args.blanks_extend_sequence_by_n_blanks else 0
    )

    model = get_model(
        max_length=sequence_length_with_blanks,
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
        residual_fn=None,
        init_type=args.init_type,
        init_scale=args.init_scale,
        n_blanks=args.n_blanks,
        blanks_use_multiple_embeddings=args.blanks_use_multiple_embeddings,
        blank_ids=torch.tensor(BLANKS_IDS, device=DEVICE, dtype=torch.int),
        blanks_add_embedding=args.blanks_add_embedding,
        blanks_residual=args.blanks_residual,
        blanks_learnable_weights=args.blanks_learnable_weights,
        blank_initial_weight=args.blank_initial_weight,
        blanks_straight_through=args.blanks_use_straight_through,
        blanks_use_custom_positional_embedding=args.blanks_use_custom_positional_embedding,
    )

    # make model data_distributed if necessary
    if rank is not None:
        print(f"Moving model to cuda:{rank}")
        model = model.to(f"cuda:{rank}")
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    scheduler = get_scheduler(args)

    common_dataloaders_kwargs = {
        "sequence_length": args.cutoff,
        "device": DEVICE,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size // args.n_gpus
        if data_distributed
        else args.batch_size,
        "seed": args.data_seed if data_seeds is None else data_seeds[rank],
        "dataset_type": args.dataset_type,
        "use_dummy_dataset": args.use_dummy_dataset,
        "tokenizer_maker": BlankTokenizer,
        "n_blanks": args.n_blanks,
        "blanks_ids": BLANKS_IDS,
        "use_only_last_blank_loss": args.blanks_use_only_last_blank_loss,
        "extend_sequence_by_n_blanks": args.blanks_extend_sequence_by_n_blanks,
    }
    train_dataloader = get_processed_dataset(
        **common_dataloaders_kwargs,
        dataset_split="train",
        dataset_path=args.train_dataset_path,
    )

    # TODO: replace eval_dataloader with something more sensible
    eval_dataloader = get_processed_dataset(
        **common_dataloaders_kwargs,
        dataset_split="eval"
        if args.dataset_type == "wikibook"
        else (
            "train"
            if args.dataset_type == "c4" and args.use_dummy_dataset
            else "validation"
        ),
        dataset_path=args.eval_dataset_path,
    )

    logger = get_logger(args, model, VOCAB_SIZE)

    # in case of data parallelism, only gpu:0 should log
    is_process_logging = True if rank is None or rank == 0 else False

    trainer = BlankTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        vocab_size=VOCAB_SIZE,
        mask_percent=args.mask_percent,
        mixed_precision=args.mixed_precision,
        logger=logger,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        lr_scheduler=scheduler,
        hack_name=args.hack_name,
        logging_interval_loss=args.logging_interval_loss,
        logging_interval_light=args.logging_interval_light,
        logging_interval_heavy=args.logging_interval_heavy,
        n_eval_steps=args.n_eval_steps,
        n_eval_batches=args.n_eval_batches,
        n_gpus=args.n_gpus,
        save_weights_path=args.save_weights_path,
        save_weights_interval=args.save_weights_interval,
        load_weights_path=args.load_weights_path,
        gradient_clipping=args.grad_clip,
        loss_checkpoint_chungs=args.loss_checkpoint_chungs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_gradients_and_weights=args.log_gradients_and_weights,
        max_sequence_length=args.cutoff,
        is_process_logging=is_process_logging,
        decoding_logging_steps=args.decoding_logging_steps,
        n_blanks=args.n_blanks,
        blanks_ids=BLANKS_IDS,
        use_only_last_blank_loss=args.blanks_use_only_last_blank_loss,
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
