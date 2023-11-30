import argparse
import os
import random
from typing import Callable, Optional
import socket

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

from lizrd.core import misc
from lizrd.support.logging import get_current_logger, get_logger
from lizrd.support.misc import generate_random_string, set_seed
from lizrd.train.train_utils import (
    get_model,
)
from lizrd.text import tokenizers
from research.conditional.utils.check_args import check_args
from research.datasets import DataloaderWrapper, get_processed_dataset
from lizrd.train.scheduler import get_scheduler
from research.conditional.utils.conditional_trainer import ConditionalTrainer
from research.conditional.utils.argparse import introduce_parser_arguments
from research.conditional.utils.model_utils import (
    get_ff_layer,
    get_attention_layer,
    get_mixed_precision_ignored_classes,
    get_residual_layer,
    get_classes_from_module_names,
)


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
    """
    rank: int - the ID of the current process (usually also the GPU ID). Only relevant for multi-GPU training.
    """
    if runner_params is not None:
        parser = argparse.ArgumentParser()
        introduce_parser_arguments(parser)
        args, extra = parser.parse_known_args(runner_params)
        if len(extra):
            print("Unknown args:", extra)

    check_args(args)
    if args.save_weights_path is not None:
        filename = args.save_weights_path.split("/")[-1]
        assert (
            "." not in filename
        ), "Do not add filename extensions (e.g. .pt or .pth) to save_weights_path! It is added automatically, along with step number."
        random_string = generate_random_string(10)
        args.save_weights_path = os.path.join(args.save_weights_path, random_string)
        args.save_weights_path = os.path.abspath(args.save_weights_path)
        os.makedirs(args.save_weights_path, exist_ok=True)

    if rank is not None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port

        init_process_group("nccl", rank=rank, world_size=args.n_gpus)
        torch.cuda.set_device(rank)

    if args.deterministic_experiment:
        set_seed(args.torch_seed)

    VOCAB_SIZE = (
        tokenizers.BertTokenizer.VOCAB_SIZE
        if args.model_type == "bert"
        else tokenizers.GPTTokenizer.VOCAB_SIZE
    )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if args.model_parallelism_fragmentation is not None:
        args.model_parallelism_fragmentation = [
            int(s) for s in args.model_parallelism_fragmentation.split(",")
        ]

    if args.mixed_precision_dtype == "float16":
        args.mixed_precision_dtype = torch.float16
    elif args.mixed_precision_dtype == "bfloat16":
        args.mixed_precision_dtype = torch.bfloat16

    if args.fsdp_enabled:
        fsdp_param_precision = args.mixed_precision_dtype
        fsdp_mixed_precision_ignore_classes = get_mixed_precision_ignored_classes(args)
        fsdp_modules_to_wrap = get_classes_from_module_names(args.fsdp_modules_to_wrap)
    else:
        fsdp_param_precision = None
        fsdp_mixed_precision_ignore_classes = None
        fsdp_modules_to_wrap = None

    # in case of data parallelism (DDP/FSDP), only gpu:0 should log
    is_logging_process = True if rank is None or rank == 0 else False

    activation_checkpointing_modules = get_classes_from_module_names(
        args.activation_checkpointing_modules
    )

    ff_layer_fun = get_ff_layer(args)
    attention_layer_fun = get_attention_layer(args)
    residual_fn = get_residual_layer(args)

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
        ),  # in case of  DDP/FSDP, we initialize the model on CPU and move it to the GPU later
        init_type=args.init_type,
        init_scale=args.init_scale,
        ddp_enabled=args.ddp_enabled,
        fsdp_enabled=args.fsdp_enabled,
        fsdp_param_precision=fsdp_param_precision,
        fsdp_mixed_precision_ignore_classes=fsdp_mixed_precision_ignore_classes,
        fsdp_offload_params=args.fsdp_offload_params,
        fsdp_min_num_params=args.fsdp_min_num_params,
        fsdp_modules_to_wrap=fsdp_modules_to_wrap,
        activation_checkpointing_modules=activation_checkpointing_modules,
        model_fragmentation=args.model_parallelism_fragmentation,
        residual_fn=residual_fn,
        is_logging_process=is_logging_process,
        rank=rank,
    )

    if args.torch_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    scheduler = get_scheduler(args)

    data_distributed = args.ddp_enabled or args.fsdp_enabled
    batch_size = args.batch_size // args.n_gpus if data_distributed else args.batch_size

    common_dataloaders_kwargs = {
        "sequence_length": args.cutoff,
        "device": DEVICE,
        "num_workers": args.num_workers,
        "batch_size": batch_size,
        "seed": args.data_seed if data_seeds is None else data_seeds[rank],
        "model_type": args.model_type,
        "dataset_type": args.dataset_type,
        "use_dummy_dataset": args.use_dummy_dataset,
    }

    train_dataloader = get_processed_dataset(
        **common_dataloaders_kwargs, dataset_split="train"
    )

    eval_dataloader = get_processed_dataset(
        **common_dataloaders_kwargs,
        dataset_split=("eval" if args.dataset_type == "wikibook" else "validation"),
    )

    if is_logging_process:
        logger = get_logger(args, model, VOCAB_SIZE)
    else:
        logger = None

    if args.model_type == "gpt" and is_logging_process:
        log_batch(
            train_dataloader,
            tokenizer_maker=tokenizers.GPTTokenizer
            if args.model_type == "gpt"
            else tokenizers.BertTokenizer,
        )

    trainer = ConditionalTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        vocab_size=VOCAB_SIZE,
        mask_percent=args.mask_percent,
        mixed_precision=False if args.fsdp_enabled else args.mixed_precision,
        mixed_precision_dtype=args.mixed_precision_dtype,
        logger=logger,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        lr_scheduler=scheduler,
        model_type=args.model_type,
        logging_interval_loss=args.logging_interval_loss,
        logging_interval_light=args.logging_interval_light,
        logging_interval_heavy=args.logging_interval_heavy,
        eval_interval=args.eval_interval,
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
        is_logging_process=is_logging_process,
        eval_dynamic_groupsize=args.eval_dynamic_groupsize,
        eval_discrete_mot=args.eval_discrete_mot,
        decoding_interval=args.decoding_interval,
        eval_min_group_size_logfactor=args.eval_min_group_size_logfactor,
        eval_max_group_size_logfactor=args.eval_max_group_size_logfactor,
        steps_until_start_temperature_learn=args.steps_until_start_temperature_learn,
    )
    trainer.train(args.n_steps)

    if rank is not None:
        destroy_process_group()


if __name__ == "__main__":
    misc.print_available_gpus()
    parser = argparse.ArgumentParser()
    introduce_parser_arguments(parser)
    args = parser.parse_args()

    if args.ddp_enabled or args.fsdp_enabled:
        random.seed(args.data_seed)
        data_seeds = [random.randint(0, 10000000) for _ in range(args.n_gpus)]

        # find free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = str(s.getsockname()[1])
        mp.spawn(
            main,
            args=[data_seeds, port, args],
            nprocs=args.n_gpus,
        )
    else:
        main(None, args=args)
