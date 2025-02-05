import argparse
import os
import random
from typing import Callable, Optional
import socket

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

from lizrd.core import misc
from lizrd.core.llm import EmbeddingLayer, Parallel
from lizrd.support.logging import get_current_logger, get_logger
from lizrd.support.misc import (
    get_n_learnable_parameters,
    set_seed,
)
from lizrd.text import tokenizers
from research.datasets import DataloaderWrapper, get_processed_dataset
from lizrd.train.scheduler import get_scheduler
from research.token_reduction.trainer import Trainer
from research.token_reduction.utils.argparse import (
    introduce_parser_arguments,
    check_args,
)
from research.token_reduction.build import (
    disable_profile_schedule_fn,
    get_classes_from_module_names,
    get_ff_layer,
    get_attention_layer,
    get_mixed_precision_ignored_classes,
    get_model,
    get_residual_layer,
    get_classes_from_module_names,
)
from lizrd.train.load_and_save_model import (
    get_checkpoint_from_path,
    load_optimizer_state,
    prepare_save_weights_path,
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
        if args.data_seed < 0:
            args.data_seed = random.randint(0, 10000000)

    check_args(args)

    save_weights_path = prepare_save_weights_path(args.save_weights_path)

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

    residual_fn = get_residual_layer(args)

    block_modules = {}
    for module_name in args.block_modules:
        if module_name == "attention":
            block_modules[module_name] = get_attention_layer(args)
        elif module_name == "feedforward":
            block_modules[module_name] = get_ff_layer(args)
        else:
            raise ValueError(f"Unknown module name: {module_name}")

    if args.parallel_blocks:
        modules = block_modules.items()
        block_modules = {
            "parallel": lambda: Parallel(*[module() for _, module in modules])
        }

    checkpoint = (
        get_checkpoint_from_path(args.load_weights_path)
        if args.load_weights_path is not None
        else None
    )

    model, train_sequence_length = get_model(
        reference_seq_len=args.cutoff,
        n_steps=args.n_steps,
        vocab_size=VOCAB_SIZE,
        block_modules=block_modules,
        dm=args.dmodel,
        n_blocks=args.n_blocks,
        device=(
            DEVICE if rank is None else torch.device("cpu")
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
        residual_fn=residual_fn,
        is_logging_process=is_logging_process,
        rank=rank,
        checkpoint=checkpoint,
        reduction_layer_type=args.reduction_layer_type,
        # sequence_length_multiplier=args.sequence_length_multiplier,
        scheduler_params=args.tr_schedule,
        is_eot_id_reducible=args.tr_reducible_eot_token,
    )

    n_learnable_parameters = get_n_learnable_parameters(model)
    args.n_learnable_parameters = n_learnable_parameters
    print(f"Number of learnable parameters: {n_learnable_parameters:_}")

    embedding = [m for m in model.modules() if isinstance(m, EmbeddingLayer)][0]
    head = model.head

    n_learnable_nonembedding_parameters = (
        n_learnable_parameters
        - get_n_learnable_parameters(embedding)
        - get_n_learnable_parameters(head)
    )
    args.n_learnable_nonembedding_parameters = n_learnable_nonembedding_parameters
    print(
        f"Number of learnable nonembedding parameters: {n_learnable_nonembedding_parameters:_}"
    )

    if args.torch_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    if checkpoint is not None:
        load_optimizer_state(optimizer, checkpoint, model, rank)

    scheduler = get_scheduler(args)

    data_distributed = args.ddp_enabled or args.fsdp_enabled
    batch_size = args.batch_size // args.n_gpus if data_distributed else args.batch_size

    common_dataloaders_kwargs = {
        "sequence_length": train_sequence_length,
        "device": DEVICE,
        "num_workers": args.num_workers,
        "batch_size": batch_size,
        "seed": args.data_seed if data_seeds is None else data_seeds[rank],
        "model_type": args.model_type,
        "dataset_type": args.dataset_type,
        "use_dummy_dataset": args.use_dummy_dataset,
    }

    train_dataloader = get_processed_dataset(
        **common_dataloaders_kwargs,
        dataset_split="train",
        dataset_path=args.train_dataset_path,
    )

    common_dataloaders_kwargs["sequence_length"] = args.cutoff

    eval_split = (
        "eval"
        if args.dataset_type == "wikibook"
        else ("train" if args.use_dummy_dataset else "validation")
    )
    eval_dataloader = get_processed_dataset(
        **common_dataloaders_kwargs,
        dataset_split=eval_split,
        dataset_path=args.validation_dataset_path,
    )

    if is_logging_process:
        logger = get_logger(args, model, VOCAB_SIZE)
    else:
        logger = None

    # if args.model_type == "gpt" and is_logging_process:
    #     log_batch(
    #         train_dataloader,
    #         tokenizer_maker=(
    #             tokenizers.GPTTokenizer
    #             if args.model_type == "gpt"
    #             else tokenizers.BertTokenizer
    #         ),
    #     )

    profiler_schedule = (
        torch.profiler.schedule(
            wait=args.profiler_schedule_wait,
            warmup=args.profiler_schedule_warmup,
            active=args.profiler_schedule_active,
            repeat=args.profiler_schedule_repeat,
            skip_first=args.profiler_schedule_skip_first,
        )
        if args.profiler_enabled
        else disable_profile_schedule_fn
    )

    trainer = Trainer(
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
        save_weights_path=save_weights_path,
        save_weights_interval=args.save_weights_interval,
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
        profiler_enabled=args.profiler_enabled,
        profiler_trace_path=args.profiler_trace_path,
        profiler_schedule=profiler_schedule,
        rank=rank,
        start_step=checkpoint["step"] + 1 if checkpoint is not None else 0,
        checkpoint=checkpoint,
    )
    trainer.train(args.n_steps)

    if rank is not None:
        destroy_process_group()


def assert_n_gpus(n_gpus):
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        assert count == n_gpus, f"Expected {n_gpus} GPUs, but found {count}."


if __name__ == "__main__":
    misc.print_available_gpus()
    parser = argparse.ArgumentParser()
    introduce_parser_arguments(parser)
    args = parser.parse_args()
    assert_n_gpus(args.n_gpus)
    if args.data_seed < 0:
        args.data_seed = random.randint(0, 10000000)

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
