import argparse
from collections import defaultdict
import os
import random
from typing import Callable, Optional
import socket

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

from lizrd.core import misc
from lizrd.core.llm import Parallel
from lizrd.support.logging import (
    get_current_logger,
    get_logger,
    log_and_print_model_param_count,
)
from lizrd.support.misc import (
    get_argument_attributes,
    set_seed,
)
from lizrd.train.checkpoints_manager import start_job_manager_assessment
from lizrd.train.train_utils import (
    get_model,
)
from lizrd.text import tokenizers
from research.conditional.utils.check_args import check_args
from research.conditional.utils.misc_tools import (
    get_slurm_job_id,
    get_termination_timestamp_slurm,
)
from research.datasets import DataloaderWrapper, get_processed_dataset
from lizrd.train.scheduler import get_scheduler
from research.conditional.utils.conditional_trainer import ConditionalTrainer
from research.conditional.utils.argparse import introduce_parser_arguments
from research.conditional.utils.model_utils import (
    disable_profile_schedule_fn,
    get_classes_from_module_names,
    get_ff_layer,
    get_attention_layer,
    get_mamba_layer,
    get_mixed_precision_ignored_classes,
    get_residual_layer,
    get_classes_from_module_names,
    update_model_fit_gpu_info,
    get_vanilla_mamba_layer,
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


def make_param_groups_and_lr_ratios(args, model):
    lr = args.learning_rate
    if args.relative_lr is None:
        return [{"params": model.parameters(), "lr": lr}], [1.0]

    relative_lr: dict = args.relative_lr

    lr_to_params = defaultdict(list)
    for name, param in model.named_parameters():
        ratio = 1.0
        for possible_name in relative_lr.keys():
            if possible_name in name:
                ratio = relative_lr[possible_name]
                break
        lr_to_params[ratio * lr].append(param)
    param_grops = [
        {"params": params, "lr": lr_group} for lr_group, params in lr_to_params.items()
    ]
    ratios_in_group_order = [param_group["lr"] / lr for param_group in param_grops]
    return param_grops, ratios_in_group_order


def rescale_params_after_init(args, model):
    relative_scale: dict[str, float] = args.relative_init_scale
    verbose = args.verbose_relative_init_scale

    if relative_scale is None:
        return
    for name, param in model.named_parameters():
        scale = 1.0
        for possible_name in relative_scale.keys():
            if possible_name in name:
                if verbose:
                    print(f"Rescaling {name} by {relative_scale[possible_name]}")
                scale = relative_scale[possible_name]
                break
        param.data *= scale


def main(
    rank: Optional[int],
    data_seeds: Optional[list[int]] = None,
    port: str = "29500",
    unique_save_weights_path: Optional[str] = None,
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

    residual_fn = get_residual_layer(args)

    model_fit_gpu_info_params = get_argument_attributes(
        args, args.model_fit_gpu_info_params
    )
    update_model_fit_gpu_info(
        args.model_fit_gpu_info_database_path, model_fit_gpu_info_params, "initialized"
    )

    if args.general_ff_layer_config is None:
        block_modules = {}
        for module_name in args.block_modules:
            if module_name == "attention":
                block_modules[module_name] = get_attention_layer(args)
            elif module_name == "feedforward":
                block_modules[module_name] = get_ff_layer(args)
            elif module_name == "mamba":
                block_modules[module_name] = get_mamba_layer(args)
            elif module_name == "vanilla_mamba":
                block_modules[module_name] = get_vanilla_mamba_layer(args)
            else:
                raise ValueError(f"Unknown module name: {module_name}")

        if args.parallel_blocks:
            modules = block_modules.items()
            block_modules = {
                "parallel": lambda: Parallel(*[module() for _, module in modules])
            }
    else:
        ff_layers = args.general_ff_layer_config.split(",")
        ff_layer_funs = []
        for layer in ff_layers:
            args.ff_mode = layer
            ff_layer_funs.append(get_ff_layer(args))
        attention_fn = get_attention_layer(args)
        block_modules = [
            {
                "attention": attention_fn,
                "feedforward": ff_fun,
            }
            for ff_fun in ff_layer_funs
        ]

    if not args.checkpoint_manager:
        checkpoint = (
            get_checkpoint_from_path(args.load_weights_path)
            if args.load_weights_path is not None
            else None
        )
    else:
        checkpoint_path, checkpoint_metadata = start_job_manager_assessment(
            get_slurm_job_id(), is_logging_process
        )
        checkpoint = (
            get_checkpoint_from_path(checkpoint_path) if checkpoint_path else None
        )

    model = get_model(
        max_length=args.cutoff,
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
        model_fragmentation=args.model_parallelism_fragmentation,
        residual_fn=residual_fn,
        is_logging_process=is_logging_process,
        rank=rank,
        include_positional_embedding=(not args.no_positional_embedding)
        and (args.attention_mode != "rope"),
        checkpoint=checkpoint,
    )

    if is_logging_process:
        if checkpoint and "logger" in checkpoint and "run_id" in checkpoint["logger"]:
            logger_runs_ids = checkpoint["logger"]["run_id"]
        else:
            if args.scheduler_trapezoidal_slides:
                logger_runs_ids = []
                for _ in range(len(args.scheduler_trapezoidal_slides) + 1):
                    logger_runs_ids.append(None)
            else:
                logger_runs_ids = None
        logger = get_logger(args, model, VOCAB_SIZE, logger_runs_ids)
        if checkpoint_path:
            logger.report_text(
                title=f"job/loaded_checkpoint",
                value=checkpoint_path,
                iteration=checkpoint["step"],
            )
    else:
        logger = None

    args.args_override = None
    if checkpoint and "args_override" in checkpoint and checkpoint["args_override"]:
        args.args_override = checkpoint["args_override"]
        for key, value in args.args_override.items():
            if hasattr(args, key):
                setattr(args, key, value)
        if is_logging_process:
            logger.report_text(
                title=f"args/args_override",  # dev atomize logging to per arg update log in args/{name of arg}
                value=str(args.args_override),
                iteration=checkpoint["step"],
            )

    log_and_print_model_param_count(args, model, vocab_size=VOCAB_SIZE)

    if args.torch_compile:
        model = torch.compile(model)

    if args.print_parameter_names:
        for name, param in model.named_parameters():
            print(name, param.shape)

    param_grops, ratios_in_group_order = make_param_groups_and_lr_ratios(args, model)

    optimizer = torch.optim.AdamW(
        param_grops,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    if checkpoint is not None:
        load_optimizer_state(optimizer, checkpoint, model, rank)

    scheduler = get_scheduler(args, ratios_in_group_order)
    print(f"Scheduler_ratios: {scheduler.ratios}")
    if not args.checkpoint_manager:
        rescale_params_after_init(args, model)

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
        **common_dataloaders_kwargs,
        dataset_split="train",
        dataset_path=args.train_dataset_path,
    )

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

    if args.model_type == "gpt" and is_logging_process:
        log_batch(
            train_dataloader,
            tokenizer_maker=(
                tokenizers.GPTTokenizer
                if args.model_type == "gpt"
                else tokenizers.BertTokenizer
            ),
        )

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
        cutoff=args.cutoff,
        lr_scheduler=scheduler,
        model_type=args.model_type,
        logging_interval_loss=args.logging_interval_loss,
        logging_interval_light=args.logging_interval_light,
        logging_interval_heavy=args.logging_interval_heavy,
        eval_interval=args.eval_interval,
        n_eval_batches=args.n_eval_batches,
        n_gpus=args.n_gpus,
        save_weights_path=unique_save_weights_path,
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
        model_fit_gpu_info_database_path=args.model_fit_gpu_info_database_path,
        model_fit_gpu_info_params=model_fit_gpu_info_params,
        profiler_enabled=args.profiler_enabled,
        profiler_trace_path=args.profiler_trace_path,
        profiler_schedule=profiler_schedule,
        rank=rank,
        start_step=checkpoint["step"] + 1 if checkpoint is not None else 0,
        checkpoint=checkpoint,
        repeater_job_end_time=get_termination_timestamp_slurm()
        if args.checkpoint_manager
        else None,
        scheduler_trapezoidal_slides=args.scheduler_trapezoidal_slides,
        args_override=args.args_override,
    )
    trainer.train(args.n_steps)

    if rank is not None:
        destroy_process_group()


if __name__ == "__main__":
    misc.print_available_gpus()
    parser = argparse.ArgumentParser()
    introduce_parser_arguments(parser)
    args = parser.parse_args()
    if args.data_seed < 0:
        args.data_seed = random.randint(0, 10000000)

    save_weights_path = prepare_save_weights_path(args.save_weights_path)

    if args.ddp_enabled or args.fsdp_enabled:
        random.seed(args.data_seed)
        data_seeds = [random.randint(0, 10000000) for _ in range(args.n_gpus)]

        # find free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = str(s.getsockname()[1])
        mp.spawn(
            main,
            args=[
                data_seeds,
                port,
                save_weights_path,
                args,
            ],
            nprocs=args.n_gpus,
        )
    else:
        main(None, args=args, unique_save_weights_path=save_weights_path)
