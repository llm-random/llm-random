import argparse
import os
import random
import socket
from typing import Callable, Optional

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from lizrd.core import misc
from lizrd.support.logging import get_current_logger, get_logger
from lizrd.support.misc import generate_random_string
from lizrd.text import tokenizers
from lizrd.train.scheduler import get_scheduler
from lizrd.train.train_utils import (
    get_model,
)
from research.conditional.utils.argparse import introduce_parser_arguments
from research.conditional.utils.conditional_trainer import ConditionalTrainer
from research.conditional.utils.misc_tools import set_seed
from research.conditional.utils.model_utils import (
    get_ff_layer,
    get_attention_layer,
    get_residual_layer,
)
from research.datasets import DataloaderWrapper, get_processed_dataset


#
# def rogue_speedtest():
#     "juts the calculation, no modules"
#     x1 = torch.randn([1024, 256, 512]).cuda()
#     y1 = torch.randn([1024, 512, 512]).cuda()
#     z1 = torch.randn([1024, 512, 512]).cuda()
#     w1 = torch.randn([1024, 256, 512]).cuda()
#
#     x2 = torch.randn([512, 256, 512]).cuda()
#     y2 = torch.randn([512, 512, 1024]).cuda()
#     z2 = torch.randn([512, 1024, 512]).cuda()
#     w2 = torch.randn([512, 256, 1024]).cuda()
#
#     x3 = torch.randn([256, 256, 512]).cuda()
#     y3 = torch.randn([256, 512, 2048]).cuda()
#     z3 = torch.randn([256, 2048, 512]).cuda()
#     w3 = torch.randn([256, 256, 2048]).cuda()
#
#     for function_name in [
#         "bmm_1_no_mixed_precision",
#         "bmm2_no_mixed_precision",
#         "bmm_3_no_mixed_precision",
#         "bmm_1_mixed_precision",
#         "bmm2_mixed_precision",
#         "bmm_3_mixed_precision",
#         "bmm_only_second_bmm_no_mixed_precision",
#         "bmm_only_second_bmm_mixed_precision",
#     ]:
#         for n_experts, (x, y, z, w) in zip(
#             [1024, 512, 256], [(x1, y1, z1, w1), (x2, y2, z2, w2), (x3, y3, z3, w3)]
#         ):
#             t = torch.utils.benchmark.Timer(
#                 stmt=f"{function_name}(x,y,z,w)",
#                 setup=f"from __main__ import {function_name}",
#                 label=f"{n_experts}_{function_name}",
#                 globals={
#                     "x": x,
#                     "y": y,
#                     "z": z,
#                     "w": w,
#                     "function_name": function_name,
#                 },
#             )
#             print(
#                 f"\n ------------------------------------------- \nFor n_experts = {n_experts} and calculation mode = {function_name}: \n"
#             )
#             print(t.timeit(1000))
#
#
# def rogue_speedtest2(args):
#     common_args = {
#         "dm": args.dmodel,
#         "dff": args.dff,
#         "group_size": args.group_size,
#         "sparsity_dim": args.sparsity_dim,
#         "temperature": args.temperature,
#         "expert_size": args.expert_size,
#         "use_opt_einsum": args.use_opt_einsum,
#         "flop_matched": args.flop_matched,
#     }
#     contmoe_1024 = SpeedtestContMoE(n_experts=1024, **common_args).cuda()
#     contmoe_512 = SpeedtestContMoE(n_experts=512, **common_args).cuda()
#     contmoe_256 = SpeedtestContMoE(n_experts=256, **common_args).cuda()
#
#     x_common = torch.randn([1, 256, 256, 512]).cuda()
#
#     weights_1024 = torch.randn([1, 256, 1024, 256]).cuda()
#     weights_512 = torch.randn([1, 256, 512, 256]).cuda()
#     weights_256 = torch.randn([1, 256, 256, 256]).cuda()
#
#     post_merge_1024 = torch.randn([1, 256, 1024, 512]).cuda()
#     post_merge_512 = torch.randn([1, 256, 512, 512]).cuda()
#     post_merge_256 = torch.randn([1, 256, 256, 512]).cuda()
#
#     post_rearrange_1024 = torch.randn([1024, 256, 512]).cuda()
#     post_rearrange_512 = torch.randn([512, 256, 512]).cuda()
#     post_rearrange_256 = torch.randn([256, 256, 512]).cuda()
#
#     post_lin1_1024 = torch.randn([1024, 256, 512]).cuda()
#     post_lin1_512 = torch.randn([512, 256, 1024]).cuda()
#     post_lin1_256 = torch.randn([256, 256, 2048]).cuda()
#
#     post_lin2_1024 = torch.randn([1024, 256, 512]).cuda()
#     post_lin2_512 = torch.randn([512, 256, 512]).cuda()
#     post_lin2_256 = torch.randn([256, 256, 512]).cuda()
#
#     post_rearrange2_1024 = torch.randn([1, 256, 1024, 512]).cuda()
#     post_rearrange2_512 = torch.randn([1, 256, 512, 512]).cuda()
#     post_rearrange2_256 = torch.randn([1, 256, 256, 512]).cuda()
#
#     post_emit_1024 = torch.randn([1, 256, 256, 512]).cuda()
#     post_emit_512 = torch.randn([1, 256, 256, 512]).cuda()
#     post_emit_256 = torch.randn([1, 256, 256, 512]).cuda()
#
#     # for 256 experts, the shape of x post merge is torch.Size([1, 256, 256, 512])
#     # for 256 experts, the shape of x post rearrange is torch.Size([256, 256, 512])
#     # for 256 experts, the shape of x post lin1 is torch.Size([256, 256, 2048])
#     # for 256 experts, the shape of x post lin2 is torch.Size([256, 256, 512])
#     # for 256 experts, the shape of x post rearrange2 is torch.Size([1, 256, 256, 512])
#     # for 256 experts, the shape of x post emit is torch.Size([1, 256, 256, 512])
#
#     # for 512 experts, the shape of x post merge is torch.Size([1, 256, 512, 512])
#     # for 512 experts, the shape of x post rearrange is torch.Size([512, 256, 512])
#     # for 512 experts, the shape of x post lin1 is torch.Size([512, 256, 1024])
#     # for 512 experts, the shape of x post lin2 is torch.Size([512, 256, 512])
#     # for 512 experts, the shape of x post rearrange2 is torch.Size([1, 256, 512, 512])
#     # for 512 experts, the shape of x post emit is torch.Size([1, 256, 256, 512])
#
#     # for 1024 experts, the shape of x post merge is torch.Size([1, 256, 1024, 512])
#     # for 1024 experts, the shape of x post rearrange is torch.Size([1024, 256, 512])
#     # for 1024 experts, the shape of x post lin1 is torch.Size([1024, 256, 512])
#     # for 1024 experts, the shape of x post lin2 is torch.Size([1024, 256, 512])
#     # for 1024 experts, the shape of x post rearrange2 is torch.Size([1, 256, 1024, 512])
#     # for 1024 experts, the shape of x post emit is torch.Size([1, 256, 256, 512])
#
#     for function_name in ["normal_forward", "mixed_precision_forward"]:
#         print(
#             f"------------------------------------------------- now testing {function_name}"
#         )
#         for (
#             layer,
#             input,
#             weights,
#             postmerge,
#             postrearrange,
#             postlin1,
#             postlin2,
#             postrearrange2,
#             postemit,
#         ) in zip(
#             [contmoe_1024, contmoe_512, contmoe_256],
#             [x_common.clone(), x_common.clone(), x_common.clone()],
#             [weights_1024, weights_512, weights_256],
#             [post_merge_1024, post_merge_512, post_merge_256],
#             [post_rearrange_1024, post_rearrange_512, post_rearrange_256],
#             [post_lin1_1024, post_lin1_512, post_lin1_256],
#             [post_lin2_1024, post_lin2_512, post_lin2_256],
#             [post_rearrange2_1024, post_rearrange2_512, post_rearrange2_256],
#             [post_emit_1024, post_emit_512, post_emit_256],
#         ):
#             t = torch.utils.benchmark.Timer(
#                 stmt=f"{function_name}(layer,x,w,post_m,post_r,post_1,post_2,post_r2,post_e)",
#                 setup=f"from __main__ import {function_name}",
#                 label=f"{layer.n_experts}_{function_name}",
#                 globals={
#                     "function_name": function_name,
#                     "layer": layer,
#                     "x": input,
#                     "w": weights,
#                     "post_m": postmerge,
#                     "post_r": postrearrange,
#                     "post_1": postlin1,
#                     "post_2": postlin2,
#                     "post_r2": postrearrange2,
#                     "post_e": postemit,
#                 },
#             )
#             print(
#                 f"\n ------------------------------------------- \nFor n_experts = {layer.n_experts} and calculation mode = {function_name}:"
#             )
#             print(t.timeit(300))


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

    if args.allow_matmul_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    print(
        "11111111111111111111111111111111111111111111111111111111111111111111111111111"
    )

    if args.granularity_expert_config:
        print(
            "`--granularity_expert_config` is deprecated. Missing granularity arguments are now always computed automatically."
        )

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

    data_distributed = True if rank is not None else False
    ff_layer_fun = get_ff_layer(args)
    attention_layer_fun = get_attention_layer(args)
    residual_fn = get_residual_layer(args)
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

    if args.torch_compile:
        model = torch.compile(model)

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
        "model_type": args.model_type,
        "dataset_type": args.dataset_type,
        "use_dummy_dataset": args.use_dummy_dataset,
    }
    train_dataloader = get_processed_dataset(
        **common_dataloaders_kwargs, dataset_split="train"
    )
    eval_dataloader = get_processed_dataset(
        **common_dataloaders_kwargs,
        dataset_split="eval"
        if args.dataset_type == "wikibook"
        else (
            "train"
            if args.dataset_type == "c4" and args.use_dummy_dataset
            else "validation"
        ),
    )
    # rogue_speedtest2(args)
    logger = get_logger(args, model, VOCAB_SIZE)

    # in case of data parallelism, only gpu:0 should log
    is_process_logging = True if rank is None or rank == 0 else False

    if args.model_type == "gpt" and (rank is None or rank == 0):
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
        vocab_size=VOCAB_SIZE,
        mask_percent=args.mask_percent,
        mixed_precision=args.mixed_precision,
        logger=logger,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        lr_scheduler=scheduler,
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
        log_gradients_and_weights=args.log_gradients_and_weights,
        max_sequence_length=args.cutoff,
        is_process_logging=is_process_logging,
        decoding_logging_steps=args.decoding_logging_steps,
        steps_until_anneal=args.steps_until_anneal,
        n_steps=args.n_steps,
        entropy_loss_weight=args.entropy_loss_weight,
    )

    trainer.train(args.n_steps)

    if rank is not None:
        destroy_process_group()


if __name__ == "__main__":
    #
    # def permute(x, y, z, w):
    #     return x.permute(1, 0, 2)
    #
    # def bmm_1_no_mixed_precision(x, y, z, w):
    #     x = torch.bmm(x, y)
    #     return x
    #
    # def bmm2_no_mixed_precision(x, y, z, w):
    #     x = torch.bmm(x, y)
    #     x = torch.relu_(x)
    #     return x
    #
    # def bmm_3_no_mixed_precision(x, y, z, w):
    #     x = torch.bmm(x, y)
    #     x = torch.relu_(x)
    #     x = torch.bmm(x, z)
    #     return x
    #
    # def bmm_1_mixed_precision(x, y, z, w):
    #     with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
    #         x = torch.bmm(x, y)
    #         return x
    #
    # def bmm2_mixed_precision(x, y, z, w):
    #     with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
    #         x = torch.bmm(x, y)
    #         x = torch.relu_(x)
    #         return x
    #
    # def bmm_3_mixed_precision(x, y, z, w):
    #     with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
    #         x = torch.bmm(x, y)
    #         x = torch.relu_(x)
    #         x = torch.bmm(x, z)
    #         return x
    #
    # def bmm_only_second_bmm_no_mixed_precision(x, y, z, w):
    #     x = torch.bmm(w, z)
    #     return x
    #
    # def bmm_only_second_bmm_mixed_precision(x, y, z, w):
    #     with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
    #         x = torch.bmm(w, z)
    #         return x
    #
    # def normal_forward(
    #     layer,
    #     x,
    #     w,
    #     a,
    #     b,
    #     c,
    #     d,
    #     e,
    #     f,
    # ):
    #     return layer(
    #         x,
    #         w,
    #         a,
    #         b,
    #         c,
    #         d,
    #         e,
    #         f,
    #     )
    #
    # def mixed_precision_forward(
    #     layer,
    #     x,
    #     w,
    #     a,
    #     b,
    #     c,
    #     d,
    #     e,
    #     f,
    # ):
    #     with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
    #         return layer(
    #             x,
    #             w,
    #             a,
    #             b,
    #             c,
    #             d,
    #             e,
    #             f,
    #         )

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
