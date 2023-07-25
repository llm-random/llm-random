import gc

import torch

from lizrd.train.train_utils import get_processed_dataset, get_model
from research.conditional.utils.conditional_trainer import ConditionalTrainer
from research.conditional.utils.model_utils import get_ff_layer, get_attention_layer


def introduce_parser_arguments(parser):
    # core hyperparameters, fixed for all experiments; needs a good reason to change

    parser.add_argument("--use_clearml", action="store_true")
    parser.add_argument("--use_neptune", action="store_true")
    parser.add_argument("--batch_size", type=int, default=600)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cutoff", type=int, default=128)
    parser.add_argument("--dmodel", type=int, default=768)
    parser.add_argument("--dff", type=int, default=3072)
    parser.add_argument("--n_att_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--logging_interval_light", type=int, default=1000000)
    parser.add_argument("--logging_interval_heavy", type=int, default=1000000)
    parser.add_argument("--mask_loss_weight", type=float, default=1.0)
    parser.add_argument("--mask_percent", type=float, default=0.15)
    parser.add_argument("--n_steps", type=int, default=90000)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--torch_seed", type=int, default=42)
    parser.add_argument("--tags", nargs="*", type=str, default=None)
    parser.add_argument(
        "--model_type", type=str, choices=["gpt", "bert"], default="bert"
    )

    # parameters usually changed for experiments

    parser.add_argument("--ff_mode", type=str, default="vanilla")
    parser.add_argument("--project_name", type=str, default="")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--loss_checkpoint_chungs", type=str, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--auto_find_grad_accumulation", action="store_true")

    parser.add_argument("--n_experts", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--sparsity_dim", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--expert_size", type=int, required=False)
    parser.add_argument("--topk_fraction", type=float, required=False)
    parser.add_argument("--logging_interval_loss", type=int, default=250)
    parser.add_argument("--every_other_layer", action="store_true")
    parser.add_argument("--expert_random_perm", action="store_true")
    parser.add_argument("--standard_ff_first", action="store_true")
    parser.add_argument("--granularity_expert_config", action="store_true")
    parser.add_argument("--total_experts_width", type=int, required=False)
    parser.add_argument("--effective_dff", type=int, required=False)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--use_opt_einsum", action="store_true")
    parser.add_argument("--share_by_experts", action="store_true")
    parser.add_argument("--share_by_emit_merge", action="store_true")

    # experimental/legacy parameters

    parser.add_argument("--hack_name", type=str, default=None)
    parser.add_argument("--x_flop", action="store_true")
    parser.add_argument("--x_logarithmic", action="store_true")

    return parser


def get_ith_chunk(tensor, chunks, i):
    list_of_chunks = torch.chunk(tensor, chunks, dim=0)
    return list_of_chunks[i]


def get_trainer(args, vocab_size, device, grad_acc_steps):
    train_dataloader = get_processed_dataset(
        max_total_length=args.cutoff,
        mask_percent=args.mask_percent,
        device=device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        seed=args.data_seed,
        model_type=args.model_type,
        distributed=False,
    )
    ff_layer_fun = get_ff_layer(args)
    attention_layer_fun = get_attention_layer(args)

    model = get_model(
        max_length=args.cutoff,
        vocab_size=vocab_size,
        ff_layer_fun=ff_layer_fun,
        attention_layer_fun=attention_layer_fun,
        dm=args.dmodel,
        n_blocks=args.n_blocks,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    trainer = ConditionalTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        vocab_size=vocab_size,
        mask_percent=args.mask_percent,
        mixed_precision=args.mixed_precision,
        logger=None,
        hack_name=args.hack_name,
        model_type=args.model_type,
        logging_interval_loss=args.logging_interval_loss,
        logging_interval_light=args.logging_interval_light,
        logging_interval_heavy=args.logging_interval_heavy,
        n_gpus=args.n_gpus,
        loss_checkpoint_chungs=args.loss_checkpoint_chungs,
        gradient_accumulation_steps=grad_acc_steps,
    )
    return trainer


def find_optimal_grad_accumulation(args, vocab_size, device):
    """
    Find the optimal number of gradient accumulation steps for a given model.
    CURRENTLY NO SUPPORT FOR DISTRIBUTED TRAINING.
    """
    grad_acc_steps = 1
    while True:
        try:
            trainer = get_trainer(args, vocab_size, device, grad_acc_steps)
            trainer.train(10)
            return grad_acc_steps
        except Exception as e:
            print(e)
            torch.cuda.empty_cache()
            gc.collect()
            grad_acc_steps += 1
