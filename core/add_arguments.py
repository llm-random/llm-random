import argparse


def add_default_parser_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument("--all_config_paths", type=str, required=False)
    parser.add_argument(
        "--attention_mode", type=str, choices=["vanilla"], required=True
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument(
        "--data_seed", type=int, help="Negative value means random seed"
    )
    parser.add_argument(
        "--dataset_type", type=str, choices=["wikibook", "c4"], required=True
    )
    parser.add_argument("--dff", type=int, required=True)
    parser.add_argument("--dhead", type=int, required=True)
    parser.add_argument("--dmodel", type=int, required=True)
    parser.add_argument(
        "--embedding_mode", type=str, choices=["vanilla"], required=True
    )
    parser.add_argument("--ff_mode", type=str, choices=["vanilla"], required=True)
    parser.add_argument(
        "--final_lr_fraction", type=float, required=True
    )  # TODO make it optional when scheduler is constant
    parser.add_argument("--final_lr_step", type=int, required=True)
    parser.add_argument("--flash_attention_enabled", action="store_true")
    parser.add_argument("--git_branch", type=str, required=False)
    parser.add_argument("--init_scale", type=float, required=True)
    parser.add_argument(
        "--init_type",
        type=str,
        choices=["kaiming_uniform", "truncated_normal"],
        required=True,
    )
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument(
        "--lr_scheduler_type", choices=["constant", "cosine"], type=str, required=True
    )
    parser.add_argument("--lr_scheduler_warmup_steps", type=int, required=True)
    parser.add_argument(
        "--model_type", type=str, choices=["gpt", "bert"], required=True
    )
    parser.add_argument("--n_att_heads", type=int, required=True)
    parser.add_argument("--n_blocks", type=int, required=True)
    parser.add_argument("--n_gpus", type=int, required=True)
    parser.add_argument("--n_steps", type=int, required=True)
    parser.add_argument(
        "--norm_class",
        type=str,
        choices=["layer_norm", "rms_norm"],
        required=True,
    )
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--optimizer_adam_beta1", type=float, required=True)
    parser.add_argument("--optimizer_adam_beta2", type=float, required=True)
    parser.add_argument("--optimizer_weight_decay", type=float, required=True)
    parser.add_argument("--path_to_entry_config", type=str, required=False)
    parser.add_argument(
        "--residual_mode", type=str, choices=["pre_norm"], required=True
    )
    parser.add_argument("--seq_length", type=int, required=True)
    parser.add_argument("--tags", nargs="*", type=str, default=None)
    parser.add_argument("--torch_seed", type=int, required=True)
    parser.add_argument("--train_dataset_path", type=str, required=False)
    parser.add_argument(
        "--use_dummy_dataset",
        action="store_true",
        help="whether to use dummy dataset (for debugging or tests)",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    return parser
