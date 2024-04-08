import argparse


def add_default_parser_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument("--name", type=str, required=True)
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

    parser.add_argument("--fsdp_enabled", action="store_true") 
    parser.add_argument(
        "--fsdp_mixed_precision_dtype",
        type=str,
        choices=["float32"], #TODO adjust
        default=None,
    )
    parser.add_argument("--fsdp_modules_to_wrap", type=str, required=False)
    parser.add_argument(
        "--fsdp_selective_precision_modules",
        type=str,
        default=None,
        help="comma-separated list of modules whose parameters should be wrapped in FSDP with a different precision than the rest of the model. For reference, see get_classes_from_module_names in research/conditional/utils/model_utils.py",
    )
    parser.add_argument(
        "--fsdp_offload_params",
        action="store_true",
    )
    parser.add_argument(
        "--fsdp_min_num_params",
        type=int,
        default=None,
        help="This argument is used only if fsdp_enabled is set to True. It is used to set the minimum number of parameters in a module to be wrapped in FSDP. If the number of parameters is smaller than this value, then the module is not wrapped in FSDP. This is useful for small modules, where the overhead of FSDP is too large compared to the compute of the module.",
    )
    return parser
