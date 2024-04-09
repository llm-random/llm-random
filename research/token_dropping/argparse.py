import argparse


def introduce_parser_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    # CORE model hyperparameters, almost always specified in baseline configs
    parser.add_argument(
        "--model_type", type=str, choices=["gpt", "bert"], required=True
    )
    parser.add_argument("--ff_mode", type=str, default="vanilla")
    parser.add_argument("--attention_mode", type=str, default="vanilla")
    parser.add_argument("--n_blocks", type=int, required=True)
    parser.add_argument("--dmodel", type=int, required=True)
    parser.add_argument("--dff", type=int, required=False)  # not used by granularity
    parser.add_argument("--n_att_heads", type=int, required=True)
    parser.add_argument("--dhead", type=int, default=None)

    # other model hyperparameters
    parser.add_argument("--activation_type", type=str, default="relu")
    parser.add_argument("--residual_mode", type=str, default="pre_norm")
    parser.add_argument("--every_other_layer", action="store_true")
    parser.add_argument("--standard_ff_first", action="store_true")

    # CORE training hyperparameters, almost always specified in baseline configs

    parser.add_argument("--n_steps", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--scheduler", type=str, required=True)
    parser.add_argument("--final_lr_step", type=int, required=False)
    parser.add_argument("--final_lr_fraction", type=float, required=False)
    parser.add_argument(
        "--init_type",
        type=str,
        choices=["kaiming_uniform", "truncated_normal"],
        required=True,
    )
    parser.add_argument("--init_scale", type=float, required=True)

    # other training hyperparameters

    parser.add_argument("--deterministic_experiment", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)

    # CORE data hyperparameters, almost always specified in baseline configs

    parser.add_argument(
        "--dataset_type", type=str, choices=["wikibook", "c4"], required=True
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--cutoff", type=int, required=True)

    # other data hyperparameters
    parser.add_argument("--num_workers", type=int, default=8)

    # as of 8.02.2024 below only works for C4 dataset, as wikibook is technically two separate datasets, but wikibook is small enough to use hf datasets_cashe
    # as of 8.02.2024 it is set automatically on DGX, on other machines use manually
    parser.add_argument("--train_dataset_path", type=str, default=None)
    parser.add_argument("--validation_dataset_path", type=str, default=None)

    # training tricks for memory and speed
    parser.add_argument(
        "--activation_checkpointing_modules",
        type=str,
        default=None,
        help="comma-separated list of modules whose forward pass should be checkpointed. For reference, see get_classes_from_module_names in research/conditional/utils/model_utils.py",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument(
        "--mixed_precision_dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default=None,
    )
    parser.add_argument("--loss_checkpoint_chungs", type=int, default=0)
    parser.add_argument("--ddp_enabled", action="store_true")
    parser.add_argument("--fsdp_enabled", action="store_true")
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
    parser.add_argument(
        "--fsdp_modules_to_wrap",
        type=str,
        default=None,
        help="This argument is used only if fsdp_enabled is set to True. It is used to set the list of modules that should be wrapped in FSDP. This is an alternative to wrapping using fsdp_min_num_of_params. For reference, see get_classes_from_module_names in research/conditional/utils/model_utils.py",
    )
    parser.add_argument(
        "--fsdp_selective_precision_modules",
        type=str,
        default=None,
        help="comma-separated list of modules whose parameters should be wrapped in FSDP with a different precision than the rest of the model. For reference, see get_classes_from_module_names in research/conditional/utils/model_utils.py",
    )
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--flash_attention", action="store_true")

    # other parameters usually not changed for experiments

    parser.add_argument(
        "--data_seed", type=int, default=-1, help="Negative value means random seed"
    )
    parser.add_argument("--torch_seed", type=int, default=42)

    # hardware
    parser.add_argument("--n_gpus", type=int, default=1)

    # Logging parameters
    parser.add_argument("--use_clearml", action="store_true")
    parser.add_argument("--use_neptune", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default="ideas_cv")
    parser.add_argument("--project_name", type=str, default="pmtest/llm-random")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--tags", nargs="*", type=str, default=None)
    parser.add_argument("--logging_interval_light", type=int, default=1000000)
    parser.add_argument("--logging_interval_heavy", type=int, default=1000000)
    parser.add_argument("--logging_interval_loss", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--n_eval_batches", type=int, default=10)
    parser.add_argument("--log_gradients_and_weights", action="store_true")
    parser.add_argument("--path_to_entry_config", type=str, default=None)
    parser.add_argument("--all_config_paths", type=str, default=None)
    parser.add_argument("--git_branch", type=str, default=None)
    parser.add_argument("--decoding_interval", type=int, default=0)

    # model versioning

    parser.add_argument("--save_weights_path", type=str, default=None)
    parser.add_argument("--save_weights_interval", type=int, default=1000)
    parser.add_argument("--load_weights_path", type=str, default=None)

    parser.add_argument(
        "--use_dummy_dataset",
        action="store_true",
        help="whether to use dummy dataset (for debugging or tests)",
    )

    # experimental/legacy parameters

    parser.add_argument("--x_flop", action="store_true")
    parser.add_argument("--x_logarithmic", action="store_true")

    parser.add_argument(
        "--norm_class",
        type=str,
        choices=[
            "layer_norm",
            "rms_norm",
        ],
        default="layer_norm",
        required=False,
    )
    parser.add_argument(
        "--block_modules", type=str, default=["attention", "feedforward"], nargs="+"
    )
    parser.add_argument("--no_positional_embedding", action="store_true")
    return parser
