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
    parser.add_argument("--dhead", type=int, default=None)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--logging_interval_light", type=int, default=1000000)
    parser.add_argument("--logging_interval_heavy", type=int, default=1000000)
    parser.add_argument("--mask_loss_weight", type=float, default=1.0)
    parser.add_argument("--mask_percent", type=float, default=0.15)
    parser.add_argument("--n_steps", type=int, default=90000)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--torch_seed", type=int, default=42)
    parser.add_argument("--deterministic_experiment", action="store_true")
    parser.add_argument("--tags", nargs="*", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="pmtest/llm-efficiency")
    parser.add_argument(
        "--model_type", type=str, choices=["gpt", "bert"], default="bert"
    )
    parser.add_argument(
        "--objects_for_propagation",
        type=str,
        default=None,
        help="comma-separated list of objects to propagate in forward pass (e.g. 'attention_keys,ff_activation')",
    )

    # parameters usually changed for experiments
    parser.add_argument("--ff_mode", type=str, default="vanilla")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_weights_path", type=str, default=None)
    parser.add_argument("--save_weights_interval", type=int, default=1000)
    parser.add_argument("--load_weights_path", type=str, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--no_ff", action="store_true")
    parser.add_argument("--loss_checkpoint_chungs", type=str, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # paremeters for specific experiments

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
    parser.add_argument("--kernel_r", type=int, default=256)
    parser.add_argument("--redraw_projections_interval", type=int, default=100)
    parser.add_argument("--no_kernel_norm", action="store_true")
    parser.add_argument("--no_average_attn", action="store_true")
    parser.add_argument("--kernel_type", type=str, default="relu")
    parser.add_argument("--activation_type", type=str, default="relu")
    parser.add_argument("--nystrom", action="store_true")
    parser.add_argument("--xfavor", action="store_true")
    parser.add_argument("--flop_matched", action="store_true")
    parser.add_argument("--mix_whole_batch", action="store_true")
    parser.add_argument(
        "--model_parallelism_fragmentation",
        type=str,
        default=None,
        help="comma-separated list of integers, that signify the numbers of model blocks that are first on the new device, e.g. 2,4 means that blocks 0,1 will be on GPU 0, blocks 2,3 will be on GPU 1, and the rest will be on GPU 2",
    )
    parser.add_argument("--data_distributed", action="store_true")
    parser.add_argument("--dataset_type", type=str, default="wikibook")

    # experimental/legacy parameters

    parser.add_argument("--hack_name", type=str, default=None)
    parser.add_argument("--x_flop", action="store_true")
    parser.add_argument("--x_logarithmic", action="store_true")

    return parser
