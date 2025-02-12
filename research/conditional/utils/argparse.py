import argparse
import json


def load_dict_in_args(s: str):
    return json.loads(s.replace("'", '"'))


def math_eval(value):
    return float(eval(value))  # fallback for other notations like float or exponential


def introduce_parser_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    # CORE model hyperparameters, almost always specified in baseline configs
    parser.add_argument("--cuda_visible", type=str, default=None)
    parser.add_argument(
        "--checkpoint_manager",
        action="store_true",
        help="Used when experiment will last longer than cluster max job time. It repeats jobs for expertiment longer continuation. Combines with periodic model saves. Additionally operates using a manager JSON file which combines with trapezoidal slides creation and execution schedule.",
    )
    # parser.add_argument("--repeater_buffer", type=int, help="In minutes, time before cluster force-clousure that jobs saves and ends itself for next job to continue. Maximu time that model and trainig data needs to be saved, plus MAX time of one step (plus TIME OF VALIDATION that can interfeer!).") #dev TODO, currently fixed 15 min buffer time
    parser.add_argument(
        "--model_type", type=str, choices=["gpt", "bert"], required=True
    )
    parser.add_argument("--ff_mode", type=str, default="vanilla")
    parser.add_argument("--attention_mode", type=str, default="vanilla")
    parser.add_argument("--parallel_blocks", action="store_true")
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
    parser.add_argument("--no_ff", action="store_true")
    parser.add_argument("--moe_inner_expert", type=str, default="ff")

    # CORE training hyperparameters, almost always specified in baseline configs

    parser.add_argument(
        "--n_steps",
        type=int,
        required=False,
        help="Number of gradient steps during training. n_steps, or n_tokens must be None",
    )
    parser.add_argument(
        "--n_tokens",
        type=float,
        default=None,
        required=False,
        help="Number of total training tokens in bilions. n_steps, or n_tokens must be None",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        required=True,
        help="Options: constant, cosine, trapezoidal",
    )
    parser.add_argument(
        "--scheduler_trapezoidal_slides",
        type=str,
        default=None,
        required=False,
        help="""List of dicts having n_steps of total training steps for a periticular split eg. scheduler_trapezoidal_slides: "[{'n_steps':1000},{'n_steps':3000,'n_jobs':2}]"; !no white symbols!; n_jobs indicates how many slurm jobs may be needed to complete slide part of training.""",
    )
    parser.add_argument("--learning_rate", type=math_eval, required=True)
    parser.add_argument("--final_lr_step", type=int, required=False)
    parser.add_argument("--lr_warmup_percent", type=float, required=False)
    parser.add_argument("--final_lr_fraction", type=float, required=False) # TODO fix - it is required
    parser.add_argument(
        "--init_type",
        type=str,
        choices=["kaiming_uniform", "truncated_normal", "truncated_normal_fixed"],
        required=True,
    )
    parser.add_argument("--init_scale", type=float, required=True)

    # other training hyperparameters

    parser.add_argument("--learning_rate_log2", type=float, required=False, default=0.0)
    # these default values are fitted using the scaling rule, to use them,
    # set use_lr_scaling to True and all other lr scaling parameters to 0
    parser.add_argument("--use_lr_scaling", action="store_true")
    parser.add_argument(
        "--lr_scaling_constant_factor",
        type=float,
        required=False,
        default=8.391640956234012,
    )
    parser.add_argument(
        "--lr_scaling_params_factor",
        type=float,
        required=False,
        default=-0.8121798244457074,
    )
    parser.add_argument(
        "--lr_scaling_exp_rate_factor",
        type=float,
        required=False,
        default=-0.25000000000000533,
    )

    parser.add_argument("--deterministic_experiment", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_decay", type=float, default=None)
    parser.add_argument("--lr_warmup_steps", type=int, default=None)
    parser.add_argument(
        "--lr_warmup_tokens", type=float, default=None, help="in bilions of tokens"
    )
    parser.add_argument("--lr_trapezoidal_decay_fraction", type=float, default=0.20)
    parser.add_argument(
        "--lr_trapezoidal_decay_fraction_unit",
        type=str,
        default="tokens",
        choices=["tokens", "steps"],
    )
    parser.add_argument("--lr_decay_interval", type=int, default=0)
    parser.add_argument(
        "--batch_size_rampup_transition_points",
        type=float,
        nargs="*",
        default=None,
        help="list of points when batch size will be ramped up to the next value. Points in of bilions of tokens, if argument batch_size_rampup_units isn't specified",
    )
    parser.add_argument("--batch_size_rampup_sizes", type=int, nargs="*", default=None)
    parser.add_argument(
        "--batch_size_rampup_units",
        type=str,
        choices=["tokens", "steps"],
        default="tokens",  # ensures backward compatibility
        help="options:\n'tokens' - bilions of tokens\n'steps' - gradient steps",
    )

    # CORE data hyperparameters, almost always specified in baseline configs

    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["wikibook", "c4", "fineweb-edu"],
        required=True,
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--cutoff", type=int, required=True)

    # other data hyperparameters
    parser.add_argument("--num_workers", type=int, default=8)

    # as of 8.02.2024 below only works for C4 dataset, as wikibook is technically two separate datasets, but wikibook is small enough to use hf datasets_cashe
    # as of 8.02.2024 it is set automatically on DGX, on other machines use manually
    parser.add_argument("--train_dataset_path", type=str, default=None)
    parser.add_argument("--validation_dataset_path", type=str, default=None)

    parser.add_argument("--final_eval_seed", type=int, default=1912)
    parser.add_argument(
        "--final_eval_dataloader_batch_size",
        type=int,
        default=64,
        help="Batch size for final evaluation dataloader. It should be the same for all runs you are comparing. Its advised to set it to the value of the biggest batch_size per GPU to avoid out of memory error.",
    )
    parser.add_argument(
        "--n_final_eval_batches",
        type=int,
        default=8,
        help="Total number of batches to generate for final evaluation. Should be the same for all runs to compare them.",
    )

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
    parser.add_argument("--torch_compile", action="store_true")
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
    parser.add_argument(
        "--model_parallelism_fragmentation",
        type=str,
        default=None,
        help="comma-separated list of integers, that signify the numbers of model blocks that are first on the new device, e.g. 2,4 means that blocks 0,1 will be on GPU 0, blocks 2,3 will be on GPU 1, and the rest will be on GPU 2",
    )
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--flash_attention", action="store_true")

    # other parameters usually not changed for experiments

    parser.add_argument("--mask_loss_weight", type=float, default=1.0)
    parser.add_argument("--mask_percent", type=float, default=0.15)
    parser.add_argument(
        "--data_seed", type=int, default=-1, help="Negative value means random seed"
    )
    parser.add_argument("--torch_seed", type=int, default=42)

    # hardware
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_nodes", type=int, default=1)

    # Logging parameters
    parser.add_argument("--logger_types", type=str, required=True)
    parser.add_argument("--wandb_entity", type=str, default="ideas_cv")
    parser.add_argument("--project_name", type=str, default="pmtest/llm-random")
    parser.add_argument("--wandb_project", type=str, default="llm-random")
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

    parser.add_argument("--model_fit_gpu_info_database_path", type=str, default=None)
    parser.add_argument("--model_fit_gpu_info_params", type=str, default=None)

    # profiler parameters
    parser.add_argument("--profiler_enabled", action="store_true")
    parser.add_argument("--profiler_trace_path", type=str, default=None)
    parser.add_argument("--profiler_schedule_wait", type=int, default=None)
    parser.add_argument("--profiler_schedule_warmup", type=int, default=None)
    parser.add_argument("--profiler_schedule_active", type=int, default=None)
    parser.add_argument("--profiler_schedule_repeat", type=int, default=None)
    parser.add_argument("--profiler_schedule_skip_first", type=int, default=None)

    # model versioning

    parser.add_argument("--save_weights_path", type=str, default="./model_ckpt")
    parser.add_argument("--dont_save_final_model", action="store_true")
    parser.add_argument("--save_weights_interval", type=int, default=-1)
    parser.add_argument("--load_weights_path", type=str, default=None)

    # paremeters for specific experiments

    ## used by MoE (common)
    parser.add_argument("--general_ff_layer_config", type=str, default=None)
    parser.add_argument(
        "--eval_dynamic_groupsize",
        action="store_true",
        help="During evaluation, evaluate model with multiple group sizes",
    )
    parser.add_argument(
        "--eval_min_group_size_logfactor",
        type=int,
        default=None,
        help="During evaluation, the smallest group size is group_size * 2**eval_min_group_size_logfactor",
    )
    parser.add_argument(
        "--eval_max_group_size_logfactor",
        type=int,
        default=None,
        help="During evaluation, the largest group size is group_size * 2**eval_max_group_size_logfactor",
    )

    ## used often by Continuous MoE

    parser.add_argument("--eval_discrete_mot", action="store_true")
    parser.add_argument("--emit_softmax_over_experts", action="store_true")
    parser.add_argument("--steps_until_start_temperature_learn", type=int, default=0)
    parser.add_argument("--n_experts", type=int)
    parser.add_argument("--group_size", type=int)
    parser.add_argument("--sparsity_dim", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--expert_size", type=int)
    parser.add_argument("--share_by_experts", action="store_true")
    parser.add_argument("--share_by_emit_merge", action="store_true")
    parser.add_argument("--flop_matched", action="store_true")

    ## used by MoE (specific)
    parser.add_argument(
        "--load_balancing_loss_weight",
        type=float,
        default=0.01,
        help="Whether to use auxiliary loss in loss calculations",
    )
    parser.add_argument(
        "--zloss_weight",
        default=0.0,
        type=float,
        help="zloss_weight, if 0 zloss won't be computed",
    )
    parser.add_argument("--topk_fraction", type=float)
    parser.add_argument("--expert_random_perm", action="store_true")
    parser.add_argument(
        "--granularity_expert_config",
        action="store_true",
        help="This argument is deprecated. Provide either (total_experts_width, n_experts, effective_dff) or (expert_size, n_experts, topk_fraction) instead.",
    )
    parser.add_argument("--total_experts_width", type=int)
    parser.add_argument(
        "--granularity",
        type=int,
        help="How smaller is each expert compared to standard MoE",
    )
    parser.add_argument(
        "--expansion_rate",
        type=int,
        help="Factor by which we expand the number of parameters in FF",
    )
    parser.add_argument(
        "--effective_dff_x",
        type=int,
        help="How much FLOPS we want to spend on FF, in multiples of d_model",
    )
    parser.add_argument(
        "--expert_use_topk_initialization",
        type=str,
        choices=["Always", "Never", "Default"],
        default="Default",
        help="Whether to init fan_in of Lin2 in Experts with topk or not. Default means yes for EC and no for TC.",
    )
    parser.add_argument("--effective_dff", type=int)
    parser.add_argument("--softmax_over", type=str, default="tokens")
    parser.add_argument("--use_opt_einsum", action="store_true")
    parser.add_argument("--simulate_group_size", type=int, default=1)
    parser.add_argument("--kernel_r", type=int, default=256)
    parser.add_argument("--redraw_projections_interval", type=int, default=100)
    parser.add_argument("--no_kernel_norm", action="store_true")
    parser.add_argument("--no_average_attn", action="store_true")
    parser.add_argument("--kernel_type", type=str, default="relu")
    parser.add_argument("--nystrom", action="store_true")
    parser.add_argument("--xfavor", action="store_true")
    parser.add_argument("--mix_whole_batch", action="store_true")
    parser.add_argument("--capacity_factor", type=float, default=1.25)
    parser.add_argument(
        "--routing_top_k",
        type=int,
        default=1,
        help="TopK (how many experts a token is routed to) in Token Choice",
    )
    parser.add_argument(
        "--ff_parallel_compute_fraction",
        type=float,
        default=0.5,
        help="This argument is used only if ff_mode is set to expert_choice_with_parallel_ff. In this setting computations "
        "are done both by experts and dense layer and then the results are added. This argument is used to set the "
        "fraction of compute (flops) that is done by FF compared to the whole compute in the layer. For example, "
        "if this argument is 0.5, then half of the compute (flops) is done by FF and half by experts",
    )
    parser.add_argument(
        "--ff_parallel_mode",
        type=str,
        default="modify_expert_size",
        help="This argument is used only if ff_mode is set to expert_choice_with_parallel_ff. In this setting computations "
        "are done both by experts and dense layer and then the results are added. This argument is used to set how the "
        "parameters of the experts are modified to adjust compute used bu experts. Possible values: modify_expert_size, "
        "modify_topk_fraction, modify_n_experts",
    )
    parser.add_argument(
        "--dont_vectorize_switch",
        action="store_true",
        help="This argument is used in Token Choice to force it to use `for`-s",
    )

    parser.add_argument("--group_granular_moe_by_batch", action="store_true")
    parser.add_argument("--layer_norm_in_expert_choice", action="store_true")
    parser.add_argument("--granular_moe_one_hot_impl", action="store_true")
    parser.add_argument(
        "--softmax_ungrouped",
        action="store_true",
        help="in grouped ExpertChoice, run softmax over non-grouped tokens",
    )
    parser.add_argument(
        "--use_full_einsum",
        action="store_true",
        help="in grouped ExpertChoice, use squash all linears with einsum",
    )
    parser.add_argument(
        "--use_torch_bmm",
        action="store_true",
        help="in grouped ExpertChoice, use one hot implementation with all "
        "linear operations performed using torch.bmm",
    )
    parser.add_argument(
        "--use_dummy_dataset",
        action="store_true",
        help="whether to use dummy dataset (for debugging or tests)",
    )

    # experimental/legacy parameters

    parser.add_argument("--x_flop", action="store_true")
    parser.add_argument("--x_logarithmic", action="store_true")

    # mamba
    parser.add_argument("--mamba_mode", type=str, default="vanilla")
    parser.add_argument(
        "--block_modules", type=str, default=["attention", "feedforward"], nargs="+"
    )
    parser.add_argument("--mamba_expansion", type=float, default=2.0)
    parser.add_argument("--no_positional_embedding", action="store_true")

    parser.add_argument(
        "--dr_routing_type",
        type=str,
        choices=["expert_choice", "token_choice"],
        default="token_choice",
    )
    parser.add_argument("--dr_linear_first", action="store_true")
    parser.add_argument("--dr_relu_with_first", action="store_true")

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
        "--relative_lr",
        type=load_dict_in_args,
        default=None,
        help="""Dictionary with relative learning rates for different modules
        Example: --relative_lr "{'attention': 0.1, 'feedforward': 0.1, 'moe': 0.1}
        Example in config yaml:
        relative_lr:
            attention: 0.1
            feedforward: 0.1
            moe: 0.1
        """,
    )
    parser.add_argument(
        "--relative_init_scale",
        type=load_dict_in_args,
        default=None,
        help="""Dictionary with relative initialization scales for different modules
        Example: --relative_init_scale "{'attention': 0.1, 'feedforward': 0.1, 'moe': 0.1}
        Example in config yaml:
        relative_init_scale:
            attention: 0.1
            feedforward: 0.1
            moe: 0.1
        """,
    )
    parser.add_argument(
        "--print_parameter_names",
        action="store_true",
        help="Print all parameter names in the model",
    )

    parser.add_argument(
        "--verbose_relative_init_scale",
        action="store_true",
        help="Print names of parameters that were rescaled",
    )

    parser.add_argument(
        "--get_router_values_from",
        type=str,
        choices=[
            "weights",
            "gate_weight",
            "lin1_weight",
        ],
        default="weights",
        required=False,
        help="'weights' is default MoE, 'gate' maximizes average activation in gating,"
        "'lin1' is similar but takes the weights from lin1 and is compatible with non-gated expert",
    )

    parser.add_argument(
        "--moe_values_exp",
        type=str,
        default="1.0",
        help="Exponent for values multiplier in MoE routing. "
        "0 means no multiplier, 1 is the standard, 2 is the square of the standard, etc. "
        "'trainable' means that the exponent is trainable. ",
    )

    parser.add_argument(
        "--moe_detach_gate", action="store_true", help="Detach gate in MoE routing"
    )

    parser.add_argument("--projected_weights_path", type=str, default=None)
    parser.add_argument("--projected_dmodel", type=int, required=False)
    parser.add_argument("--projected_dff", type=int, required=False)
    parser.add_argument("--projected_distillation", action="store_true")
    parser.add_argument("--projection_init_type", type=str, default=None, help="eg. random, None (does not share same projection)") # None does not share same projection
    parser.add_argument("--no_projected_head", action="store_true")
    parser.add_argument("--no_layer_norm", action="store_true")
    parser.add_argument("--fsdp_use_orig_params", action="store_true")
    parser.add_argument("--unprojected_embeddings", action="store_true")
    parser.add_argument("--unprojected_attention", action="store_true")
    parser.add_argument("--unprojected_ff", action="store_true")
    
    return parser
