def check_args(args):
    if args.granularity_expert_config:
        print(
            "`--granularity_expert_config` is deprecated. Missing granularity arguments are now always computed automatically."
        )

    assert (not args.mixed_precision) or (
        args.mixed_precision and args.mixed_precision_dtype is not None
    ), "To use mixed_precision set mixed_precision_dtype either to 'float16' or 'bfloat16'"

    assert (not args.flash_attention) or (
        args.flash_attention and args.mixed_precision
    ), "Flash attention requires mixed precision to be enabled. Please set `--mixed_precision True`."

    assert (not args.fsdp_enabled) or (
        args.fsdp_enabled and args.mixed_precision_dtype != "float16"
    ), "Our FSDP implementation currently does not support float16 precision (no distributed GradScaler implemented). Please use bfloat16 or disable mixed precision and set its type to None."
