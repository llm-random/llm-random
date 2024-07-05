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

    assert (not args.profiler_enabled) or (
        args.profiler_enabled
        and args.profiler_schedule_wait is not None
        and args.profiler_schedule_warmup is not None
        and args.profiler_schedule_active is not None
        and args.profiler_schedule_repeat is not None
        and args.profiler_schedule_skip_first is not None
        and args.profiler_trace_path is not None
    ), "To use profiler set all profiler_schedule arguments"

    if args.save_weights_path is not None:
        filename = args.save_weights_path.split("/")[-1]
        assert (
            "." not in filename
        ), "Do not add filename extensions (e.g. .pt or .pth) to save_weights_path! It is added automatically, along with step number."


    # Tokenizex

    atomization_strategy = args.atomization_strategy
    steps_p_sum = 0
    for e in atomization_strategy:
        assert e[1] <= 1 and e[1] >= 0
        steps_p_sum += e[0]
    normalized_a_strat = []
    for e in atomization_strategy:
        normalized_a_strat.append((e[0]/normalized_a_strat, e[1]))
    args.atomization_strategy = normalized_a_strat

    if args.input_part_no_atomized != 0:
        raise NotImplementedError("Not implemented - optimization feature (mostly)")
    
    if args.residual_mode is not None:
        raise NotImplementedError("Not implemented")
    
    if args.input_wise_positional_embedding and args.no_positional_embedding:
        raise Exception("Cannot have both")
    
    if args.model_type is not None or args.model_type is not "gpt":
        raise Exception("Only GPT")
    