from ast import literal_eval


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

    if args.checkpoint_manager:
        assert (
            args.load_weights_path == None
        ), "Loads model according to checkpoint manager"
        assert (
            args.relative_init_scale == None
        ), "Seems wrong to apply init scale on loaded and already trained weights"
        assert (
            args.logger_types == "neptune"
        ), "Checkpoint manager is implemented only for neptune logger"

    if args.scheduler_trapezoidal_slides:
        assert args.scheduler == "trapezoidal"
        args.scheduler_trapezoidal_slides = literal_eval(
            args.scheduler_trapezoidal_slides
        )
        new_scheduler_trapezoidal_slides = []
        for slide in args.scheduler_trapezoidal_slides:
            slide["split_step"] = (
                int(slide["n_steps"] * (1 - args.lr_trapezoidal_decay_fraction)) - 1
            )
            new_scheduler_trapezoidal_slides.append(slide)
        args.scheduler_trapezoidal_slides = new_scheduler_trapezoidal_slides
