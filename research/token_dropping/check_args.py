def check_args(args):
    assert (not args.mixed_precision) or (
        args.mixed_precision and args.mixed_precision_dtype is not None
    ), "To use mixed_precision set mixed_precision_dtype either to 'float16' or 'bfloat16'"

    assert (not args.flash_attention) or (
        args.flash_attention and args.mixed_precision
    ), "Flash attention requires mixed precision to be enabled. Please set `--mixed_precision True`."

    if args.save_weights_path is not None:
        filename = args.save_weights_path.split("/")[-1]
        assert (
            "." not in filename
        ), "Do not add filename extensions (e.g. .pt or .pth) to save_weights_path! It is added automatically, along with step number."
