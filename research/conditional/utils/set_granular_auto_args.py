def set_granular_auto_args(args):
    # check everything not required is set
    assert (
        args.batch_size_per_gpu is not None and args.granular_model_config is not None
    )

    args = set_model_config(args)
    args.batch_size = 2048
    args.gradient_accumulation_steps = int(8 / args.n_gpus)
    args.final_lr_fraction = 0.1
    args.final_lr_step = args.n_steps
    args.lr_warmup_steps = int(round(args.n_steps * 0.01))

    return args


def set_model_config(args):
    assert args.granular_model_config in (
        "mini",
        "mini_8",
        "small",
        "medium",
        "base",
    )

    if args.granular_model_config == "mini":
        args.dmodel = 256
        args.dff = 1024
        args.n_blocks = 4
        args.n_att_heads = 4
    elif args.granular_model_config == "mini_8":
        args.dmodel = 256
        args.dff = 1024
        args.n_blocks = 8
        args.n_att_heads = 4
    elif args.granular_model_config == "small":
        args.dmodel = 512
        args.dff = 2048
        args.n_blocks = 4
        args.n_att_heads = 8
    elif args.granular_model_config == "medium":
        args.dmodel = 512
        args.dff = 2048
        args.n_blocks = 8
        args.n_att_heads = 8
    elif args.granular_model_config == "base":
        args.dmodel = 768
        args.dff = 3072
        args.n_blocks = 12
        args.n_att_heads = 12

    return args
