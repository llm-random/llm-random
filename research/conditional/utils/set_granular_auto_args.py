def set_granular_auto_args(args):
    # check everything not required is set
    assert (
        args.batch_size_per_gpu is not None and args.granular_model_config is not None
    )

    args = set_model_config(args)
    args.gradient_accumulation_steps = int(8 / args.n_gpus)
    args.batch_size = (
        args.n_gpus * args.gradient_accumulation_steps * args.batch_size_per_gpu
    )
    print(
        f"Setting auto args:\nBatch size: {args.batch_size}\nN gpus: {args.n_gpus}\nN grad acc steps: {args.gradient_accumulation_steps}"
    )
    args.final_lr_fraction = 0.1
    args.final_lr_step = args.n_steps
    args.lr_warmup_steps = int(round(args.n_steps * 0.01))

    return args


def set_model_config(args):
    # assert args.granular_model_config in (
    #     "mini",
    #     "mini_8",
    #     "small",
    #     "medium",
    #     "base_4",
    #     "base",
    #     "base_14",
    # )

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
    elif args.granular_model_config == "mini_big":
        args.dmodel = 384
        args.dff = 1536
        args.n_blocks = 4
        args.n_att_heads = 6
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
    elif args.granular_model_config == "medium_big":
        args.dmodel = 640
        args.dff = 2560
        args.n_blocks = 10
        args.n_att_heads = 10
    elif args.granular_model_config == "base_4":
        args.dmodel = 768
        args.dff = 3072
        args.n_blocks = 4
        args.n_att_heads = 12
    elif args.granular_model_config == "base":
        args.dmodel = 768
        args.dff = 3072
        args.n_blocks = 12
        args.n_att_heads = 12
    elif args.granular_model_config == "base_14":
        args.dmodel = 768
        args.dff = 3072
        args.n_blocks = 14
        args.n_att_heads = 12
    elif args.granular_model_config == "flag":
        args.dmodel = 896
        args.dff = 3584
        args.n_blocks = 14
        args.n_att_heads = 14
    elif args.granular_model_config == "flag_big":
        args.dmodel = 1280
        args.dff = 5120
        args.n_blocks = 20
        args.n_att_heads = 20
    elif args.granular_model_config == "clark_medium_small":
        args.dmodel = 1536
        args.dff = 6144
        args.n_blocks = 12
        args.n_att_heads = 12
    elif args.granular_model_config == "clark_medium":
        args.dmodel = 2048
        args.dff = 8192
        args.n_blocks = 16
        args.n_att_heads = 16
    elif args.granular_model_config == "clark_big":
        args.dmodel = 2048
        args.dff = 8192
        args.n_blocks = 24
        args.n_att_heads = 16

    return args
