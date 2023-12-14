def set_granular_auto_args(args):
    # check everything not required is set
    assert args.batch_size_per_gpu is not None

    args.gradient_accumulation_steps = 8 / args.n_gpus
    args.final_lr_fraction = 0.1
    args.final_lr_step = args.n_steps
    args.lr_warmup_steps = round(args.n_steps * 0.01)
