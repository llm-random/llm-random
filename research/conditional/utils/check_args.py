import torch

def check_args(args):
    if args.granularity_expert_config:
        print(
            "`--granularity_expert_config` is deprecated. Missing granularity arguments are now always computed automatically."
        )

    if args.mixed_precision:
        if args.mixed_precision_dtype == "float16":
            args.mixed_precision_dtype = torch.float16
        elif args.mixed_precision_dtype == "bfloat16":
            args.mixed_precision_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Unknown mixed precision dtype: {args.mixed_precision_dtype}"
            )


    if args.flash_attention:
        assert (
            args.mixed_precision
        ), "Flash attention requires mixed precision to be enabled. Please set `--mixed_precision True`."
        assert args.mixed_precision_dtype in [
            torch.bfloat16,
            torch.float16,
        ], "Flash attention requires bfloat16 or float16 precision. Please set `--mixed_precision_dtype bfloat16` or `--mixed_precision_dtype float16`."
