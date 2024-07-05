def get_attention_layer(args):
    if args.blanks_use_custom_attention and args.n_blanks > 0:
        attention_layer_fun = lambda: BlankAttention(
            dmodel=args.dmodel,
            heads=args.n_att_heads,
            dhead=args.dhead,
            flash=False,
            init_type=args.init_type,
            init_scale=args.init_scale,
        )
    else:
        attention_layer_fun = lambda: llm.Attention(
            dmodel=args.dmodel,
            heads=args.n_att_heads,
            dhead=args.dhead,
            flash=True,
            causal=True,
            init_type=args.init_type,
            init_scale=args.init_scale,
        )
    return attention_layer_fun
