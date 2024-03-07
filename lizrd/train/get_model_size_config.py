def get_model_size_config(predefined_config):
    if predefined_config == "mini":
        dmodel = 256
        dff = 1024
        n_att_heads = 4
        n_blocks = 4
    elif predefined_config == "small":
        dmodel = 512
        dff = 2048
        n_blocks = 4
        n_att_heads = 8
    elif predefined_config == "medium":
        dmodel = 512
        dff = 2048
        n_blocks = 8
        n_att_heads = 8
    elif predefined_config == "base":
        dmodel = 768
        dff = 3072
        n_blocks = 12
        n_att_heads = 12
    else:
        raise ValueError(f"Unknown predefined config: {predefined_config}")

    return dmodel, dff, n_att_heads, n_blocks
