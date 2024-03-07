def get_model_size_config(predefined_config):
    if predefined_config == "mini":
        dmodel = 256
        dff = 1024
        n_att_heads = 4
        n_blocks = 4
    else:
        raise ValueError(f"Unknown predefined config: {predefined_config}")

    return dmodel, dff, n_att_heads, n_blocks
