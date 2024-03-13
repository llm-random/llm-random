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
    elif predefined_config == "base_bigger":
        dmodel = 1024
        dff = 4096
        n_blocks = 16
        n_att_heads = 16
    elif predefined_config == "flag":
        dmodel = 896
        dff = 3584
        n_blocks = 14
        n_att_heads = 14
    elif predefined_config == "flag_big":
        dmodel = 1280
        dff = 5120
        n_blocks = 20
        n_att_heads = 20
    elif predefined_config == "clark_medium_small":
        dmodel = 1536
        dff = 6144
        n_blocks = 12
        n_att_heads = 12
    elif predefined_config == "big_between":
        dmodel = 1792
        dff = 7168
        n_blocks = 14
        n_att_heads = 14
    elif predefined_config == "clark_medium":
        dmodel = 2048
        dff = 8192
        n_blocks = 16
        n_att_heads = 16
    elif predefined_config == "clark_big_smaller":
        dmodel = 1792
        dff = 7168
        n_blocks = 23
        n_att_heads = 14
    elif predefined_config == "clark_big":
        dmodel = 2048
        dff = 8192
        n_blocks = 24
        n_att_heads = 16
    elif predefined_config == "flag_dense":
        dmodel = 3072
        dff = 12288
        n_blocks = 36
        n_att_heads = 24
    else:
        raise ValueError(f"Unknown predefined config: {predefined_config}")

    return dmodel, dff, n_att_heads, n_blocks
