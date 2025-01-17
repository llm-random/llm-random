import torch

TRANSFER_PARAMS = [
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.",

    ".block.residual_attention.layer.attention.input_projection.weight", #ATT
    ".block.residual_attention.layer.attention.output_projection.weight",
]

CAST_PROJECTED_PARAMS_NAME_PARTS = [
    (".output_projection.output_projection.", ".output_projection."),
    (".input_projection.input_projection.", ".input_projection.")
]


def load_projected_weights(model:torch.nn.Module, projected_weights):
    # print(list(projected_weights["model"].keys())) #dev
    # print("----------------replace with new values----------------") #dev
    for name, params in model.named_parameters():
        for e in CAST_PROJECTED_PARAMS_NAME_PARTS:
            if e[0] in name:
                name = name.replace(e[0], e[1])
                # print("replaced name", name) #dev
        prj_params = projected_weights.get(name)
        # print(name, type(prj_params))
        if (prj_params is not None) and any([reg in name for reg in TRANSFER_PARAMS]):
            print(name) #dev
            params.data.copy_(prj_params)
    print("--------------end------------------") #dev
