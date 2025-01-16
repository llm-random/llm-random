import torch

TRANSFER_PARAMS = [
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.",

    ".block.residual_attention.layer.attention.input_projection.input_projection.weight", #ATT
    ".block.residual_attention.layer.attention.output_projection.input_projection.weight",
]

def load_projected_weights(model:torch.nn.Module, projected_weights):
    print("----------------replace with new values----------------") #dev
    for name, params in model.named_parameters():
        prj_params = projected_weights.get(name)
        if (prj_params is not None) and any([reg in name for reg in TRANSFER_PARAMS]):
            print(name) #dev
            params.data.copy_(prj_params)
    print("--------------end------------------") #dev