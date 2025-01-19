import torch

TRANSFER_PARAMS = [
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.", #FF

    ".block.residual_attention.layer.attention.input_projection.weight", #ATT
    ".block.residual_attention.layer.attention.output_projection.weight", #ATT

    "embedding_layer.layers.0.weight", #TE
    "embedding_layer.layers.1.layer.weight" #PE

    "head.weight", #Head
]

CAST_PROJECTED_PARAMS_NAME_PARTS = [
    (".output_projection.output_projection.", ".output_projection."), #ATT
    (".input_projection.input_projection.", ".input_projection."), #ATT
    ("embedding_layer.layers.0.embedding.weight", "embedding_layer.layers.0.weight"), #TE
    ("head.head.weight", "head.weight"), #Head
]


def load_projected_weights(model:torch.nn.Module, projected_weights):
    print(list(projected_weights.keys())) #dev
    print("------------------------------replace with new values------------------------") #dev
    for name, params in model.named_parameters():
        for e in CAST_PROJECTED_PARAMS_NAME_PARTS:
            if e[0] in name:
                name = name.replace(e[0], e[1])
                # print("replaced name", name) #dev
        prj_params = projected_weights.get(name)
        if (prj_params is not None) and any([reg in name for reg in TRANSFER_PARAMS]):
            params.data.copy_(prj_params)
            print(f"REPLACED: {name}")
    print("------------------------------replace with new values end------------------------") #dev

