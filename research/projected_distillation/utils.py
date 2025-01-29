import numpy as np
import torch

from lizrd.core.initialization import get_init_weight

FREEZE_PARAMS_REGULES = [
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.",

    ".block.residual_attention.layer.attention.input_projection.input_projection.weight", #ATT
    ".block.residual_attention.layer.attention.output_projection.output_projection.weight",

    "embedding_layer.layers.0.embedding.weight", #TE
    "embedding_layer.layers.1.projected_layer.pe_layer.weight", #PE

    "head.head.weight", #Head

    "head.weight",
    "embedding_layer.layers.0.weight",
    "embedding_layer.layers.1.layer.weight",
]

def freeze_projected_params(model):
    frozen_modules = []
    for name, param in model.named_parameters():
        if any([reg in name for reg in FREEZE_PARAMS_REGULES]):  # Check if the parameter belongs to layer1
            param.requires_grad = False
            frozen_modules.append(param)
    return frozen_modules


FREEZE_LN_REGULES = [
    ".pre_norm.", # Layer norm
]

def freeze_ln_params(model):
    frozen_modules = []
    for name, param in model.named_parameters():
        if any([reg in name for reg in FREEZE_LN_REGULES]):  # Check if the parameter belongs to layer1
            param.requires_grad = False
            frozen_modules.append(param)
    return frozen_modules 


PROJECTIONS_1_1 = [
    ".block.residual_attention.layer.attention.input_projection.input_projection_p11.weight",
    ".block.residual_attention.layer.attention.output_projection.output_projection_p21.weight",
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p11.weight",
    "head.head_p.weight",
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.weight", #FF in - 1ff configuration
]

PROJECTIONS_1_1_T = [
    # "embedding_layer.layers.0.embedding_p.weight", #dev inverted_test
    # "embedding_layer.layers.1.projected_layer.pe_layer_p.weight", #dev inverted_test
    ".block.residual_attention.layer.attention.output_projection.output_projection_p22.weight",
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p22.weight",
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.weight", #FF out - 1ff configuration
]

PROJECTIONS_1_4 = [
    # ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.weight",
]

PROJECTIONS_1_4_T = [
    # ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.weight",
]

PROJECTIONS_1_3 = [

]

PROJECTIONS_1_3_T = [
    ".block.residual_attention.layer.attention.input_projection.input_projection_p12.weight",
]
# encoder.blocks.block_7.block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p11.weight, shape: torch.Size([512, 256]) requires_grad: True, cuda:0
# encoder.blocks.block_7.block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.weight, shape: torch.Size([256, 512]) requires_grad: True, cuda:0
# encoder.blocks.block_7.block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.weight, shape: torch.Size([512, 256]) requires_grad: True, cuda:0
# encoder.blocks.block_7.block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p22.weight, shape: torch.Size([256, 512]) requires_grad: True, cuda:0
MULTIPLY = [

]

MULTIPLY_T = [

]


def is_in_partial_list(elemen_name:str, partials_list:list[str]):
    for weight_name in partials_list:
        if weight_name in elemen_name:
            return True
    return False

def initialize_projections(model:torch.nn.Module, dmodel:int, projected_dmodel:int, projection:torch.Tensor, diagonal=True):
    if projection is None:
        print("No projection initialization")
        return
    # if not init_type:
    #     print("No projection initialization")
    #     return
    # elif init_type == "random":
    #     print("Random projection initialization")
    #     projection = get_init_weight(
    #         shape=(projected_dmodel, dmodel),
    #         fan_in=1,  # fan_in=1 is also default in pytorch
    #         init_type="truncated_normal",
    #         scale=init_scale/projected_dmodel,
    #     )
    #     projection_z = torch.zeros((projected_dmodel, dmodel))
    # else:
    #     raise Exception("Wrong projection init type")

    projection_3 = torch.concat((projection, projection, projection))
    projection_3 = torch.concat((projection_3, projection_3, projection_3), dim=1)
    projection_3.shape

    if diagonal:
        projection_z = torch.zeros((projected_dmodel, dmodel))
        projection_4 = torch.concat((
            torch.concat((projection, projection_z, projection_z, projection_z), dim=0),
            torch.concat((projection_z, projection, projection_z, projection_z), dim=0),
            torch.concat((projection_z, projection_z, projection, projection_z), dim=0),
            torch.concat((projection_z, projection_z, projection_z, projection), dim=0),
            ), dim=1)
        # projection_T = projection.T
        # projection_z_T = projection_z.T
        # projection_4_T = torch.concat((
        #     torch.concat((projection_T, projection_z_T, projection_z_T, projection_z_T), dim=0),
        #     torch.concat((projection_z_T, projection_T, projection_z_T, projection_z_T), dim=0),
        #     torch.concat((projection_z_T, projection_z_T, projection_T, projection_z_T), dim=0),
        #     torch.concat((projection_z_T, projection_z_T, projection_z_T, projection_T), dim=0),
        #     ), dim=1)
        projection_4_T = projection_4.T
    else:
        print("NO DIAGONAL INIT") #dev
        projection_4 = torch.concat((projection, projection, projection, projection))
        projection_4 = torch.concat((projection_4, projection_4, projection_4, projection_4), dim=1)
        projection_4_T = projection_4.T

    print("------------------------------init projections------------------------") #dev
    for name, params in model.named_parameters():
        if is_in_partial_list(name, PROJECTIONS_1_1):
            # projection
            print(f"projection: {name}, {params.shape}")
            eye = torch.eye(params.data.shape[0])
            params.data.copy_(eye)
        elif is_in_partial_list(name, PROJECTIONS_1_1_T):
            # projection_T
            print(f"projection_T: {name}, {params.shape}")
            eye = torch.eye(params.data.shape[0])
            params.data.copy_(eye)
        elif is_in_partial_list(name, PROJECTIONS_1_4):
            # projection_4
            print(f"projection_4: {name}, {params.shape}")
            params.data.copy_(projection_4)
        elif is_in_partial_list(name, PROJECTIONS_1_4_T):
            # projection_4_T
            print(f"projection_4_T: {name}, {params.shape}")
            # params.data.copy_(projection_4.T)
            params.data.copy_(projection_4_T)
        elif is_in_partial_list(name, PROJECTIONS_1_3):
            # projection_3
            print(f"projection_3: {name}, {params.shape}")
            raise NotImplemented()
            params.data.copy_(projection_3)
        elif is_in_partial_list(name, PROJECTIONS_1_3_T):
            # projection_3_T
            print(f"projection_3_T: {name}, {params.shape}")
            eye = torch.eye(params.data.shape[0])
            params.data.copy_(eye)
        elif is_in_partial_list(name, MULTIPLY):
            # projection
            print(f"projection: {name}, {params.shape}")
            raise NotImplemented()
            params.data.copy_(projection)
        elif is_in_partial_list(name, MULTIPLY_T):
            # projection
            print(f"projection: {name}, {params.shape}")
            raise NotImplemented()
            params.data.copy_(projection)
        else:
            print(f"Not projection: {name}, {params.shape}, {params.requires_grad}")
    print("------------------------------init projections end------------------------") #dev