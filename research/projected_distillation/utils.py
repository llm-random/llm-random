from typing import Union
import numpy as np
import torch

from lizrd.core.initialization import get_init_weight


def get_var_head_projection(dm, pdm, n_heads):
    projection = torch.zeros(pdm, pdm)
    mask = torch.eye(pdm).bool()
    projection = projection.masked_fill(mask, 1)

    columns_to_chose = torch.randperm(int(pdm/n_heads))[:int(dm/n_heads)]
    mask_1d = torch.zeros(int(pdm/n_heads), dtype=torch.bool)
    mask_1d[columns_to_chose] = True

    projection = projection[:, torch.concat([mask_1d]*n_heads)]
    return projection, mask_1d


FREEZE_PARAMS_REGULES = [
    # ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    # ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.",
    ".block.residual_feedforward.layer.feedforward.ff_in.logging_ff_pre_relu.weight",
    ".block.residual_feedforward.layer.feedforward.ff_out.logging_ff_post_relu.weight",

    ".block.residual_attention.layer.attention.input_projection_q.projected_weight.weight", #ATT
    ".block.residual_attention.layer.attention.input_projection_k.projected_weight.weight",
    ".block.residual_attention.layer.attention.input_projection_v.projected_weight.weight",
    ".block.residual_attention.layer.attention.output_projection.output_projection.weight",

    # "embedding_layer.layers.0.embedding.weight", #TE
    "embedding_layer.layers.0.embedding.embedding.weight", 
    "embedding_layer.layers.1.projected_layer.pe_layer.weight", #PE

    "head.head.weight", #Head
]

FF_PARAMS_BLACKLIST = [
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.",
]

def freeze_projected_params(model, unprojected_ff):
    frozen_modules = []
    for name, param in model.named_parameters():
        if unprojected_ff and any([reg in name for reg in FF_PARAMS_BLACKLIST]):  # Check if the parameter belongs to layer1
            continue
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

def is_in_partial_list(elemen_name:str, partials_list:list[str]):
    for weight_name in partials_list:
        if weight_name in elemen_name:
            return True
    return False

def print_dict_hierarchy(d, indent=0): #dev debug
    """Recursively print dictionary keys hierarchically."""
    for key, value in d.items():
        print(' ' * indent + str(key))
        if isinstance(value, dict):
            print_dict_hierarchy(value, indent + 2)

def svd_init_truncated_sv(weight:torch.Tensor, dmodel, projected_dm):
    u, s, v = torch.svd(weight)
    s = torch.diag(s)
    assert u.shape[0] == u.shape[1] == s.shape[0] == s.shape[1] == v.shape[0] == v.shape[1] == projected_dm
    
    projection_in = u[:dmodel, :] # f.e. projected_dm=512, than shape=[512, 256]
    projection_out = v[:dmodel, :] # f.e. projected_dm=512, than shape=[256, 512]
    projected_weight = s # f.e. projected_dm=512, than shape=[512, 512]
    
    return projection_in, projected_weight, projection_out

def add_projections(parameters:dict[str, torch.Tensor], projection, projection_t, projection_subnames, projection_t_subnames):
    for name, params in parameters.items():
        if is_in_partial_list(name, projection_subnames):
            # projection
            print(f"projection: {name}, {params.shape}, {params.requires_grad}")
            params.data.copy_(projection)
            # params.data = projection #dev coupled 
        elif is_in_partial_list(name, projection_t_subnames):
            # projection_T
            print(f"projection_T: {name}, {params.shape}, {params.requires_grad}")
            params.data.copy_(projection_t)
            # params.data = projection.T #dev coupled 
            # params.data.copy_(torch.inverse(projection).T) #dev inverted_test
            # params.data.copy_(torch.inverse(projection)) #dev inverted_test
        else:
            print(f"Not projection: {name}, {params.shape}, {params.requires_grad}")

def initialize_compressor(model:torch.nn.Module, projected_weights:dict, dmodel:int, projected_dmodel:int, n_att_heads:int, projection:Union[torch.Tensor, str], projection_mask:torch.Tensor):
    print(list(projected_weights.keys()))

    weight_dependent_projections = None

    if projection is None:
        print("No projection initialization")
        return
    elif projection in ["svd", "shared_block"]:
        weight_dependent_projections = projection
        projection = None
    
    embedding_layer_tag = "embedding_layer."
    head_tag = "head."
    encode_block_tag = "encoder.blocks.block_"
    model_grouped = {
        embedding_layer_tag: {},
        head_tag: {},
        encode_block_tag: {
        }
    }
    
    for name, params in model.named_parameters():
        if embedding_layer_tag == name[:len(embedding_layer_tag)]:
            model_grouped[embedding_layer_tag][name[len(embedding_layer_tag):]] = params
            continue
        if head_tag == name[:len(head_tag)]:
            model_grouped[head_tag][name[len(head_tag):]] = params
            continue
        if encode_block_tag == name[:len(encode_block_tag)]:
            parsed_name = name[len(encode_block_tag):].split('.')
            block_number = int(parsed_name[0])
            block_component_name = ".".join(parsed_name[1:])
            if model_grouped[encode_block_tag].get(str(block_number)) is None:
                model_grouped[encode_block_tag][str(block_number)] = {}
            model_grouped[encode_block_tag][str(block_number)][block_component_name] = params
            continue
        raise Exception(f"Could not parse model into expected template, unexpected name: name")
        
    print_dict_hierarchy(model_grouped, 3) #dev

    print("------------------------------init projections------------------------") #dev

    EMBEDDING_P = []
    EMBEDDING_P_T = [
        "layers.0.embedding.embedding_p.weight", 
        "layers.1.projected_layer.pe_layer_p.weight", ]
    DEEMBEDDING_P = [
        "head.head_p.weight",]
    DEEMBEDDING_P_T = []
    BLOCK_P = [
        "block.residual_attention.layer.attention.input_projection_q.input_projection.weight",
        "block.residual_attention.layer.attention.input_projection_k.input_projection.weight",
        "block.residual_attention.layer.attention.input_projection_v.input_projection.weight",
        # "block.residual_attention.layer.attention.input_projection.input_projection.weight", #dev switch
        "block.residual_attention.layer.attention.output_projection.output_projection_p21.weight",
        # "block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p11.weight",
        # "block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.weight",
        "block.residual_feedforward.layer.feedforward.ff_in.logging_ff_pre_relu_p11.weight",
        "block.residual_feedforward.layer.feedforward.ff_out.logging_ff_post_relu_p21.weight",
    ]
    BLOCK_P_T = [
        "block.residual_attention.layer.attention.input_projection_q.output_projection.weight",
        "block.residual_attention.layer.attention.input_projection_k.output_projection.weight",
        "block.residual_attention.layer.attention.input_projection_v.output_projection.weight",
        # "block.residual_attention.layer.attention.input_projection_out.output_projection.weight", #dev switch
        "block.residual_attention.layer.attention.output_projection.output_projection_p22.weight",
        # "block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.weight",
        # "block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p22.weight",
        "block.residual_feedforward.layer.feedforward.ff_in.logging_ff_pre_relu_p12.weight",
        "block.residual_feedforward.layer.feedforward.ff_out.logging_ff_post_relu_p22.weight",
    ]

    if not weight_dependent_projections:
        add_projections(model_grouped[embedding_layer_tag], projection,  projection.T, EMBEDDING_P, EMBEDDING_P_T)
        add_projections(model_grouped[head_tag], projection,  projection.T, DEEMBEDDING_P, DEEMBEDDING_P_T)
        for block_id, block_params in model_grouped[encode_block_tag].items():
            print(f"Block: {block_id}")
            add_projections(block_params, projection,  projection.T, BLOCK_P, BLOCK_P_T)
    elif weight_dependent_projections == "shared_block":
        projection, mask_1d = get_var_head_projection(dmodel, projected_dmodel, n_att_heads)
        add_projections(model_grouped[embedding_layer_tag], projection,  projection.T, EMBEDDING_P, EMBEDDING_P_T)
        add_projections(model_grouped[head_tag], projection,  projection.T, DEEMBEDDING_P, DEEMBEDDING_P_T)
        for block_id, block_params in model_grouped[encode_block_tag].items():
            projection, mask_1d = get_var_head_projection(dmodel, projected_dmodel, n_att_heads)
            print(f"Block: {block_id}")
            add_projections(block_params, projection,  projection.T, BLOCK_P, BLOCK_P_T)
    elif weight_dependent_projections == "svd":
        projection = torch.zeros(projected_dmodel, projected_dmodel)
        mask = torch.eye(projected_dmodel).bool()
        projection = projection.masked_fill(mask, 1)
        projection = projection[:, projection_mask]

        add_projections(model_grouped[embedding_layer_tag], projection,  projection.T, EMBEDDING_P, EMBEDDING_P_T)
        add_projections(model_grouped[head_tag], projection,  projection.T, DEEMBEDDING_P, DEEMBEDDING_P_T)
        for block_id, block_params in model_grouped[encode_block_tag].items():
            print(f"Preinit block: {block_id}")
            add_projections(block_params, projection,  projection.T, BLOCK_P, BLOCK_P_T)
    
        for block_id, block_params in model_grouped[encode_block_tag].items():
            print(f"Block: {block_id}")

            # att_output_proj_w = block_params["block.residual_attention.layer.attention.output_projection.output_projection.weight"]
            # u, s, v = svd_init_truncated_sv(att_output_proj_w, dmodel, projected_dmodel)
            # block_params["block.residual_attention.layer.attention.output_projection.output_projection_p21.weight"].data.copy_(u.T)
            # block_params["block.residual_attention.layer.attention.output_projection.output_projection.weight"].data.copy_(s)
            # block_params["block.residual_attention.layer.attention.output_projection.output_projection_p22.weight"].data.copy_(v)

            # ff_pre_relu_w = block_params["block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.weight"]
            # u, s, v = svd_init_truncated_sv(ff_pre_relu_w, dmodel, projected_dmodel)
            # block_params["block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p11.weight"].data.copy_(u.T)
            # block_params["block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.weight"].data.copy_(s)
            # block_params["block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.weight"].data.copy_(v)

            # ff_post_relu_w = block_params["block.residual_feedforward.layer.feedforward.logging_ff_post_relu.weight"]
            # u, s, v = svd_init_truncated_sv(ff_post_relu_w, dmodel, projected_dmodel)
            # block_params["block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.weight"].data.copy_(u.T)
            # block_params["block.residual_feedforward.layer.feedforward.logging_ff_post_relu.weight"].data.copy_(s)
            # block_params["block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p22.weight"].data.copy_(v)
    elif weight_dependent_projections == "shared_att_in":
        projection = torch.zeros(projected_dmodel, projected_dmodel)
        mask = torch.eye(projected_dmodel).bool()
        projection = projection.masked_fill(mask, 1)
        projection = projection[:, projection_mask]

        add_projections(model_grouped[embedding_layer_tag], projection,  projection.T, EMBEDDING_P, EMBEDDING_P_T)
        add_projections(model_grouped[head_tag], projection,  projection.T, DEEMBEDDING_P, DEEMBEDDING_P_T)

        for block_id, block_params in model_grouped[encode_block_tag].items():
            print(f"Block: {block_id}")

            block_params["block.residual_attention.layer.attention.input_projection_q.input_projection.weight"].data.copy_(projection) # proj
            block_proj = block_params["block.residual_attention.layer.attention.input_projection_q.input_projection.weight"]
            block_params["block.residual_attention.layer.attention.input_projection_q.output_projection.weight"].data.copy_(projection.T) # proj.T
            # block_params["block.residual_attention.layer.attention.input_projection_q.output_projection.weight"].data = block_proj.T # proj.T #dev switch - input-output entanglment
            block_proj_t = block_params["block.residual_attention.layer.attention.input_projection_q.output_projection.weight"]
            
            block_params["block.residual_attention.layer.attention.input_projection_k.input_projection.weight"].data = block_proj #dev switch - input entanglment
            block_params["block.residual_attention.layer.attention.input_projection_v.input_projection.weight"].data = block_proj
            # block_params["block.residual_attention.layer.attention.input_projection_k.input_projection.weight"].data.copy_(block_proj)
            # block_params["block.residual_attention.layer.attention.input_projection_v.input_projection.weight"].data.copy_(block_proj)

            # block_params["block.residual_attention.layer.attention.input_projection_k.output_projection.weight"].data = block_proj_t #dev switch - output entanglment
            # block_params["block.residual_attention.layer.attention.input_projection_v.output_projection.weight"].data = block_proj_t
            block_params["block.residual_attention.layer.attention.input_projection_k.output_projection.weight"].data.copy_(block_proj_t)
            block_params["block.residual_attention.layer.attention.input_projection_v.output_projection.weight"].data.copy_(block_proj_t)

            block_params["block.residual_attention.layer.attention.output_projection.output_projection_p21.weight"].data.copy_(block_proj)
            block_params["block.residual_attention.layer.attention.output_projection.output_projection_p22.weight"].data.copy_(block_proj_t)

            block_params["block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p11.weight"].data.copy_(block_proj)
            block_params["block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.weight"].data.copy_(block_proj)
            block_params["block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.weight"].data.copy_(block_proj_t)
            block_params["block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p22.weight"].data.copy_(block_proj_t)
    elif weight_dependent_projections == "shared_att_out":
        projection = torch.zeros(projected_dmodel, projected_dmodel)
        mask = torch.eye(projected_dmodel).bool()
        projection = projection.masked_fill(mask, 1)
        projection = projection[:, projection_mask]

        add_projections(model_grouped[embedding_layer_tag], projection,  projection.T, EMBEDDING_P, EMBEDDING_P_T)
        add_projections(model_grouped[head_tag], projection,  projection.T, DEEMBEDDING_P, DEEMBEDDING_P_T)

        for block_id, block_params in model_grouped[encode_block_tag].items():
            print(f"Block: {block_id}")

            block_params["block.residual_attention.layer.attention.input_projection_q.input_projection.weight"].data.copy_(projection) # proj
            block_proj = block_params["block.residual_attention.layer.attention.input_projection_q.input_projection.weight"]
            block_params["block.residual_attention.layer.attention.input_projection_q.output_projection.weight"].data.copy_(projection.T) # proj.T
            # block_params["block.residual_attention.layer.attention.input_projection_q.output_projection.weight"].data = block_proj.T # proj.T #dev switch - input-output entanglment
            block_proj_t = block_params["block.residual_attention.layer.attention.input_projection_q.output_projection.weight"]
            
            # block_params["block.residual_attention.layer.attention.input_projection_k.input_projection.weight"].data = block_proj #dev switch - input entanglment
            # block_params["block.residual_attention.layer.attention.input_projection_v.input_projection.weight"].data = block_proj
            block_params["block.residual_attention.layer.attention.input_projection_k.input_projection.weight"].data.copy_(block_proj)
            block_params["block.residual_attention.layer.attention.input_projection_v.input_projection.weight"].data.copy_(block_proj)

            block_params["block.residual_attention.layer.attention.input_projection_k.output_projection.weight"].data = block_proj_t #dev switch - output entanglment
            block_params["block.residual_attention.layer.attention.input_projection_v.output_projection.weight"].data = block_proj_t
            # block_params["block.residual_attention.layer.attention.input_projection_k.output_projection.weight"].data.copy_(block_proj_t)
            # block_params["block.residual_attention.layer.attention.input_projection_v.output_projection.weight"].data.copy_(block_proj_t)

            block_params["block.residual_attention.layer.attention.output_projection.output_projection_p21.weight"].data.copy_(block_proj)
            block_params["block.residual_attention.layer.attention.output_projection.output_projection_p22.weight"].data.copy_(block_proj_t)

            block_params["block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p11.weight"].data.copy_(block_proj)
            block_params["block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.weight"].data.copy_(block_proj)
            block_params["block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.weight"].data.copy_(block_proj_t)
            block_params["block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p22.weight"].data.copy_(block_proj_t)
    elif weight_dependent_projections == "shared_att_in_out":
        projection = torch.zeros(projected_dmodel, projected_dmodel)
        mask = torch.eye(projected_dmodel).bool()
        projection = projection.masked_fill(mask, 1)
        projection = projection[:, projection_mask]

        add_projections(model_grouped[embedding_layer_tag], projection,  projection.T, EMBEDDING_P, EMBEDDING_P_T)
        add_projections(model_grouped[head_tag], projection,  projection.T, DEEMBEDDING_P, DEEMBEDDING_P_T)

        for block_id, block_params in model_grouped[encode_block_tag].items():
            print(f"Block: {block_id}")

            block_params["block.residual_attention.layer.attention.input_projection_q.input_projection.weight"].data.copy_(projection) # proj
            block_proj = block_params["block.residual_attention.layer.attention.input_projection_q.input_projection.weight"]
            block_params["block.residual_attention.layer.attention.input_projection_q.output_projection.weight"].data.copy_(projection.T) # proj.T
            # block_params["block.residual_attention.layer.attention.input_projection_q.output_projection.weight"].data = block_proj.T # proj.T #dev switch - input-output entanglment
            block_proj_t = block_params["block.residual_attention.layer.attention.input_projection_q.output_projection.weight"]
            
            block_params["block.residual_attention.layer.attention.input_projection_k.input_projection.weight"].data = block_proj #dev switch - input entanglment
            block_params["block.residual_attention.layer.attention.input_projection_v.input_projection.weight"].data = block_proj
            # block_params["block.residual_attention.layer.attention.input_projection_k.input_projection.weight"].data.copy_(block_proj)
            # block_params["block.residual_attention.layer.attention.input_projection_v.input_projection.weight"].data.copy_(block_proj)

            block_params["block.residual_attention.layer.attention.input_projection_k.output_projection.weight"].data = block_proj_t #dev switch - output entanglment
            block_params["block.residual_attention.layer.attention.input_projection_v.output_projection.weight"].data = block_proj_t
            # block_params["block.residual_attention.layer.attention.input_projection_k.output_projection.weight"].data.copy_(block_proj_t)
            # block_params["block.residual_attention.layer.attention.input_projection_v.output_projection.weight"].data.copy_(block_proj_t)

            block_params["block.residual_attention.layer.attention.output_projection.output_projection_p21.weight"].data.copy_(block_proj)
            block_params["block.residual_attention.layer.attention.output_projection.output_projection_p22.weight"].data.copy_(block_proj_t)

            block_params["block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p11.weight"].data.copy_(block_proj)
            block_params["block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.weight"].data.copy_(block_proj)
            block_params["block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.weight"].data.copy_(block_proj_t)
            block_params["block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p22.weight"].data.copy_(block_proj_t)        
    else:
        raise ValueError(f"Invalid `weight_dependent_projections` = {weight_dependent_projections}")
    print("------------------------------init projections end------------------------") #dev
    print("------------------------------copy weights start------------------------") #dev

    # "{partial_name_of_compressor_weight}": {transformations}
    EMBEDDING_TRANSFER = {
        "layers.0.embedding.embedding.weight": [("layers.0.embedding.embedding.weight", "embedding_layer.layers.0.weight")],
        "layers.1.projected_layer.pe_layer.weight": [("layers.1.projected_layer.pe_layer.weight", "embedding_layer.layers.1.layer.weight")],
        "head.head.weight": [("head.head.weight","head.weight")],
    }

    embedding_params = {}
    embedding_params.update(model_grouped[embedding_layer_tag])
    embedding_params.update(model_grouped[head_tag])

    for name, params in embedding_params.items():
        print(f"Considering: {name}, {params.shape}") #dev
        for k, transfer_rules in EMBEDDING_TRANSFER.items():
            if k in name:
                copied_params_name = name
                for transfer_tuple in transfer_rules:
                    copied_params_name = copied_params_name.replace(transfer_tuple[0], transfer_tuple[1])
                transfered_params = projected_weights.get(copied_params_name)
                print(f" --- Transferred params name: {copied_params_name}, {transfered_params.shape}") #dev
                if transfered_params is not None:
                    print(" --- TRANSFERRED")
                    params.data.copy_(transfered_params)

    for block_id, block_params in model_grouped[encode_block_tag].items():
       
        input_projections = torch.chunk(projected_weights[encode_block_tag+block_id+"."+"block.residual_attention.layer.attention.input_projection.weight"], 3, dim=0) #dev switch
        block_params.get("block.residual_attention.layer.attention.input_projection_q.projected_weight.weight").data.copy_(input_projections[0])
        block_params.get("block.residual_attention.layer.attention.input_projection_k.projected_weight.weight").data.copy_(input_projections[1])
        block_params.get("block.residual_attention.layer.attention.input_projection_v.projected_weight.weight").data.copy_(input_projections[2])
        
        # block_params.get("block.residual_attention.layer.attention.input_projection_q.projected_weight.weight").data.copy_(projected_weights[encode_block_tag+block_id+"."+"block.residual_attention.layer.attention.input_projection_q.weight"]) #dev switch
        # block_params.get("block.residual_attention.layer.attention.input_projection_k.projected_weight.weight").data.copy_(projected_weights[encode_block_tag+block_id+"."+"block.residual_attention.layer.attention.input_projection_k.weight"])
        # block_params.get("block.residual_attention.layer.attention.input_projection_v.projected_weight.weight").data.copy_(projected_weights[encode_block_tag+block_id+"."+"block.residual_attention.layer.attention.input_projection_v.weight"])

        block_params.get("block.residual_attention.layer.attention.output_projection.output_projection.weight").data.copy_(projected_weights[encode_block_tag+block_id+"."+"block.residual_attention.layer.attention.output_projection.weight"])
        
        ff_in = block_params.get("block.residual_feedforward.layer.feedforward.ff_in.logging_ff_pre_relu.weight")
        ff_out = block_params.get("block.residual_feedforward.layer.feedforward.ff_out.logging_ff_post_relu.weight")
        
        ff_in.data.copy_(projected_weights[encode_block_tag+block_id+"."+"block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.weight"])
        ff_out.data.copy_(projected_weights[encode_block_tag+block_id+"."+"block.residual_feedforward.layer.feedforward.logging_ff_post_relu.weight"])
        
        
        # print(f'{block_id}, 3 x {projected_weights[encode_block_tag+block_id+"."+"block.residual_attention.layer.attention.input_projection_q.weight"]}, {projected_weights[encode_block_tag+block_id+"."+"block.residual_attention.layer.attention.output_projection.weight"].shape}, {projected_weights[encode_block_tag+block_id+"."+"block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.weight"].shape}, {projected_weights[encode_block_tag+block_id+"."+"block.residual_feedforward.layer.feedforward.logging_ff_post_relu.weight"].shape}') #dev
        
    print("------------------------------copy weights end------------------------") #dev
