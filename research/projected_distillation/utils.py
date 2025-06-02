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

# Compressor model names {block}.{Name}

P_ATT_Q_1 = "block.residual_attention.layer.attention.input_projection_q.input_projection.weight"
P_ATT_Q = "block.residual_attention.layer.attention.input_projection_q.projected_weight.weight"
P_ATT_Q_2 = "block.residual_attention.layer.attention.input_projection_q.output_projection.weight"
P_ATT_Q_R = "block.residual_attention.layer.attention.input_projection_q_res.weight"

P_ATT_K_1 = "block.residual_attention.layer.attention.input_projection_k.input_projection.weight"
P_ATT_K = "block.residual_attention.layer.attention.input_projection_k.projected_weight.weight"
P_ATT_K_2 = "block.residual_attention.layer.attention.input_projection_k.output_projection.weight"
P_ATT_K_R = "block.residual_attention.layer.attention.input_projection_r_res.weight"

P_ATT_V_1 = "block.residual_attention.layer.attention.input_projection_v.input_projection.weight"
P_ATT_V = "block.residual_attention.layer.attention.input_projection_v.projected_weight.weight"
P_ATT_V_2 = "block.residual_attention.layer.attention.input_projection_v.output_projection.weight"
P_ATT_V_R = "block.residual_attention.layer.attention.input_projection_v_res.weight"


P_ATT_OUT_1 = "block.residual_attention.layer.attention.output_projection.output_projection_p21.weight"
P_ATT_OUT = "block.residual_attention.layer.attention.output_projection.output_projection.weight"
P_ATT_OUT_2 = "block.residual_attention.layer.attention.output_projection.output_projection_p22.weight"
P_ATT_OUT_R = "block.residual_attention.layer.attention.output_projection_res.weight"


P_FF_IN_1 = "block.residual_feedforward.layer.feedforward.ff_in.logging_ff_pre_relu_p11.weight"
P_FF_IN = "block.residual_feedforward.layer.feedforward.ff_in.logging_ff_pre_relu.weight"
P_FF_IN_2 = "block.residual_feedforward.layer.feedforward.ff_in.logging_ff_pre_relu_p12.weight"
P_FF_IN_R = "block.residual_feedforward.layer.feedforward.ff_in_res.weight"

P_FF_OUT_1 = "block.residual_feedforward.layer.feedforward.ff_out.logging_ff_post_relu_p21.weight"
P_FF_OUT = "block.residual_feedforward.layer.feedforward.ff_out.logging_ff_post_relu.weight"
P_FF_OUT_2 = "block.residual_feedforward.layer.feedforward.ff_out.logging_ff_post_relu_p22.weight"
P_FF_OUT_R = "block.residual_feedforward.layer.feedforward.ff_out_res.weight"

P_EMB_PE = "embedding_layer.layers.1.projected_layer.pe_layer.weight"
P_EMB = "layers.0.embedding.embedding.weight"
P_EMB_2 = "layers.0.embedding.embedding_p.weight"
P_EMB_R = "layers.0.embedding_res.weight"

P_HEAD = "head.head.weight"
P_HEAD_1 = "head.head_p.weight"
P_HEAD_R = "head_res.weight"


# Default model names {block}.{Name}
T_ATT_Q = "block.residual_attention.layer.attention.input_projection_q.weight"
T_ATT_K = "block.residual_attention.layer.attention.input_projection_k.weight"
T_ATT_V = "block.residual_attention.layer.attention.input_projection_v.weight"
T_ATT_OUT = "block.residual_attention.layer.attention.output_projection.weight"
T_FF_IN = "block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.weight"
T_FF_OUT = "block.residual_feedforward.layer.feedforward.logging_ff_post_relu.weight"
T_EMB = "embedding_layer.layers.0.weight"
T_HEAD = "head.unembedding.head.weight"


FREEZE_PARAMS_REGULES = [
    P_FF_IN, #FF
    P_FF_OUT,
    P_ATT_Q, #ATT
    P_ATT_K,
    P_ATT_V,
    P_ATT_OUT,
    P_EMB, #EMB
    P_HEAD,#HED
    P_EMB_PE, #PEM
]

EMBEDDING_LAYER_TAG = "embedding_layer."
HEAD_TAG = "head."
ENCODE_BLOCK_TAG = "encoder.blocks.block_"

def freeze_projected_params(model, unprojected_ff):
    frozen_modules = []
    for name, param in model.named_parameters():
        if unprojected_ff and any([reg in name for reg in [P_FF_IN, P_FF_OUT]]):  # Check if the parameter belongs to layer1
            continue
        if any([reg in name for reg in FREEZE_PARAMS_REGULES]):  # Check if the parameter belongs to layer1
            param.requires_grad = False
            frozen_modules.append(param)
    return frozen_modules

FREEZE_LN_REGULES = [
    ".pre_norm.", # Layer norm
    ".unembedding.head_norm.weight", # Pre head ayer norm
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
            # params.data = projection #dev switch coupled 
        elif is_in_partial_list(name, projection_t_subnames):
            # projection_T
            print(f"projection_T: {name}, {params.shape}, {params.requires_grad}")
            params.data.copy_(projection_t)
            # params.data = projection.T #dev switch coupled  
            # params.data.copy_(torch.inverse(projection).T) #dev switch inverted_test
            # params.data.copy_(torch.inverse(projection)) #dev switch inverted_test
        else:
            print(f"Not projection: {name}, {params.shape}, {params.requires_grad}")
        

def initialize_compressor(model:torch.nn.Module, projected_weights:dict, dmodel:int, projected_dmodel:int, n_att_heads:int, projection:Union[torch.Tensor, str], projection_mask:torch.Tensor):
    print("Projected model (doner) params ------------------------------------------------------------------------------------------------------------------------")
    print(list(projected_weights.keys()))
    print("end doner ---------------------------------------------------------------------------------------------------------------------------------------------------")

    weight_dependent_projections = None

    if projection is None:
        print("No projection initialization")
        return
    elif projection in ["svd", "shared_block", "magnitude", "magnitude_global"]:
        weight_dependent_projections = projection
        projection = None
    
    model_grouped = {
        EMBEDDING_LAYER_TAG: {},
        HEAD_TAG: {},
        ENCODE_BLOCK_TAG: {}
    }
    
    for name, params in model.named_parameters():
        if EMBEDDING_LAYER_TAG == name[:len(EMBEDDING_LAYER_TAG)]:
            model_grouped[EMBEDDING_LAYER_TAG][name[len(EMBEDDING_LAYER_TAG):]] = params
            continue
        if HEAD_TAG == name[:len(HEAD_TAG)]:
            model_grouped[HEAD_TAG][name[len(HEAD_TAG):]] = params
            continue
        if ENCODE_BLOCK_TAG == name[:len(ENCODE_BLOCK_TAG)]:
            parsed_name = name[len(ENCODE_BLOCK_TAG):].split('.')
            block_number = int(parsed_name[0])
            block_component_name = ".".join(parsed_name[1:])
            if model_grouped[ENCODE_BLOCK_TAG].get(str(block_number)) is None:
                model_grouped[ENCODE_BLOCK_TAG][str(block_number)] = {}
            model_grouped[ENCODE_BLOCK_TAG][str(block_number)][block_component_name] = params
            continue
        raise Exception(f"Could not parse model into expected template, unexpected name: {name}")
        
    print_dict_hierarchy(model_grouped, 3) #dev

    print("------------------------------copy weights start------------------------") #dev
    model_grouped[EMBEDDING_LAYER_TAG].get(P_EMB).data.copy_(projected_weights.get(T_EMB))
    model_grouped[HEAD_TAG].get(P_HEAD).data.copy_(projected_weights.get(T_HEAD))
    for block_id, block_params in model_grouped[ENCODE_BLOCK_TAG].items():
        block_params.get(P_ATT_Q).data.copy_(projected_weights[ENCODE_BLOCK_TAG+block_id+"."+T_ATT_Q])
        block_params.get(P_ATT_K).data.copy_(projected_weights[ENCODE_BLOCK_TAG+block_id+"."+T_ATT_K])
        block_params.get(P_ATT_V).data.copy_(projected_weights[ENCODE_BLOCK_TAG+block_id+"."+T_ATT_V])

        block_params.get(P_ATT_OUT).data.copy_(projected_weights[ENCODE_BLOCK_TAG+block_id+"."+T_ATT_OUT])
        
        block_params.get(P_FF_IN).data.copy_(projected_weights[ENCODE_BLOCK_TAG+block_id+"."+T_FF_IN])
        block_params.get(P_FF_OUT).data.copy_(projected_weights[ENCODE_BLOCK_TAG+block_id+"."+T_FF_OUT])
    print("------------------------------copy weights end------------------------") #dev

    print("------------------------------init projections------------------------") #dev
    EMBEDDING_P = []
    EMBEDDING_P_T = [P_EMB_2, ]# P_EMB_PE #dev PE 
    DEEMBEDDING_P = [P_HEAD_1,]
    DEEMBEDDING_P_T = []
    BLOCK_P = [P_ATT_Q_1, P_ATT_K_1, P_ATT_V_1, P_ATT_OUT_1, P_FF_IN_1, P_FF_OUT_1,]
    BLOCK_P_T = [P_ATT_Q_2, P_ATT_K_2, P_ATT_V_2, P_ATT_OUT_2, P_FF_IN_2, P_FF_OUT_2,]

    if not weight_dependent_projections:
        add_projections(model_grouped[EMBEDDING_LAYER_TAG], projection,  projection.T, EMBEDDING_P, EMBEDDING_P_T)
        add_projections(model_grouped[HEAD_TAG], projection,  projection.T, DEEMBEDDING_P, DEEMBEDDING_P_T)
        for block_id, block_params in model_grouped[ENCODE_BLOCK_TAG].items():
            print(f"Block: {block_id}")
            add_projections(block_params, projection,  projection.T, BLOCK_P, BLOCK_P_T)
    elif weight_dependent_projections == "shared_block":
        projection, mask_1d = get_var_head_projection(dmodel, projected_dmodel, n_att_heads)
        add_projections(model_grouped[EMBEDDING_LAYER_TAG], projection,  projection.T, EMBEDDING_P, EMBEDDING_P_T)
        add_projections(model_grouped[HEAD_TAG], projection, projection.T, DEEMBEDDING_P, DEEMBEDDING_P_T)
        for block_id, block_params in model_grouped[ENCODE_BLOCK_TAG].items():
            projection, mask_1d = get_var_head_projection(dmodel, projected_dmodel, n_att_heads)
            print(f"Block: {block_id}")
            add_projections(block_params, projection,  projection.T, BLOCK_P, BLOCK_P_T)
    elif weight_dependent_projections == "svd":

        projection = torch.zeros(projected_dmodel, projected_dmodel)
        mask = torch.eye(projected_dmodel).bool()
        projection = projection.masked_fill(mask, 1)
        projection = projection[:, projection_mask]

        add_projections(model_grouped[EMBEDDING_LAYER_TAG], projection,  projection.T, EMBEDDING_P, EMBEDDING_P_T)
        add_projections(model_grouped[HEAD_TAG], projection,  projection.T, DEEMBEDDING_P, DEEMBEDDING_P_T)
        for block_id, block_params in model_grouped[ENCODE_BLOCK_TAG].items():
            print(f"Preinit block: {block_id}")
            add_projections(block_params, projection,  projection.T, BLOCK_P, BLOCK_P_T)
    
        for block_id, block_params in model_grouped[ENCODE_BLOCK_TAG].items():
            print(f"Block: {block_id}")

            att_output_proj_w = block_params[P_ATT_OUT]
            u, s, v = svd_init_truncated_sv(att_output_proj_w, dmodel, projected_dmodel)
            block_params[P_ATT_OUT_1].data.copy_(u.T)
            block_params[P_ATT_OUT].data.copy_(s)
            block_params[P_ATT_OUT_2].data.copy_(v)

            ff_in = block_params[P_FF_IN]
            u, s, v = svd_init_truncated_sv(ff_in, dmodel, projected_dmodel)
            block_params[P_FF_IN_1].data.copy_(u.T)
            block_params[P_FF_IN].data.copy_(s)
            block_params[P_FF_IN_2].data.copy_(v)

            ff_out = block_params[P_FF_OUT]
            u, s, v = svd_init_truncated_sv(ff_out, dmodel, projected_dmodel)
            block_params[P_FF_OUT_1].data.copy_(u.T)
            block_params[P_FF_OUT].data.copy_(s)
            block_params[P_FF_OUT_2].data.copy_(v)
    elif weight_dependent_projections == "magnitude_global":

        def topk_from_equal_subsets(tensor, k_per_chunk, num_chunks):
            LARGEST_SCORES = True
            # LARGEST_SCORES = False #dev SWITCH

            assert tensor.numel() % num_chunks == 0, "Tensor size must be divisible by num_chunks"
            chunk_size = tensor.numel() // num_chunks
            topk_indices = []

            for i in range(num_chunks):
                start = i * chunk_size
                end = (i + 1) * chunk_size
                chunk = tensor[start:end]

                _, local_topk = torch.topk(chunk, k=k_per_chunk, largest=LARGEST_SCORES)
                local_topk = local_topk.sort()[0]
                global_topk = local_topk + start  # shift to global index
                topk_indices.append(global_topk)
            return torch.cat(topk_indices)
        
        def calculate_scores(w, input:bool, marked_size=projected_dmodel):
            LNP = 2 #dev SWITCH
            # LNP = 8
            if input:
                assert w.shape[1] == marked_size
                scores = torch.norm(w, p=LNP, dim=0)
            else:
                assert w.shape[0] == marked_size
                scores = torch.norm(w, p=LNP, dim=1)
            assert scores.shape[0] == marked_size
            return scores
        dm_scores = []
        dm_scores.append(calculate_scores(model_grouped[EMBEDDING_LAYER_TAG].get(P_EMB).T, False))
        dm_scores.append(calculate_scores(model_grouped[HEAD_TAG].get(P_HEAD), True))
        ff_scores = []
        for block_id, block_params in model_grouped[ENCODE_BLOCK_TAG].items():
            block_ff_scores = []
            dm_scores.append(calculate_scores(block_params.get(P_ATT_Q), True))
            dm_scores.append(calculate_scores(block_params.get(P_ATT_K), True))
            dm_scores.append(calculate_scores(block_params.get(P_ATT_V), True))

            dm_scores.append(calculate_scores(block_params.get(P_ATT_OUT), False))

            dm_scores.append(calculate_scores(block_params.get(P_FF_IN), True))
            dm_scores.append(calculate_scores(block_params.get(P_FF_OUT), False))

            # dm_scores.append(calculate_scores(block_params.get(P_FF_IN), False)) #dev SWITCH
            # dm_scores.append(calculate_scores(block_params.get(P_FF_OUT), True)) #dev SWITCH

            block_ff_scores.append(calculate_scores(block_params.get(P_FF_IN), False, projected_dmodel*4))
            block_ff_scores.append(calculate_scores(block_params.get(P_FF_OUT), True, projected_dmodel*4))
            ff_scores.append(block_ff_scores)


        projection = torch.eye(projected_dmodel)
        global_importance = torch.stack(dm_scores, dim=0).mean(dim=0)
        
        indices = topk_from_equal_subsets(global_importance, int(dmodel/n_att_heads), n_att_heads)
        # n_chunks_select=int(projected_dmodel/4) #dev SWITCH
        # n_chunks_select=1 #dev SWITCH
        # indices = topk_from_equal_subsets(global_importance, int(dmodel/n_chunks_select), n_chunks_select) #dev SWITCH
        # indices = torch.topk(global_importance, k=dmodel, largest=True)[1].sort()[0] #dev SWITCH
        projection = projection[:, indices]

        # projection, mask_1d = get_var_head_projection(dmodel, projected_dmodel, n_att_heads)#dev switch
        # mask = torch.eye(projected_dmodel).bool()
        # projection = projection.masked_fill(mask, 1)
        # mask_1d = torch.ones(int(projected_dmodel/n_att_heads), dtype=torch.bool)
        # mask_1d[int(dmodel/n_att_heads):] = False #dev
        # projection = projection[:, torch.concat([mask_1d]*n_att_heads)]#dev switch

        print("global magnitude initialized projection -------------------") #dev
        print(projection) #dev
        add_projections(model_grouped[EMBEDDING_LAYER_TAG], projection,  projection.T, EMBEDDING_P, EMBEDDING_P_T)
        add_projections(model_grouped[HEAD_TAG], projection,  projection.T, DEEMBEDDING_P, DEEMBEDDING_P_T)

        # for (block_id, block_params), block_ff_scores in zip(model_grouped[ENCODE_BLOCK_TAG].items(), ff_scores):
        #     print(f"Block: {block_id}")
        #     add_projections(block_params, projection,  projection.T, BLOCK_P, BLOCK_P_T)

        for (block_id, block_params), block_ff_scores in zip(model_grouped[ENCODE_BLOCK_TAG].items(), ff_scores):
            block_params[P_ATT_Q_1].data.copy_(projection)
            block_params[P_ATT_K_1].data.copy_(projection)
            block_params[P_ATT_V_1].data.copy_(projection)
            block_params[P_ATT_OUT_2].data.copy_(projection.T)

            block_ff_projection = torch.eye(projected_dmodel*4)
            block_ff_importance = torch.stack(block_ff_scores, dim=0).mean(dim=0)
            # block_ff_indices = topk_from_equal_subsets(block_ff_importance, int(dmodel*4/n_att_heads), n_att_heads) #dev ff*4
            block_ff_indices = topk_from_equal_subsets(block_ff_importance, dmodel*4, 1) #dev ff*4
            # block_ff_indices = topk_from_equal_subsets(block_ff_importance, int(dmodel/n_chunks_select), n_chunks_select) #dev SWITCH
            # block_ff_indices = torch.topk(block_ff_importance, k=dmodel, largest=True)[1].sort()[0] #dev SWITCH
            block_ff_projection = block_ff_projection[:, block_ff_indices]

            print(f"projection.shape {projection.shape}") #dev
            print(f"block_ff_projection.shape {block_ff_projection.shape}") #dev

            print(f"block_params[P_FF_IN_1].data.shape {block_params[P_FF_IN_1].data.shape}") #dev
            print(f"block_params[P_FF_IN_2].data.shape {block_params[P_FF_IN_2].data.shape}") #dev

            print(f"block_params[P_FF_OUT_1].data.shape {block_params[P_FF_OUT_1].data.shape}") #dev
            print(f"block_params[P_FF_OUT_2].data.shape {block_params[P_FF_OUT_2].data.shape}") #dev

            block_params[P_FF_IN_1].data.copy_(projection)
            block_params[P_FF_IN_2].data.copy_(block_ff_projection.T) #dev error

            block_params[P_FF_OUT_1].data.copy_(block_ff_projection)
            block_params[P_FF_OUT_2].data.copy_(projection.T)



    else:
        raise ValueError(f"Invalid `weight_dependent_projections` = {weight_dependent_projections}")
    print("------------------------------init projections end------------------------") #dev

