from collections import OrderedDict
from typing import Callable, Optional, Union, Type

from lizrd.core.initialization import get_init_weight
from lizrd.core.misc import Linear
from research.projected_distillation.llm import PredictionHeadRes, ProjectedPositionalEmbedding, ProjectedPositionalEmbeddingRes, ProjectedTokenEmbedding, ProjectedTokenEmbeddingRes
from research.projected_distillation.utils import freeze_ln_params, freeze_projected_params, get_var_head_projection, initialize_compressor
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from lizrd.core import llm
from lizrd.core.distributed import wrap_in_fsdp, wrap_in_ddp
from lizrd.train.checkpointing import make_checkpoint_wrapper_function
from lizrd.train.load_and_save_model import load_model_weights
from torch.distributed import (
    broadcast_object_list,
    barrier,
)
def prune_every_second_block(encoder_tower: torch.nn.Module):
    assert isinstance(encoder_tower.blocks, torch.nn.Sequential), "Expected nn.Sequential block container"
    pruned_blocks = OrderedDict(
        (name, block) for i, (name, block) in enumerate(encoder_tower.blocks._modules.items()) if i % 2 == 0
    )
    encoder_tower.blocks = torch.nn.Sequential(pruned_blocks)

def get_model(
    max_length: int,
    vocab_size: int,
    block_modules: dict[str, Callable[[], torch.nn.Module]],
    dm: int,
    n_blocks: int,
    device: torch.device,
    init_type,
    init_scale,
    ddp_enabled: bool, 
    fsdp_enabled: bool,
    fsdp_param_precision: torch.dtype,
    fsdp_mixed_precision_ignore_classes: list[Type[torch.nn.Module]],
    fsdp_offload_params: bool,
    fsdp_min_num_params: int,
    fsdp_modules_to_wrap: Union[tuple[Type[torch.nn.Module]], None],
    activation_checkpointing_modules: Union[tuple[Type[torch.nn.Module]], None],
    is_logging_process: bool,
    local_rank=None,
    model_fragmentation: Optional[list[int]] = None,
    residual_fn: Callable[[], torch.nn.Module] = None,
    include_positional_embedding: bool = True,
    checkpoint: dict[str, torch.Tensor] = None,
    projected_checkpoint: dict[str, torch.Tensor] = None,
    projected_dmodel:int = None,
    projection_init_type:str = None,
    no_projected_head:bool = False,
    no_layer_norm:bool = False,
    fsdp_use_orig_params:bool = False,
    unprojected_embeddings:bool = False,
    unprojected_attention:bool = False,
    unprojected_ff:bool = False,
    n_att_heads:int=None,
    distillation_type:Optional[str]=None,
    head_layer_norm:Optional[bool]=False,
    head_ln=True #dev - add param to config connection
):
    if model_fragmentation is None or device == torch.device("cpu"):
        first_gpu = device
        last_gpu = device
    else:
        first_gpu = torch.device("cuda:0")
        last_gpu = torch.device(f"cuda:{len(model_fragmentation)}")

    if projected_checkpoint and not unprojected_embeddings:
        # embedding_components = [
        #     ProjectedTokenEmbedding(vocab_size, dm, projected_dmodel, init_type=init_type, init_scale=init_scale)
        # ] #dev switch weights residuals
        embedding_components = [
            ProjectedTokenEmbeddingRes(vocab_size, dm, projected_dmodel, init_type=init_type, init_scale=init_scale)
        ]
    else:
        embedding_components = [
            llm.TokenEmbedding(vocab_size, dm, init_type=init_type, init_scale=init_scale)
        ]

    if include_positional_embedding:
        if projected_checkpoint and not unprojected_embeddings:
            embedding_components.append(
                # ProjectedPositionalEmbedding(
                #     max_length, dm, projected_dmodel, init_type=init_type, init_scale=init_scale
                # ) #dev switch weights residuals
                ProjectedPositionalEmbeddingRes(
                    max_length, dm, projected_dmodel, init_type=init_type, init_scale=init_scale
                )
            )
        else:
            embedding_components.append(
                llm.PositionalEmbedding(
                    max_length, dm, init_type=init_type, init_scale=init_scale
                )
            )

    embedding_layer = llm.EmbeddingLayer(*embedding_components).to(first_gpu)

    # Python officially preserves dict order since 3.7, so we pass the layer dict
    encoder_tower = llm.TransformerTower(
        n_blocks,
        dm,
        block_modules,
        device,
        model_fragmentation=model_fragmentation,
        residual_fn=residual_fn,
    )

    if projected_checkpoint and not no_projected_head and not unprojected_embeddings:
        # head = llm.PredictionHead(
        #     projected_dmodel, vocab_size, init_type=init_type, init_scale=init_scale
        # ).to(last_gpu)
        # head = torch.nn.Sequential(
        #     OrderedDict([
        #         (
        #             "head_p",
        #             Linear(
        #                 dm, #xs
        #                 projected_dmodel, #xb
        #                 bias=False,
        #                 init_type=init_type,
        #                 init_scale=init_scale,
        #             ).to(last_gpu),
        #         ),
        #         (
        #             "head",
        #             head,
        #         )
        #     ])
        # ) #dev switch weights residuals

        head = PredictionHeadRes( #dev
            projected_dmodel, vocab_size, dm, init_type=init_type, init_scale=init_scale, ln=head_ln
        ).to(last_gpu)


    else:
        head = llm.PredictionHead(
            dm, vocab_size, init_type=init_type, init_scale=init_scale, ln=head_layer_norm
        ).to(last_gpu)


    model = llm.LLM(embedding_layer, encoder_tower, head)

    if checkpoint is not None:
        load_model_weights(model, checkpoint)
    
    if distillation_type =="distilgpt":
        print("Prunning every second block! #dev") #dev
        print(f"loaded: {checkpoint}") #dev
        prune_every_second_block(model.encoder)


    frozen_modules = []
    mask_1d = None
    if projected_checkpoint is not None:
        if local_rank == 0 or local_rank is None:
            if not projection_init_type:
                projection = None
                print("No projection initialization")
            elif projection_init_type == "half":
                print("Projection initialization: half")
                projection = torch.zeros(projected_dmodel, projected_dmodel)
                mask = torch.eye(projected_dmodel).bool()
                projection = projection.masked_fill(mask, 1)
                projection = projection[:, :int(dm)]
            elif projection_init_type == "orthogonal":
                print("Projection initialization: orthogonal")
                projection = torch.empty(projected_dmodel, dm)
                projection = torch.nn.init.orthogonal_(projection)
            elif projection_init_type == "head_half_var":
                print("Projection initialization: head_half_var")
                assert (projected_dmodel/n_att_heads)%2 == 0
                assert (dm/n_att_heads)%2 == 0
                
                projection, mask_1d = get_var_head_projection(dm, projected_dmodel, n_att_heads)
                print(mask_1d)
            elif projection_init_type == "head_half":
                print("Projection initialization: head_half")
                assert (projected_dmodel/n_att_heads)%2 == 0
                
                projection = torch.zeros(projected_dmodel, projected_dmodel)
                mask = torch.eye(projected_dmodel).bool()
                projection = projection.masked_fill(mask, 1)

                mask_1d = torch.ones(int(projected_dmodel/n_att_heads), dtype=torch.bool)
                mask_1d[int(dm/n_att_heads):] = False #dev
                print(mask_1d) #dev
                projection = projection[:, torch.concat([mask_1d]*n_att_heads)]
            elif projection_init_type == "svd_half":
                print("Projection initialization: svd_half")
                assert (projected_dmodel/n_att_heads)%2 == 0
                
                mask_1d = torch.ones(int(projected_dmodel/n_att_heads), dtype=torch.bool)
                mask_1d[int(len(mask_1d)/2):] = False #dev
                mask_1d = torch.concat([mask_1d]*n_att_heads)
                print(mask_1d) #dev
                projection = "svd"
            elif projection_init_type == "shared_block_half_var":
                print("Projection initialization: shared_block_half_var")
                assert (projected_dmodel/n_att_heads)%2 == 0
                assert (dm/n_att_heads)%2 == 0

                projection, mask_1d = get_var_head_projection(dm, projected_dmodel, n_att_heads)
                projection = "shared_block"
            elif projection_init_type == "shared_att_in_half":
                print("Projection initialization: shared_att_in_half")
                assert (projected_dmodel/n_att_heads)%2 == 0
                
                mask_1d = torch.ones(int(projected_dmodel/n_att_heads), dtype=torch.bool)
                mask_1d[int(len(mask_1d)/2):] = False
                mask_1d = torch.concat([mask_1d]*n_att_heads)
                print(mask_1d) #dev
                projection = "shared_att_in"
            elif projection_init_type == "shared_att_out_half":
                print("Projection initialization: shared_att_out_half")
                assert (projected_dmodel/n_att_heads)%2 == 0
                
                mask_1d = torch.ones(int(projected_dmodel/n_att_heads), dtype=torch.bool)
                mask_1d[int(len(mask_1d)/2):] = False
                mask_1d = torch.concat([mask_1d]*n_att_heads)
                print(mask_1d) #dev
                projection = "shared_att_out"
            elif projection_init_type == "shared_att_in_out_half":
                print("Projection initialization: shared_att_in_out_half")
                assert (projected_dmodel/n_att_heads)%2 == 0
                
                mask_1d = torch.ones(int(projected_dmodel/n_att_heads), dtype=torch.bool)
                mask_1d[int(len(mask_1d)/2):] = False
                mask_1d = torch.concat([mask_1d]*n_att_heads)
                print(mask_1d) #dev
                projection = "shared_att_in_out"
            else:
                raise Exception("Wrong projection init type")
            
        if local_rank is not None:
            if local_rank == 0:
                projection = [projection]
            else:
                projection = [None]
            barrier()
            broadcast_object_list(projection, src=0)
            projection = projection[0]
            print(f"rank: {local_rank} - {projection}") #dev

        if isinstance(projection, torch.Tensor):
            projection = projection.to(device) #dev to device projection reference 
        # load_projected_weights(model, projected_checkpoint["model"], projection, dm, projected_dmodel, init_scale, unprojected_embeddings, unprojected_attention, unprojected_ff)
        initialize_compressor(model, projected_checkpoint["model"], dm, projected_dmodel, n_att_heads, projection, mask_1d) #dev
        frozen_modules = freeze_projected_params(model, unprojected_ff)

    

    if no_layer_norm:
        ln_frozen_modules = freeze_ln_params(model)
        frozen_modules = frozen_modules+ln_frozen_modules
        
    for name, param in model.named_parameters(): #dev
        print(f"{name}, shape: {param.shape} requires_grad: {param.requires_grad}, {param.device}")
        
    if ddp_enabled:
        model = wrap_in_ddp(module=model, local_rank=local_rank)
    elif fsdp_enabled:
        model = wrap_in_fsdp(
            module=model,
            local_rank=local_rank,
            param_precision=fsdp_param_precision,
            cast_inputs=True,
            mixed_precision_ignored_classes=fsdp_mixed_precision_ignore_classes,
            offload_params=fsdp_offload_params,
            print_model=True,
            min_num_params=fsdp_min_num_params,
            modules_to_wrap=fsdp_modules_to_wrap,
            is_logging_process=is_logging_process,
            # frozen_modules=frozen_modules
            fsdp_use_orig_params=fsdp_use_orig_params,
        )

    if activation_checkpointing_modules is not None:
        check_fn = lambda x: isinstance(x, activation_checkpointing_modules)
        apply_activation_checkpointing(
            model,
            check_fn=check_fn,
            checkpoint_wrapper_fn=make_checkpoint_wrapper_function(),
        )

    return model
