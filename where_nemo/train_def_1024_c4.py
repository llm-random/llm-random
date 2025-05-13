
import torch
import torch.nn.functional as F
import nemo_run as run
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
from nemo.collections.common.tokenizers import SentencePieceTokenizer, AutoTokenizer
from pytorch_lightning.loggers import NeptuneLogger
import fiddle as fdl  # This is needed to build configs
import glob
import numpy as np
import random
import pytorch_lightning as pl
from omegaconf import OmegaConf
from nemo.lightning.io.pl import TrainerContext, ckpt_to_weights_subdir
from nemo.collections.llm.modelopt.prune import prune_gpt_model, save_pruned_model
from nemo.collections.nlp.models.language_modeling.megatron  import GPTModel
# from nemo.collections import llm.GPTModel GPTModelConfig

import os
import sys
import platform
import torch
import json
import subprocess

from megatron.core import dist_checkpointing

accumulate_grad_batches=1
seq_length = 512
global_batch_size = int(512/accumulate_grad_batches)
micro_batch_size = int(128/accumulate_grad_batches)
# max_steps = 15_533*4*accumulate_grad_batches #SWITCH
max_steps = 15_533*accumulate_grad_batches #SWITCH
# max_steps = 10_179 #SWITCH
# max_steps = 100

# dataset_path = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744842516/training/code/results_preprocessing/c4_en_train_part_00.jsonl_text_document" # megatron hf files 
# dataset_path = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744843405/training/code/results_preprocessing/c4_en_train_part_00.jsonl_text_document" # megatron auto
# dataset_path = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744842905/training/code/results_preprocessing/c4_en_train_part_00.jsonl_text_document" # hf auto 

# dsp_preambule = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744842905/training/code/results_preprocessing/"
# dataset_path = [
#     f"{dsp_preambule}c4_en_train_part_0{i}.jsonl_text_document" for i in range(0, 10)
# ] 

# DEV datasets:
# --append-eod: #dev yes
# dsp_preambule = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1746191063/training/code/results_preprocessing/"
# dataset_path = [
#     f"{dsp_preambule}c4_en_train_part_0{i}.jsonl_text_document" for i in range(0, 1)
# ] 

# "--append-eod", #dev yes
# # "--apply-ftfy", #dev no
# # '--need-pad-id', #dev no
# dsp_preambule = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1746191424/training/code/results_preprocessing/"
# dataset_path = [
#     f"{dsp_preambule}c4_en_train_part_0{i}.jsonl_text_document" for i in range(0, 1)
# ] 

# "--append-eod", #dev yes
# "--apply-ftfy", #dev yes
# '--need-pad-id', #dev yes
# dsp_preambule = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1746191643/training/code/results_preprocessing/"
# dataset_path = [
#     f"{dsp_preambule}c4_en_train_part_0{i}.jsonl_text_document" for i in range(0, 1)
# ] 

# "--append-eod", #dev yes
# # "--apply-ftfy", #dev no
# '--need-pad-id', #dev yes
dsp_preambule = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1746522127/training/code/results_preprocessing/" #dev test
dataset_path = [
    f"{dsp_preambule}c4_en_train_part_0{i}.jsonl_text_document" for i in range(0, 1)
] 

# ENTROPY
# "--append-eod", #dev yes
# # "--apply-ftfy", #dev no
# dsp_preambule = "/home/mstefaniak/mas/nemo/mstest/"
# dataset_path = [
#     f"{dsp_preambule}c4_en_train_part_0{i}.jsonl_text_document" for i in range(0, 1)
# ] 

# FINAL DATASETS

# HELIOS
# "--append-eod", #dev yes
# # "--apply-ftfy", #dev no
# '--need-pad-id', #dev yes
# dsp_preambule = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1746711699/training/code/results_preprocessing/" #dev test
# dataset_path = [
#     f"{dsp_preambule}c4_en_train_part_0{i}.jsonl_text_document" for i in range(0, 10)
# ] 


num_layers=16
num_attention_heads=16
hidden_size=1024 #SWITCH
ffn_hidden_size=1024 #SWITCH
# hidden_size=768 #SWITCH
# ffn_hidden_size=768 #SWITCH
init_method_std=0.02
hidden_dropout=0.0
attention_dropout=0.0
layernorm_epsilon=1e-5
make_vocab_size_divisible_by=128
tags = ["projected_dis", "1ff", "nemo", "dm1024"] + ["test_dataset_prep", "append_eod", "need_pad_id", "relu", "no_head_ln"] # + ["long"] #SWITCH need_pad_id apply_ftfy no_head_ln
# tags = ["projected_dis", "1ff", "nemo", "dm768"] #SWITCH
seed = 27 
base_lr = 0.001 #SWITCH
# base_lr = 0.0005 #SWITCH
final_lr_fraction = 0.03
warmup_percent = 0.01
activation_func = F.relu # F.silu   
weight_decay = 0.1
clip_grad = 0.5
position_embedding_type = "rope" # rope ; learned_absolute
mm_precision="bf16-mixed" 
# mm_precision="32"
tensor_model_parallel_size = 1
# tensor_model_parallel_size = 2
pipeline_model_parallel_size = 1
share_embeddings_and_output_weights = False
# fp8_wgrad = False
# add_bias_linear = False
# fp32_residual_connection = False
# attention_softmax_in_fp32 = True

pl.seed_everything(seed, workers=True)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def before_exps_logs(neptune_logger):
    # Log general environment info
    neptune_logger.experiment["environment"] = {
        "working_directory": os.getenv("HOST_PATH", "UNKNOWN"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": torch.cuda.device_count(),
    }

    # Log Git commit if in repo
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        neptune_logger.experiment["git/commit"] = commit
    except Exception:
        neptune_logger.experiment["git/commit"] = "Not a git repo or error"

    # Log all hyperparameters
    hyperparams = {
        "seed": seed,
        "seq_length": seq_length,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "max_steps": max_steps,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "hidden_size": hidden_size,
        "ffn_hidden_size": ffn_hidden_size,
        "init_method_std": init_method_std,
        "hidden_dropout": hidden_dropout,
        "attention_dropout": attention_dropout,
        "layernorm_epsilon": layernorm_epsilon,
        "make_vocab_size_divisible_by": make_vocab_size_divisible_by,
        "base_lr": base_lr,
        "final_lr_fraction": final_lr_fraction,
        "warmup_percent": warmup_percent,
        "dataset_path": dataset_path,
        "tags": tags,
        "activation_func": str(activation_func),
        "weight_decay": weight_decay,
        "clip_grad": clip_grad,
        "position_embedding_type": position_embedding_type,
        "mm_precision":mm_precision,
        "accumulate_grad_batches":accumulate_grad_batches,
        "tensor_model_parallel_size":tensor_model_parallel_size,
        "pipeline_model_parallel_size":pipeline_model_parallel_size,
        "share_embeddings_and_output_weights":share_embeddings_and_output_weights,
        # "fp8_wgrad": fp8_wgrad,
        # "add_bias_linear": add_bias_linear,
        # "fp32_residual_connection": fp32_residual_connection,
        # "attention_softmax_in_fp32": attention_softmax_in_fp32,
    }
    neptune_logger.experiment["hyperparameters"] = hyperparams

def save_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix='')
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)

if __name__ == "__main__":
    
    print("Its TRAINING BS ---------------------------------------------------------------") #dev 
    print(dataset_path) #dev
    
    tokenizer_cfg = run.Config(AutoTokenizer, 
        pretrained_model_name="gpt2", #dev ? /net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/gpt2/ openai-community/gpt2 gpt2 GPT2Tokenizer  
    )
    tokenizer = fdl.build(tokenizer_cfg)  # ✅ Build it here

    data = llm.PreTrainingDataModule(
        # paths=[dataset_path],
        paths=dataset_path,
        # paths="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744842905/training/code/results_preprocessing",
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        tokenizer=tokenizer,
        split="98,1,1",
        num_workers=32,
        # index_mapping_dir=dataset_path,
        # index_mapping_dir="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744842905/training/code/results_preprocessing/",
    )


    # Define the GPT model configuration
    gpt_config = llm.GPTConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        seq_length=seq_length,
        init_method_std=init_method_std,
        hidden_dropout=hidden_dropout,
        attention_dropout=attention_dropout,
        layernorm_epsilon=layernorm_epsilon,
        make_vocab_size_divisible_by=make_vocab_size_divisible_by,
        position_embedding_type=position_embedding_type,
        activation_func = activation_func,
        share_embeddings_and_output_weights = share_embeddings_and_output_weights,
        # fp8_wgrad = fp8_wgrad,
        # add_bias_linear = add_bias_linear,
        # fp32_residual_connection = fp32_residual_connection,
        # attention_softmax_in_fp32 = attention_softmax_in_fp32,
    )
    model = llm.GPTModel(gpt_config, tokenizer=data.tokenizer)
    

    # Initialize the training strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=torch.bfloat16,
    )

    # Setup the optimizer
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=base_lr,
        weight_decay=weight_decay,
        clip_grad=clip_grad,
        bf16=True,
    )

    lr_scheduler = CosineAnnealingScheduler(
        max_steps=max_steps,
        warmup_steps=int(max_steps*warmup_percent),
        constant_steps=0,
        min_lr=final_lr_fraction*base_lr,
    )

    opt = nl.MegatronOptimizerModule(config=opt_config, lr_scheduler=lr_scheduler)

    trainer = nl.Trainer(
        devices=4,
        max_steps=max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision=mm_precision),
        log_every_n_steps=1,
        val_check_interval=None,
        limit_val_batches=50,
        limit_test_batches=50,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_progress_bar=False,
    )

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZDg5ZTI2ZS00MWUyLTRkMTUtYTEzMC01OTVhYzE1ZWVmYzIifQ==",
        project="pmtest/llm-random",
        tags=tags,
        log_model_checkpoints=False,
        name="PD_nemo",
    )

    before_exps_logs(neptune_logger)

    nemo_logger = nl.NeMoLogger(
        log_dir="checkpoints/nemotron",
        extra_loggers=[neptune_logger],
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer="data",
        optim=opt,
    )

    save_pruned_model(save_path="gimmi_checkpoint_pls", trainer=trainer)

