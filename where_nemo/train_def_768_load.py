
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
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.collections.llm.modelopt import setup_trainer_and_restore_model_with_modelopt_spec


accumulate_grad_batches = 1
seq_length = 512
global_batch_size = 512
micro_batch_size = 128
max_steps = 10_179 #SWITCH
max_steps = 20_358 #SWITCH
max_steps = 40_716 #SWITCH
max_steps = 81_432 #SWITCH
# max_steps = 1 

# OLD
# dsp_preambule = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744842905/training/code/results_preprocessing/"
# dataset_path = [
#     f"{dsp_preambule}c4_en_train_part_0{i}.jsonl_text_document" for i in range(0, 10)
# ] 

# "--append-eod", #dev yes
# "--apply-ftfy", #dev yes
# '--need-pad-id', #dev yes
dsp_preambule = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/llm_reproduction/" #dev test
dataset_path = [
    f"{dsp_preambule}c4_en_train_part_0{i}.jsonl_text_document" for i in range(0, 10)
] 



# 768_4PO:
load_checkpoint = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1747401850/training/code/prrruned_nyan"
# load_checkpoint = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1747405644/training/code/prrruned_nyan" #param match
hidden_size=768
ffn_hidden_size=768
# 832_4PO:
# load_checkpoint = "asd"
# hidden_size=832
# ffn_hidden_size=832
# 256_4PO
# load_checkpoint = "asd"
# hidden_size=256
# ffn_hidden_size=256
# Not tig. emb.:
# 768_4PO:
# load_checkpoint = "asd"
# hidden_size=768
# ffn_hidden_size=768
# 832_4PO:
# load_checkpoint = "asd"
# hidden_size=832
# ffn_hidden_size=832


num_layers=16
num_attention_heads=num_layers

init_method_std=0.02
hidden_dropout=0.0
attention_dropout=0.0
layernorm_epsilon=1e-5
make_vocab_size_divisible_by=64
tags = ["projected_dis", "nemo", "1ff", "dm768_1024", "pruned", "COMP"]
# tags = ["projected_dis", "nemo", "1ff", "dm832_1024", "pruned", "4xPO"]
# tags = ["projected_dis", "nemo", "1ff", "dm256_1024", "pruned", "4xPO"]
seed = 27
base_lr = 0.0005 #dev
final_lr_fraction = 0.03
warmup_percent = 0.01
activation_func = F.silu
weight_decay = 0.1
clip_grad = 0.5
position_embedding_type = "rope"
devices = 4
mm_precision="bf16-mixed" 
tensor_model_parallel_size = 1
pipeline_model_parallel_size = 1
share_embeddings_and_output_weights = False

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
    }
    neptune_logger.experiment["hyperparameters"] = hyperparams

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


    model, trainer = setup_trainer_and_restore_model_with_modelopt_spec(
        load_checkpoint,
        devices = devices,
        tensor_model_parallel_size = tensor_model_parallel_size,
        pipeline_model_parallel_size = pipeline_model_parallel_size,
        trainer_kwargs = {
            "max_steps":max_steps,
            "log_every_n_steps":1,
            "val_check_interval":None,
            "limit_val_batches":10,
            "limit_test_batches":10,
            "accumulate_grad_batches":accumulate_grad_batches,
            "enable_progress_bar":False,
        },
        inference_only = False,
    )

    print("model ----------------------------------------------------") #dev
    print(f"type(model) {type(model)} eop") #dev
    print(f"model(model) {model} eop") #def
    print("end model ----------------------------------------------------") #dev

    # Initialize the training strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
    )
    
    trainer = nl.Trainer(
        devices=devices,
        max_steps=max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision=mm_precision),
        log_every_n_steps=1,
        val_check_interval=None,
        limit_val_batches=10,
        limit_test_batches=10,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_progress_bar=False,
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

