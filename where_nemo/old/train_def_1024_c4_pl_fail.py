
import os
import torch
import nemo_run as run
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo import lightning as nl
from nemo.lightning.pytorch.strategies.fsdp_strategy import FSDPStrategy
from lightning.pytorch.plugins.precision.fsdp import FSDPPrecision
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
from nemo.collections.common.tokenizers import SentencePieceTokenizer, AutoTokenizer
from pytorch_lightning.loggers import NeptuneLogger
import fiddle as fdl  # This is needed to build configs
import glob
import numpy as np
import random
import pytorch_lightning as pl
from torch.optim import SGD, AdamW
from functools import partial
 
from nemo.collections.llm.gpt.model import llama_embedding
from nemo.collections.llm.gpt.model import base
from nemo.collections.llm.gpt.data import pre_training
from nemo.collections.llm.gpt.data import mock
base.HAVE_TE = False
pre_training.HAVE_TE = False
llama_embedding.HAVE_TE = False
mock.HAVE_TE = False

os.environ["NVTE_NO_CUBLASLT"] = "1"

seq_length = 256
global_batch_size = 512
max_steps = 15_533
# max_steps = 1000

# dataset_path = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744842516/training/code/results_preprocessing/c4_en_train_part_00.jsonl_text_document" # megatron hf files 
# dataset_path = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744843405/training/code/results_preprocessing/c4_en_train_part_00.jsonl_text_document" # megatron auto
dataset_path = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744842905/training/code/results_preprocessing/c4_en_train_part_00.jsonl_text_document" # hf auto 

num_layers=16
num_attention_heads=16
hidden_size=1024
ffn_hidden_size=1024
seq_length=seq_length
init_method_std=0.02
hidden_dropout=0.1
attention_dropout=0.1
layernorm_epsilon=1e-5
make_vocab_size_divisible_by=64
tags = ["projected_dis", "1ff", "dm1024", "doner", "nemo", "seeding", "dev"]
seed = 27
base_lr = 0.001
final_lr_fraction = 0.03
warmup_percent = 0.01
weight_decay=0.1


pl.seed_everything(seed, workers=True)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def list_all_files_recursive(directory):
    return glob.glob(f"{directory}/**/*", recursive=True)


if __name__ == "__main__":
    

    print("Its TRAINING BS ---------------------------------------------------------------") #dev 
    
    # tokenizer_cfg = run.Config(SentencePieceTokenizer, model_path="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/tokenizer.model")
    tokenizer_cfg = run.Config(AutoTokenizer, 
        pretrained_model_name="gpt2", #dev ? /net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/gpt2/ openai-community/gpt2 gpt2 GPT2Tokenizer  
    )
    tokenizer = fdl.build(tokenizer_cfg)  # ✅ Build it here

    print("Dir")
    print(list_all_files_recursive("/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744659529/training/code/preprocessing_results/"))

    data = llm.PreTrainingDataModule(
        paths=[dataset_path],
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=128,
        tokenizer=tokenizer,
        split="99,8,2",
        num_workers=32,
        index_mapping_dir=dataset_path,
        create_attention_mask=True
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
        use_transformer_engine_full_layer_spec=False
    )
    model = llm.GPTModel(gpt_config, tokenizer=data.tokenizer)
    
    # # Initialize the training strategy
    # strategy = nl.MegatronStrategy(
    #     tensor_model_parallel_size=1,
    #     pipeline_model_parallel_size=1,
    #     pipeline_dtype=torch.bfloat16,
    # )

    # Setup the optimizer
    # opt_config = OptimizerConfig(
    #     optimizer='adam',
    #     lr=base_lr,
    #     weight_decay=0.1,
    #     bf16=True,
    # )

    strategy = FSDPStrategy( # full
        # auto_wrap_policy={llm.GPTModel},
        cpu_offload=False,
        state_dict_type="full",
    )


    optimizer_fn = partial(
        AdamW,
        lr=base_lr,
        weight_decay=weight_decay,
    )
    

    lr_scheduler = CosineAnnealingScheduler(
        max_steps=max_steps,
        warmup_steps=int(max_steps*warmup_percent),
        constant_steps=0,
        min_lr=final_lr_fraction*base_lr
    )

    opt = nl.PytorchOptimizerModule(optimizer_fn=optimizer_fn, lr_scheduler=lr_scheduler)
    # opt = nl.MegatronOptimizerModule(config=opt_config, lr_scheduler=lr_scheduler)

    # optimizer_fn = run.Partial(
    #     SGD,
    #     lr=lr,
    #     weight_decay=wd,
    # )
    # lr_scheduler = MyLRSchedulerModule(...)
    # optimizer_module = PytorchOptimizerModule(optimizer_fn, lr_scheduler)

    trainer = nl.Trainer(
        devices=4,
        max_steps=max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=FSDPPrecision(precision="bf16-mixed"),
        log_every_n_steps=100,
        val_check_interval=1000,
        limit_val_batches=50,
        limit_test_batches=50,
        accumulate_grad_batches=1,
    )

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZDg5ZTI2ZS00MWUyLTRkMTUtYTEzMC01OTVhYzE1ZWVmYzIifQ==",
        project="pmtest/llm-random",
        tags=tags,
        log_model_checkpoints=False,
        name="PD_nemo"
    )
    nemo_logger = nl.NeMoLogger(
        log_dir="checkpoints/nemotron",
        extra_loggers=[neptune_logger]
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer="data",
        optim=opt,
    )
