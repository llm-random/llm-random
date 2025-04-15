
import os
import torch
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import NeptuneLogger
# from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
# from nemo.collections.nlp.parts.preprocessing import TokenizerSpec
import nemo_run as run
from nemo.collections.common.tokenizers import SentencePieceTokenizer
import fiddle as fdl  # This is needed to build configs
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
# from nemo.core.optim.lr_scheduler import CosineAnnealing

import glob

def list_all_files_recursive(directory):
    return glob.glob(f"{directory}/**/*", recursive=True)


if __name__ == "__main__":
    seq_length = 256
    global_batch_size = 512
    max_steps = int(15_533*6)

    tokenizer_cfg = run.Config(SentencePieceTokenizer, model_path="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/tokenizer.model")
    tokenizer = fdl.build(tokenizer_cfg)  # ✅ Build it here

    print("Dir")
    print(list_all_files_recursive("/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744659529/training/code/preprocessing_results/"))

    data = llm.PreTrainingDataModule(
        paths=["/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744659529/training/code/preprocessing_results/c4_en_train.jsonl_text_document"],
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=128,
        tokenizer=tokenizer,
        split="99,8,2",
        num_workers=32,
        index_mapping_dir="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744659529/training/code/preprocessing_results/c4_en_train.jsonl_text_document"
    )


    # Define the GPT model configuration
    gpt_config = llm.GPTConfig(
        num_layers=int(16*1.5),
        hidden_size=int(1024*1.5),
        ffn_hidden_size=int(1024*1.5*4),
        num_attention_heads=int(16*1.5),
        seq_length=seq_length,
        init_method_std=0.02,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
    )
    model = llm.GPTModel(gpt_config, tokenizer=data.tokenizer)

    # Initialize the training strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
    )

    
    base_lr = 0.001
    final_lr_fraction = 0.03
    warmup_percent = 0.01

    # Setup the optimizer
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=base_lr,
        weight_decay=0.1,
        bf16=True,
    )

    # lr_scheduler = OptimizerParamScheduler(
    #     scheduler=CosineAnnealing(
    #         max_steps=max_steps,
    #         min_lr=base_lr * final_lr_fraction,
    #         warmup_ratio=warmup_percent,
    #         optimizer=opt
    #     )
    # )

    # def __init__(
    #     self,
    #     max_steps: int = 10,
    #     warmup_steps: int = 750,
    #     constant_steps: int = 80000,
    #     min_lr: float = 6e-5,
    #     interval: str = "step",
    #     frequency: int = 1,
    #     monitor: str = "val_loss",
    # ):

    lr_scheduler = CosineAnnealingScheduler(
        max_steps=max_steps,
        warmup_steps=int(max_steps*warmup_percent),
        constant_steps=0,
        min_lr=final_lr_fraction*base_lr
    )

    opt = nl.MegatronOptimizerModule(config=opt_config, lr_scheduler=lr_scheduler)

    # Configure the trainer
    trainer = nl.Trainer(
        devices=4,  # Adjust based on your hardware setup
        max_steps=max_steps,  # Total training steps
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        log_every_n_steps=100,
        val_check_interval=1000,
        limit_val_batches=50,
        limit_test_batches=50,
        accumulate_grad_batches=1,
    )

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZDg5ZTI2ZS00MWUyLTRkMTUtYTEzMC01OTVhYzE1ZWVmYzIifQ==",  # 🔐 Replace with your actual token or use env var
        project="pmtest/llm-random",  # 🔧 Replace with your project path
        # tags=["GPT", "test", "nemo2.0"],  # Optional
        tags=["GPT", "def_1024", "nemo2.0", "32w128b", "big_model"],  # Optional
        log_model_checkpoints=False,
        name="gpt-100m-run"
    )

    # Setup the logger
    nemo_logger = nl.NeMoLogger(
        log_dir="checkpoints/nemotron",  # Directory for logs and checkpoints
        extra_loggers=[neptune_logger]
    )

    # Start the training process
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer="data",
        optim=opt,
    )
