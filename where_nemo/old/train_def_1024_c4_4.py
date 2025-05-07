
import os
import torch
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
from nemo.core.optim.lr_scheduler import CosineAnnealing

import glob

def list_all_files_recursive(directory):
    return glob.glob(f"{directory}/**/*", recursive=True)


if __name__ == "__main__":
    # seq_length = 256
    # global_batch_size = 64

    seq_length = 512
    global_batch_size = 512

    # c4_dataset_train = load_from_disk("/nemo_run/datasets/c4/train")
    
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # tokenizer.pad_token = tokenizer.eos_token  # Necessary

    # collate_fn = build_collate_fn_with_tokenizer(tokenizer, max_length=seq_length)

    # data = HFDatasetDataModule(
    #     path_or_dataset=c4_dataset_train,
    #     split=None,
    #     micro_batch_size=4,
    #     pad_token_id=tokenizer.pad_token_id,
    # )

    # run.Config(
    #     llm.PreTrainingDataModule,
    #     paths=["/data/slimpajama_megatron/concatenated_chunk1.jsonl_text_document"],
    #     seq_length=seq_length,
    #     global_batch_size=gbs,
    #     micro_batch_size=mbs,
    #     tokenizer=run.Config(SentencePieceTokenizer, model_path="/data/tokenizer/tokenizer.model"),
    #     split="99,8,2",
    #     num_workers=2,
    #     index_mapping_dir="/data/index_mapping",
    # )

    # # Define the tokenizer configuration
    # tokenizer_cfg = run.Config(SentencePieceTokenizer, model_path="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/tokenizer.model")

    # # Build the tokenizer instance
    # tokenizer = run.build(tokenizer_cfg)
    # print("Tokenizer exists:", os.path.isfile("/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/tokenizer.model"))

    tokenizer_cfg = run.Config(SentencePieceTokenizer, model_path="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/tokenizer.model")
    tokenizer = fdl.build(tokenizer_cfg)  # ✅ Build it here

    print("Dir")
    print(list_all_files_recursive("/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744659529/training/code/preprocessing_results/"))

    data = llm.PreTrainingDataModule(
        paths=["/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744659529/training/code/preprocessing_results/c4_en_train.jsonl_text_document"],
        # paths=["/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744659529/training/code/preprocessing_results"],
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=64,
        tokenizer=tokenizer,
        split="99,8,2",
        num_workers=32,
        index_mapping_dir="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744659529/training/code/preprocessing_results/c4_en_train.jsonl_text_document"
    )


    # Define the GPT model configuration
    gpt_config = llm.GPTConfig(
        num_layers=16,
        hidden_size=1024,
        ffn_hidden_size=1024,
        num_attention_heads=16,
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

    # Setup the optimizer
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=6e-4,
        weight_decay=0.1,
        bf16=True,
    )


    opt = nl.MegatronOptimizerModule(config=opt_config)

    # Configure the trainer
    trainer = nl.Trainer(
        devices=4,  # Adjust based on your hardware setup
        max_steps=15_533,  # Total training steps
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
        tags=["GPT", "def_1024", "nemo2.0"],  # Optional
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
