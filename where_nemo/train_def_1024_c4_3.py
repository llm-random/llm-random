
import torch
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import NeptuneLogger
from datasets import load_from_disk
from nemo.collections.llm.gpt.data import HFDatasetDataModule  # assuming it's defined or imported already
from transformers import AutoTokenizer
# from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
# from nemo.collections.nlp.parts.preprocessing import TokenizerSpec


# def build_collate_fn_with_tokenizer(tokenizer, max_length=1024, pad=True, pad_to_multiple_of=None):
#     def collate_fn(batch):
#         texts = [item["text"] for item in batch]
#         encoded = tokenizer(
#             texts,
#             padding="longest" if pad else False,
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt",
#             pad_to_multiple_of=pad_to_multiple_of,
#         )
#         encoded["labels"] = encoded["input_ids"].clone()
#         return encoded
#     return collate_fn

if __name__ == "__main__":
    seq_length = 256
    global_batch_size = 64

    c4_dataset_train = load_from_disk("/nemo_run/datasets/c4/train")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Necessary

    dataset = load_from_disk("/nemo_run/datasets/c4/train")

    # collate_fn = build_collate_fn_with_tokenizer(tokenizer, max_length=seq_length)


    data = HFDatasetDataModule(
        path_or_dataset=c4_dataset_train,
        split="train",
        micro_batch_size=4,
        pad_token_id=tokenizer.pad_token_id,
    )

    def preprocess(example):
        example['input_ids'] = tokenizer.encode(example['text'])
        return example

    data.map(preprocess, split_names=["train"])

    # Define the GPT model configuration
    gpt_config = llm.GPTConfig(
        num_layers=4,
        hidden_size=256,
        ffn_hidden_size=1024,
        num_attention_heads=4,
        seq_length=seq_length,
        init_method_std=0.02,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
    )
    model = llm.GPTModel(gpt_config, tokenizer=tokenizer)

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
        max_steps=500,  # Total training steps
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        log_every_n_steps=5,
        val_check_interval=500,
        limit_val_batches=50,
        limit_test_batches=50,
        accumulate_grad_batches=1,
    )

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZDg5ZTI2ZS00MWUyLTRkMTUtYTEzMC01OTVhYzE1ZWVmYzIifQ==",  # 🔐 Replace with your actual token or use env var
        project="pmtest/llm-random",  # 🔧 Replace with your project path
        tags=["GPT", "test", "nemo2.0"],  # Optional
        log_model_checkpoints=False,
        name="gpt-100m-run"
    )

    # Setup the logger
    nemo_logger = nl.NeMoLogger(
        log_dir="checkpoints/nemotron",  # Directory for logs and checkpoints
        extra_loggers=[neptune_logger]
    )

    # tokenizer_spec = TokenizerSpec(
    #     tokenizer=tokenizer,
    #     pad_id=tokenizer.pad_token_id,
    #     eos_id=tokenizer.eos_token_id
    # )

    # Start the training process
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        # tokenizer="model",
        optim=opt,
    )
