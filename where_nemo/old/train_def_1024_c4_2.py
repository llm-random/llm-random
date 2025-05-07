import torch
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import NeptuneLogger
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
# from transformers import AutoTokenizer
# from nemo.collections.common.tokenizers.huggingface import HuggingFaceTokenizer
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer


if __name__ == "__main__":
    seq_length = 256
    global_batch_size = 64

    # Load C4 dataset (from local cache if available) 
    # raw_dataset = load_dataset("c4", "en", split="train[:1%]", trust_remote_code=True)
    # raw_dataset = load_dataset("c4", "en", split="train", trust_remote_code=True)
    raw_dataset = load_from_disk("/nemo_run/datasets/c4/train")
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # tokenizer = HuggingFaceTokenizer(pretrained_model_name="EleutherAI/gpt-neox-20b")
    tokenizer = get_nmt_tokenizer("megatron", "GPT2BPETokenizer")


    # Tokenize in memory
    # def preprocess(example):
    #     return tokenizer(example["text"], truncation=True, max_length=seq_length)

    # tokenized = raw_dataset.map(preprocess, batched=True, keep_in_memory=True)

    # Wrap tokenized dataset into PyTorch Dataset
    class HuggingFaceDatasetWrapper(Dataset):
        def __init__(self, hf_dataset, tokenizer, max_length=1024):
            self.dataset = hf_dataset
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            text = item["text"]  # Replace with your actual field
            tokenized = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            return {key: val.squeeze(0) for key, val in tokenized.items()}
        
    class NeMoCompatibleDataModule(LightningDataModule):
        def __init__(self, train_dataset, val_dataset, batch_size=8, num_workers=2):
            super().__init__()
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.batch_size = batch_size
            self.num_workers = num_workers

        def train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )

        def val_dataloader(self):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )

    torch_dataset = HFWrapperDataset(raw_dataset, tokenizer, seq_length)

    # Create a custom LightningDataModule
    class C4DataModule(LightningDataModule):
        def __init__(self, dataset, batch_size):
            super().__init__()
            self.dataset = dataset
            self.batch_size = batch_size

        def train_dataloader(self):
            return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    data = C4DataModule(torch_dataset, batch_size=global_batch_size)

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
        max_steps=500,
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
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZDg5ZTI2ZS00MWUyLTRkMTUtYTEzMC01OTVhYzE1ZWVmYzIifQ==",
        project="pmtest/llm-random",
        tags=["GPT", "test", "nemo2.0"],
        log_model_checkpoints=False,
        name="gpt-100m-run"
    )

    # Setup the logger
    nemo_logger = nl.NeMoLogger(
        log_dir="checkpoints/nemotron",
        extra_loggers=[neptune_logger]
    )

    train_loader = data.train_dataloader()

    # Start the training process
    llm.train(
        model=model,
        data=train_loader,
        trainer=trainer,
        log=nemo_logger,
        tokenizer=tokenizer,
        optim=opt,
    )
