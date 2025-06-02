
import torch
import nemo_run as run
from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer
import fiddle as fdl  # This is needed to build configs
import numpy as np
import random
import pytorch_lightning as pl
from nemo.collections.llm.modelopt import PruningConfig

from nemo.collections import llm
from nemo.collections.llm.modelopt import setup_trainer_and_restore_model_with_modelopt_spec

seq_length = 512
global_batch_size = 128
# max_steps = 15_533

dataset_path = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744842905/training/code/results_preprocessing/c4_en_train_part_00.jsonl_text_document" # hf auto 

# Not tig. emb.:
# 4xPO:
nemo_checkpoint = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1747302840/training/code/gimmi_checkpoint_pls"
# 1xPO:
# nemo_checkpoint = "asd"

target_ffn_hidden_size = 768
target_hidden_size = 768
# target_ffn_hidden_size = 256
# target_hidden_size = 256

target_num_attention_heads = 12
save_path = "prrruned_nyan"

import torch
import torch.nn.functional as F
import nemo_run as run
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
from nemo.collections.common.tokenizers import SentencePieceTokenizer, AutoTokenizer
import fiddle as fdl  # This is needed to build configs
import glob
import numpy as np
import random
import pytorch_lightning as pl
from omegaconf import OmegaConf
from nemo.lightning.io.pl import TrainerContext, ckpt_to_weights_subdir
from nemo.collections.nlp.models.language_modeling.megatron  import GPTModel
# from nemo.collections import llm.GPTModel GPTModelConfig


dsp_preambule = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744842905/training/code/results_preprocessing/"
dataset_path = [
    f"{dsp_preambule}c4_en_train_part_0{i}.jsonl_text_document" for i in range(0, 10)
] 

seed = 27


pl.seed_everything(seed, workers=True)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    print("Its PRUNNING BS ---------------------------------------------------------------") #dev 
        
    tokenizer_cfg = run.Config(AutoTokenizer, 
        pretrained_model_name="gpt2", #dev ? /net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/gpt2/ openai-community/gpt2 gpt2 GPT2Tokenizer  
    )
    tokenizer = fdl.build(tokenizer_cfg)  # ✅ Build it here

    data = llm.PreTrainingDataModule(
        paths=dataset_path,
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=global_batch_size,
        tokenizer=tokenizer,
        split="90,7,3",
        num_workers=32,
    )

    llm.prune(
        nemo_checkpoint=nemo_checkpoint,
        save_path=save_path,
        pruning_config=PruningConfig(
            target_ffn_hidden_size = target_ffn_hidden_size,
            target_hidden_size = target_hidden_size,
            target_num_attention_heads=target_num_attention_heads,
            target_num_query_groups=target_num_attention_heads,
            # target_num_layers=14,
            # drop_layers=[1,3,5,7,9,11,13]
        ),
        devices=1,
        num_nodes=1,
        tp_size=1,
        pp_size=1,
        num_train_samples=1024*4,
        data=data,
        legacy_ckpt=False,
    )

    model, trainer = setup_trainer_and_restore_model_with_modelopt_spec(
        save_path,
        devices = 1,
        tensor_model_parallel_size = 1,
        pipeline_model_parallel_size = 1,
        trainer_kwargs = {
            "max_steps":123,
            "log_every_n_steps":1,
            "val_check_interval":None,
            "limit_val_batches":10,
            "limit_test_batches":10,
            "accumulate_grad_batches":1,
            "enable_progress_bar":False,
        },
        inference_only = False,
    )

    print("model ----------------------------------------------------") #dev
    print(f"type(model) {type(model)} eop") #dev
    print(f"model(model) {model} eop") #def
    print("end model ----------------------------------------------------") #dev

    