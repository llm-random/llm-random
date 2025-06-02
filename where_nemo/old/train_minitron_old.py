
import torch
import nemo_run as run
from nemo.collections import llm
from nemo.collections.common.tokenizers import SentencePieceTokenizer, AutoTokenizer
from pytorch_lightning.loggers import NeptuneLogger
import fiddle as fdl  # This is needed to build configs
import numpy as np
import random
import pytorch_lightning as pl
from nemo.collections.llm.modelopt import PruningConfig

from nemo.collections import llm
from nemo.collections.llm.modelopt.recipes import prune_recipe
from nemo.collections.llm.modelopt import setup_trainer_and_restore_model_with_modelopt_spec

seq_length = 512
global_batch_size = 128
# max_steps = 15_533

dataset_path = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744842905/training/code/results_preprocessing/c4_en_train_part_00.jsonl_text_document" # hf auto 

# OLD:
# nemo_checkpoint = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1745255133/training/code/my_model_single_file.ckpt"
# nemo_checkpoint = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1745255133/training/code/checkpoints/nemotron/default/2025-04-21_19-06-26/checkpoints/default--None=0.0000-epoch=0-consumed_samples=7680000.0"
# nemo_checkpoint = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1745349253/training/code/my_my_my_checkpoint_pls"
# nemo_checkpoint = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1745578390/training/code/gimmi_checkpoint_pls"
# 4xPO:
# nemo_checkpoint = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1745716918/training/code/gimmi_checkpoint_pls"
# 1xPO:
# nemo_checkpoint = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1745714456/training/code/gimmi_checkpoint_pls"

# Not tig. emb.:
# 4xPO:
# nemo_checkpoint = "asd"
# 1xPO:
# nemo_checkpoint = "asd"

target_ffn_hidden_size = 768
target_hidden_size = 768
# target_ffn_hidden_size = 256
# target_hidden_size = 256

save_path = "prrruned_nyan"

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


dsp_preambule = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744842905/training/code/results_preprocessing/"
dataset_path = [
    f"{dsp_preambule}c4_en_train_part_0{i}.jsonl_text_document" for i in range(0, 10)
] 

# num_layers=16
# num_attention_heads=16
# hidden_size=1024
# ffn_hidden_size=1024
# seq_length=seq_length
# init_method_std=0.02
# hidden_dropout=0.1
# attention_dropout=0.1
# layernorm_epsilon=1e-5
# make_vocab_size_divisible_by=1
# tags = ["projected_dis", "1ff", "dm1024", "nemo", "vocab_size_1"]
seed = 27
# base_lr = 0.001
# final_lr_fraction = 0.03
# warmup_percent = 0.01
# activation_func = F.silu
# weight_decay = 0.1
# clip_grad = 0.5
# position_embedding_type = "rope"


pl.seed_everything(seed, workers=True)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# def before_exps_logs(neptune_logger):
#     # Log general environment info
#     neptune_logger.experiment["environment"] = {
#         "working_directory": os.getenv("HOST_PATH", "UNKNOWN"),
#         "python_version": sys.version,
#         "platform": platform.platform(),
#         "torch_version": torch.__version__,
#         "cuda_available": torch.cuda.is_available(),
#         "num_gpus": torch.cuda.device_count(),
#     }

#     # Log Git commit if in repo
#     try:
#         commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
#         neptune_logger.experiment["git/commit"] = commit
#     except Exception:
#         neptune_logger.experiment["git/commit"] = "Not a git repo or error"

#     # Log all hyperparameters
#     hyperparams = {
#         "seed": seed,
#         "seq_length": seq_length,
#         "global_batch_size": global_batch_size,
#         "micro_batch_size": micro_batch_size,
#         "max_steps": max_steps,
#         "num_layers": num_layers,
#         "num_attention_heads": num_attention_heads,
#         "hidden_size": hidden_size,
#         "ffn_hidden_size": ffn_hidden_size,
#         "init_method_std": init_method_std,
#         "hidden_dropout": hidden_dropout,
#         "attention_dropout": attention_dropout,
#         "layernorm_epsilon": layernorm_epsilon,
#         "make_vocab_size_divisible_by": make_vocab_size_divisible_by,
#         "base_lr": base_lr,
#         "final_lr_fraction": final_lr_fraction,
#         "warmup_percent": warmup_percent,
#         "dataset_path": dataset_path,
#         "tags": tags,
#         "activation_func": str(activation_func),
#         "weight_decay": weight_decay,
#         "clip_grad": clip_grad,
#         "position_embedding_type": position_embedding_type,
#     }
#     neptune_logger.experiment["hyperparameters"] = hyperparams

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

    