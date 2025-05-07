from pathlib import Path

# To suppress BF16 compile related issue in the CI runs with turing/V100
import torch._dynamo
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

torch._dynamo.config.suppress_errors = True

mp.set_start_method("spawn", force=True)

seq_length = 512
global_batch_size = 512
micro_batch_size = 128
# max_steps = 15_533
max_steps = 100

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

# pl.seed_everything(seed, workers=True)
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)


if __name__ == "__main__":
    

    print("Its TRAINING BS ---------------------------------------------------------------") #dev 
    
    # tokenizer_cfg = run.Config(SentencePieceTokenizer, model_path="/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/tokenizer.model")
    # tokenizer_cfg = run.Config(AutoTokenizer, 
    #     pretrained_model_name="gpt2", #dev ? /net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/datasets/c4/tokenizers/gpt2/ openai-community/gpt2 gpt2 GPT2Tokenizer  
    # )
    # tokenizer = fdl.build(tokenizer_cfg)  # ✅ Build it here

    # data = llm.PreTrainingDataModule(
    #     paths=[dataset_path],
    #     seq_length=seq_length,
    #     global_batch_size=global_batch_size,
    #     micro_batch_size=micro_batch_size,
    #     tokenizer=tokenizer,
    #     split="99,8,2",
    #     num_workers=32,
    #     index_mapping_dir=dataset_path
    # )

    # from: where_nemo/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py !!!

    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.max_epochs=null \
    trainer.max_steps=300000 \
    trainer.val_check_interval=300 \
    trainer.log_every_n_steps=50 \
    trainer.limit_val_batches=50 \
    trainer.limit_test_batches=50 \
    trainer.accumulate_grad_batches=1 \
    trainer.precision=16 \
    
    model.micro_batch_size=6 \
    model.global_batch_size=192 \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    model.max_position_embeddings=1024 \
    model.encoder_seq_length=1024 \
    model.hidden_size=768 \
    model.ffn_hidden_size=3072 \
    model.num_layers=12 \
    model.num_attention_heads=12 \
    model.init_method_std=0.021 \
    model.hidden_dropout=0.1 \
    model.layernorm_epsilon=1e-5 \
    model.tokenizer.vocab_file=gpt2-vocab.json \
    model.tokenizer.merge_file=gpt2-merges.txt \
    model.data.data_prefix=[1.0,hfbpe_gpt_training_data_text_document] \
    model.data.num_workers=2 \
    model.data.seq_length=1024 \
    model.data.splits_string=\'980,10,10\' \
    model.optim.name=fused_adam \
    model.optim.lr=6e-4 \
    model.optim.betas=[0.9,0.95] \
    model.optim.weight_decay=0.1 \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=750 \
    model.optim.sched.constant_steps=80000 \
    model.optim.sched.min_lr=6e-5 \
    
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_checkpoint_callback=True \
    exp_manager.checkpoint_callback_params.monitor=val_loss \
    exp_manager.checkpoint_callback_params.save_top_k=3 \
    exp_manager.checkpoint_callback_params.mode=min \
    exp_manager.checkpoint_callback_params.always_save_nemo=False

    model_cfg = OmegaConf.create({
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "ffn_hidden_size": ffn_hidden_size,
        "num_attention_heads": num_attention_heads,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "max_position_embeddings": seq_length,
        "init_method_std": init_method_std,
        "hidden_dropout": hidden_dropout,
        "attention_dropout": attention_dropout,
        "layernorm_epsilon": layernorm_epsilon,
        "precision": 16,
        "megatron_amp_O2": False,
        "transformer_engine": False,
        # "data": {
        #     "vocab_file": "path/to/vocab_file.model",
        #     "seq_length": 512,
        # },
    })

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    # Continual training
    model = MegatronGPTModel(cfg.model, trainer)

    trainer.fit(model)

    # # Initialize the training strategy
    # strategy = nl.MegatronStrategy(
    #     tensor_model_parallel_size=1,
    #     pipeline_model_parallel_size=1,
    #     pipeline_dtype=torch.bfloat16,
    # )

    # # Setup the optimizer
    # opt_config = OptimizerConfig(
    #     optimizer='adam',
    #     lr=base_lr,
    #     weight_decay=0.1,
    #     bf16=True,
    # )

    # lr_scheduler = CosineAnnealingScheduler(
    #     max_steps=max_steps,
    #     warmup_steps=int(max_steps*warmup_percent),
    #     constant_steps=0,
    #     min_lr=final_lr_fraction*base_lr
    # )

    # opt = nl.MegatronOptimizerModule(config=opt_config, lr_scheduler=lr_scheduler)


    # checkpoint_callback = NeMoModelCheckpoint(
    #     always_save_nemo=True,
    #     save_nemo_on_train_end=True,
    # )

    # trainer = nl.Trainer(
    #     devices=4,
    #     max_steps=max_steps,
    #     accelerator="gpu",
    #     strategy=strategy,
    #     plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    #     log_every_n_steps=100,
    #     val_check_interval=1000,
    #     limit_val_batches=50,
    #     limit_test_batches=50,
    #     accumulate_grad_batches=1,
    #     callbacks=[checkpoint_callback],  # <-- add this
    # )

    # neptune_logger = NeptuneLogger(
    #     api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZDg5ZTI2ZS00MWUyLTRkMTUtYTEzMC01OTVhYzE1ZWVmYzIifQ==",
    #     project="pmtest/llm-random",
    #     tags=tags,
    #     log_model_checkpoints=False,
    #     name="PD_nemo"
    # )
    # nemo_logger = nl.NeMoLogger(
    #     log_dir="checkpoints/nemotron",
    #     extra_loggers=[neptune_logger]
    # )

    # llm.train(
    #     model=model,
    #     data=data,
    #     trainer=trainer,
    #     log=nemo_logger,
    #     tokenizer="data",
    #     optim=opt,
    # )

    # model.to_file("asd")

    # # save_on_rank0(model, "my_model_single_file.ckpt")

