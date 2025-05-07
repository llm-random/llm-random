from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.llm.modelopt import setup_trainer_and_restore_model_with_modelopt_spec
# def setup_trainer_and_restore_model_with_modelopt_spec(
#     model_path: str,
#     tensor_model_parallel_size: int = 1,
#     pipeline_model_parallel_size: int = 1,
#     num_layers_in_first_pipeline_stage: int | None = None,
#     num_layers_in_last_pipeline_stage: int | None = None,
#     devices: int = 1,
#     num_nodes: int = 1,
#     inference_only: bool = True,
#     tokenizer_path: str | None = None,
#     legacy_ckpt: bool = False,
#     strategy_kwargs: dict | None = None,
#     trainer_kwargs: dict | None = None,
#     model_config_overrides: dict | None = None,
# ) -> tuple[llm.GPTModel, nl.Trainer]:
# from omegaconf import OmegaCaonf

# checkpoin_file = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1744878751/training/code/checkpoints/nemotron/default/2025-04-17_10-33-24/checkpoints/default--None=0.0000-epoch=0-consumed_samples=7680000.0/weights"  # <- path to folder
load_checkpoint = "/net/storage/pr3/plgrid/plggllmeffi/plgmstefaniak/nemo_cementary/nemo_2_training_experiment/nemo_2_training_experiment_1745578390/training/code/gimmi_checkpoint_pls"

# # Load your base config
# config = OmegaConf.load("path/to/your_config.yaml")

# # Adjust for loading sharded checkpoint
# config.model.restore_from_path = "path/to/your/checkpoint/dir"  # This is the folder with .distcp files
# config.trainer.devices = 1  # Inference or fine-tune on single GPU
# config.trainer.precision = 16  # Or 32 if you trained in fp32
# config.trainer.strategy = "ddp"  # Or "auto", depends on what you're doing

# # Override TP if needed
# config.model.tensor_model_parallel_size = 4

# # Load the model
# model = MegatronGPTModel.restore_from(config)


# pretrained_cfg = MegatronGPTModel.restore_from(
#     restore_path=checkpoin_file,
#     # trainer=trainer,
#     # return_config=True,
#     # save_restore_connector=save_restore_connector,
# )

model, trainer = setup_trainer_and_restore_model_with_modelopt_spec(
    load_checkpoint,
    devices = 4,
)


print("type(pretrained_cfg)-----------------------------------------------------")
print(type(model))
print(model)

print(type(trainer))
print(trainer)