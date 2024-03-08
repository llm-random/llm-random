import os
from typing import List, Optional


def make_singularity_mount_paths(setup_args: dict, training_args: dict) -> str:
    singularity_mount_paths = f"-B={os.getcwd()}:/llm-random"
    is_hf_datasets_cache_needed = (
        training_args["train_dataset_path"] is None
        or training_args["validation_dataset_path"] is None
    )
    singularity_mount_paths += (
        f",{setup_args['hf_datasets_cache']}:{setup_args['hf_datasets_cache']}"
        if is_hf_datasets_cache_needed
        else ""
    )
    singularity_mount_paths += (
        f",{training_args['train_dataset_path']}:{training_args['train_dataset_path']}"
        if training_args["train_dataset_path"] is not None
        else ""
    )
    singularity_mount_paths += (
        f",{training_args['validation_dataset_path']}:{training_args['validation_dataset_path']}"
        if training_args["validation_dataset_path"] is not None
        else ""
    )
    return singularity_mount_paths


def make_singularity_env_arguments(
    hf_datasets_cache_path: Optional[str],
    neptune_key: Optional[str],
    wandb_key: Optional[str],
) -> List[str]:
    variables_and_values = {}

    if hf_datasets_cache_path is not None:
        variables_and_values["HF_DATASETS_CACHE"] = hf_datasets_cache_path

    if neptune_key is not None:
        variables_and_values["NEPTUNE_API_TOKEN"] = neptune_key

    if wandb_key is not None:
        variables_and_values["WANDB_API_KEY"] = wandb_key

    return (
        ["--env", ",".join([f"{k}={v}" for k, v in variables_and_values.items()])]
        if len(variables_and_values) > 0
        else []
    )
