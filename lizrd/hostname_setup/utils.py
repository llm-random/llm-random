from enum import Enum
import os
from typing import List, Optional
import platform


class MachineBackend(Enum):
    ENTROPY = 1
    ATHENA = 2
    IDEAS = 3
    LOCAL = 4


def get_machine_backend() -> MachineBackend:
    node = platform.uname().node
    if node == "asusgpu0":
        return MachineBackend.ENTROPY
    elif "athena" in node:
        return MachineBackend.ATHENA
    elif node == "login01":
        return MachineBackend.IDEAS
    else:
        return MachineBackend.LOCAL


def get_common_directory(machine_backend: MachineBackend) -> str:
    if machine_backend == MachineBackend.ATHENA:
        return "/net/pr2/projects/plgrid/plggllmeffi"
    elif machine_backend == MachineBackend.IDEAS:
        return "/raid/NFS_SHARE/llm-random"
    elif machine_backend == MachineBackend.ENTROPY:
        return "/home/jkrajewski_a100"
    else:
        return os.getenv("HOME")


def get_cache_path(machine_backend: MachineBackend) -> str:
    if machine_backend == MachineBackend.LOCAL:
        return f"{os.getenv('HOME')}/.cache/huggingface/datasets"
    elif machine_backend == MachineBackend.ATHENA:
        return f"/net/tscratch/people/{os.environ.get('USER')}/.cache"
    elif machine_backend == MachineBackend.ENTROPY:
        return "/local_storage_2/dataset_cache"
    else:
        common_dir = get_common_directory(machine_backend)
        return f"{common_dir}/.cache"


def get_singularity_image(machine_backend: MachineBackend) -> str:
    image_name = "sparsity_2024.02.06_16.14.02.sif"
    common_dir = get_common_directory(machine_backend)
    return f"{common_dir}/images/{image_name}"


def get_grid_entrypoint(machine_backend: MachineBackend) -> str:
    if machine_backend in [
        MachineBackend.ATHENA,
        MachineBackend.IDEAS,
        MachineBackend.ENTROPY,
    ]:
        return "lizrd/scripts/grid_entrypoint.sh"
    elif machine_backend in [MachineBackend.LOCAL]:
        raise ValueError(f"Local machine should use main function directly. ")
    else:
        raise ValueError(f"Unknown machine backend: {machine_backend}")


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


def get_default_train_dataset_path(CLUSTER_NAME: MachineBackend, dataset_type: str):
    if dataset_type == "c4":
        if CLUSTER_NAME == MachineBackend.IDEAS:
            return "/raid/NFS_SHARE/datasets/c4/train/c4_train"
        elif CLUSTER_NAME == MachineBackend.ENTROPY:
            return "/local_storage_2/llm-random/datasets/c4_train"

    return None


def get_default_validation_dataset_path(
    CLUSTER_NAME: MachineBackend, dataset_type: str
):
    if dataset_type == "c4":
        if CLUSTER_NAME == MachineBackend.IDEAS:
            return "/raid/NFS_SHARE/datasets/c4/validation/c4_validation"
        elif CLUSTER_NAME == MachineBackend.ENTROPY:
            return "/local_storage_2/llm-random/datasets/c4_validation"

    return None


def maybe_set_default_datasets_paths(
    grid: list[dict[str, str]], CLUSTER_NAME: MachineBackend
):
    for _, (training_args, _) in enumerate(grid):
        if training_args.get("train_dataset_path") is None:
            training_args["train_dataset_path"] = get_default_train_dataset_path(
                CLUSTER_NAME, training_args["dataset_type"]
            )
        if training_args.get("validation_dataset_path") is None:
            training_args[
                "validation_dataset_path"
            ] = get_default_validation_dataset_path(
                CLUSTER_NAME, training_args["dataset_type"]
            )


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
