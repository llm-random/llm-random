from enum import Enum
import os
import platform


class MachineBackend(Enum):
    ENTROPY = 1
    ATHENA = 2
    IDEAS = 3
    LOCAL = 4


COMMON_DEFUALT_INFRASTRUCTURE_ARGS = {
    "gres": "gpu:1",
    "time": "1-00:00:00",
    "n_gpus": 1,
    "cpus_per_gpu": 8,
    "nodelist": None,
    "hf_datasets_cache": f"~/.cache/huggingface/datasets",
    "runs_multiplier": 1,
    "runner": "research.conditional.train.cc_train",
    "interactive_debug_session": False,
}


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
        return None
    else:
        raise ValueError(f"Unknown machine backend: {machine_backend}")


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


def get_cluster_default_params(CLUSTER_NAME, dataset_type) -> dict:
    return {
        "train_dataset_path": get_default_train_dataset_path(
            CLUSTER_NAME, dataset_type
        ),
        "validation_dataset_path": get_default_validation_dataset_path(
            CLUSTER_NAME, dataset_type
        ),
        "common_directory": get_common_directory(CLUSTER_NAME),
        "hf_datasets_cache": get_cache_path(CLUSTER_NAME),
        "singularity_image": get_singularity_image(CLUSTER_NAME),
        "grid_entrypoint": get_grid_entrypoint(CLUSTER_NAME),
    }


def prepare_default_infrastructure_params(CLUSTER_NAME, dataset_type: str):
    infrastructure_params_dict = COMMON_DEFUALT_INFRASTRUCTURE_ARGS
    cluster_default_arg_dict = get_cluster_default_params(CLUSTER_NAME, dataset_type)
    infrastructure_params_dict.update(cluster_default_arg_dict)
    return infrastructure_params_dict
