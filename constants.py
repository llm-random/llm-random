from enum import Enum
import os


class MachineBackend(str, Enum):
    ENTROPY = "entropy"
    ATHENA = "athena"
    IDEAS = "ideas"
    LOCAL = "local"


REPOSITORY_NAME = "llm-random"
GRID_ENTRYPOINT = "lizrd/scripts/grid_entrypoint.sh"
IMAGE_NAME = "sparsity_2024.02.06_16.14.02.sif"
clusters = {
    f"{MachineBackend.ENTROPY.value}": {
        "partition": "a100",
        "common_directory": "/home/jkrajewski_a100",
        "cache_path": "/local_storage_2/dataset_cache",
        "datasets": {
            "c4": {
                "train": "/local_storage_2/datasets/c4/train/c4_train",
                "validation": "/local_storage_2/datasets/c4/validation/c4_validation",
            }
        },
    },
    f"{MachineBackend.ATHENA.value}": {
        "partition": "plgrid-gpu-a100",
        "account": "plgplggllmeffi-gpu-a100",
        "common_directory": "/net/pr2/projects/plgrid/plggllmeffi",
        "cache_path": f"/net/tscratch/people/{os.environ.get('USER')}/.cache",
    },
    f"{MachineBackend.IDEAS.value}": {
        "cache_path": "/raid/NFS_SHARE/llm-random/.cache",
        "common_directory": "/raid/NFS_SHARE/llm-random",
        "datasets": {
            "c4": {
                "train": "/raid/NFS_SHARE/datasets/c4/train/c4_train",
                "validation": "/raid/NFS_SHARE/datasets/c4/validation/c4_validation",
            }
        },
    },
}
