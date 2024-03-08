import abc
import os
import platform

from lizrd.grid.setup_arguments import make_singularity_mount_paths


class MachineBackend(abc.ABC):
    @abc.abstractmethod
    def get_common_directory(self) -> str:
        pass

    @abc.abstractmethod
    def get_cache_path(self) -> str:
        pass

    @abc.abstractmethod
    def get_grid_entrypoint(self) -> str:
        pass

    @abc.abstractmethod
    def get_subprocess_args(
        self,
        slurm_command,
        setup_args,
        training_args,
        singularity_env_arguments,
        runner_params,
    ):
        pass

    def get_singularity_image(self) -> str:
        image_name = "sparsity_2024.02.06_16.14.02.sif"
        common_dir = self.get_common_directory()
        return f"{common_dir}/images/{image_name}"

    def get_default_train_dataset_path(self, dataset_type: str):
        return None

    def get_default_validation_dataset_path(self, dataset_type: str):
        return None

    def get_cluster_default_params(self, dataset_type) -> dict:
        return {
            "train_dataset_path": self.get_default_train_dataset_path(dataset_type),
            "validation_dataset_path": self.get_default_validation_dataset_path(
                dataset_type
            ),
            "common_directory": self.get_common_directory(),
            "hf_datasets_cache": self.get_cache_path(),
            "singularity_image": self.get_singularity_image(),
            "grid_entrypoint": self.get_grid_entrypoint(),
        }

    def prepare_default_infrastructure_params(self, dataset_type: str):
        infrastructure_params_dict = COMMON_DEFAULT_INFRASTRUCTURE_ARGS
        cluster_default_arg_dict = self.get_cluster_default_params(dataset_type)
        infrastructure_params_dict.update(cluster_default_arg_dict)
        return infrastructure_params_dict

    def get_runner_command(self, runner, runner_params):
        return ["python3", "-m", runner, *runner_params]


class AthenaBackend(MachineBackend):
    def get_common_directory(self) -> str:
        return "/net/pr2/projects/plgrid/plggllmeffi"

    def get_cache_path(self) -> str:
        return f"/net/tscratch/people/{os.environ.get('USER')}/.cache"

    def get_grid_entrypoint(self) -> str:
        return "lizrd/scripts/grid_entrypoint.sh"

    def get_subprocess_args(
        self,
        slurm_command,
        setup_args,
        training_args,
        singularity_env_arguments,
        runner_params,
    ):
        return [
            slurm_command,
            f"--gres=gpu:{setup_args['n_gpus']}",
            "--partition=plgrid-gpu-a100",
            f"--mem={max(125, setup_args['mem_per_gpu']*setup_args['n_gpus'])}G",
            "--account=plgsubslearnath-gpu-a100",
            f"--job-name={training_args['name']}",
            f"--time={setup_args['time']}",
            f"{setup_args['grid_entrypoint']}",
            "singularity",
            "run",
            "--bind=/net:/net",
            *singularity_env_arguments,
            make_singularity_mount_paths(setup_args, training_args),
            "--nv",
            setup_args["singularity_image"],
            *self.get_runner_command(setup_args["runner"], runner_params),
        ]


class IdeasBackend(MachineBackend):
    def get_common_directory(self) -> str:
        return "/raid/NFS_SHARE/llm-random"

    def get_cache_path(self) -> str:
        common_dir = self.get_common_directory()
        return f"{common_dir}/.cache"

    def get_grid_entrypoint(self) -> str:
        return "lizrd/scripts/grid_entrypoint.sh"

    def get_default_train_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/raid/NFS_SHARE/datasets/c4/train/c4_train"
        return super().get_default_train_dataset_path(dataset_type)

    def get_default_validation_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/raid/NFS_SHARE/datasets/c4/validation/c4_validation"
        return super().get_default_train_dataset_path(dataset_type)

    def get_subprocess_args(
        self,
        slurm_command,
        setup_args,
        training_args,
        singularity_env_arguments,
        runner_params,
    ):
        return [
            slurm_command,
            f"--gres=gpu:ampere:{setup_args['n_gpus']}",
            f"--cpus-per-gpu={setup_args['cpus_per_gpu']}",
            f"--job-name={training_args['name']}",
            f"--time={setup_args['time']}",
            f"--mem={max(125, setup_args['mem_per_gpu']*setup_args['n_gpus'])}G",
            setup_args["nodelist"],
            f"{setup_args['grid_entrypoint']}",
            "singularity",
            "run",
            *singularity_env_arguments,
            make_singularity_mount_paths(setup_args, training_args),
            "--nv",
            setup_args["singularity_image"],
            *self.get_runner_command(setup_args["runner"], runner_params),
        ]


class EntropyBackend(MachineBackend):
    def get_common_directory(self) -> str:
        return "/home/jkrajewski_a100"

    def get_cache_path(self) -> str:
        return "/local_storage_2/dataset_cache"

    def get_grid_entrypoint(self) -> str:
        return "lizrd/scripts/grid_entrypoint.sh"

    def get_default_train_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/local_storage_2/llm-random/datasets/c4_train"
        return super().get_default_train_dataset_path(dataset_type)

    def get_default_validation_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/local_storage_2/llm-random/datasets/c4_validation"
        return super().get_default_train_dataset_path(dataset_type)

    def get_subprocess_args(
        self,
        slurm_command,
        setup_args,
        training_args,
        singularity_env_arguments,
        runner_params,
    ):
        return [
            slurm_command,
            "--partition=a100",
            f"--gres=gpu:a100:{setup_args['n_gpus']}",
            f"--cpus-per-gpu={setup_args['cpus_per_gpu']}",
            f"--mem={max(125, setup_args['mem_per_gpu']*setup_args['n_gpus'])}G",
            f"--job-name={training_args['name']}",
            f"--time={setup_args['time']}",
            f"{setup_args['grid_entrypoint']}",
            "singularity",
            "run",
            *singularity_env_arguments,
            make_singularity_mount_paths(setup_args, training_args),
            "--nv",
            setup_args["singularity_image"],
            *self.get_runner_command(setup_args["runner"], runner_params),
        ]


class LocalBackend(MachineBackend):
    def get_common_directory(self) -> str:
        return os.getenv("HOME")

    def get_cache_path(self) -> str:
        return f"{os.getenv('HOME')}/.cache/huggingface/datasets"

    def get_grid_entrypoint(self) -> str:
        return None

    def get_subprocess_args(
        self,
        slurm_command,
        setup_args,
        training_args,
        singularity_env_arguments,
        runner_params,
    ):
        raise Exception("Local machine should use main function")


COMMON_DEFAULT_INFRASTRUCTURE_ARGS = {
    "gres": "gpu:1",
    "time": "1-00:00:00",
    "n_gpus": 1,
    "cpus_per_gpu": 8,
    "mem_per_gpu": 125,  # Entropy only for now
    "nodelist": None,
    "hf_datasets_cache": f"~/.cache/huggingface/datasets",
    "runs_multiplier": 1,
    "runner": "research.conditional.train.cc_train",
    "interactive_debug_session": False,
}


def get_machine_backend() -> MachineBackend:
    node = platform.uname().node
    if node == "asusgpu0":
        return EntropyBackend()
    elif "athena" in node:
        return AthenaBackend()
    elif node == "login01":
        return IdeasBackend()
    else:
        return LocalBackend()
