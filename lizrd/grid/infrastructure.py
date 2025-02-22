import abc
import importlib
import os
import platform
import hashlib
from typing import Callable, Optional


from lizrd.grid.setup_arguments import make_singularity_mount_paths


class MachineBackend(abc.ABC):
    max_exp_time = 14 * 24 * 60 * 60

    def __init__(self, username=None):
        self.username = username

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
        n_consecutive: int = 1,
    ):
        pass

    @abc.abstractmethod
    def get_cemetery_directory(self):
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

    def get_runner_command(self, runner, runner_params, setup_args):
        if setup_args["n_nodes"] == 1:
            return ["python3", "-m", runner, *runner_params]
        else:  # we use torchrun for multi-node
            runner = (
                runner.replace(".", "/") + ".py"
            )  # we need a regular path for torchrun
            return [
                "torchrun",
                f"--nnodes={setup_args['n_nodes']}",
                f"--nproc_per_node={setup_args['n_gpus'] // setup_args['n_nodes']}",
                "--rdzv_id",
                "__RANDOM__",
                "--rdzv_backend",
                "c10d",
                "--rdzv_endpoint",
                "__HEAD_NODE_IP__:29500",
                runner,
                *runner_params,
            ]


class AthenaBackend(MachineBackend):
    max_exp_time = 2 * 24 * 60 * 60

    def get_default_train_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/net/tscratch/people/plgkciebiera/datasets/c4/train"
        elif dataset_type == "fineweb-edu":
            return (
                "/net/tscratch/people/plgmaciejpioro/datasets/fineweb-edu/train/train"
            )
        return super().get_default_train_dataset_path(dataset_type)

    def get_default_validation_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/net/tscratch/people/plgkciebiera/datasets/c4/validation"
        elif dataset_type == "fineweb-edu":
            return (
                "/net/tscratch/people/plgmaciejpioro/datasets/fineweb-edu/train/train"
            )
        return super().get_default_train_dataset_path(dataset_type)

    def get_common_directory(self) -> str:
        return "/net/pr2/projects/plgrid/plggllmeffi"

    def get_cache_path(self) -> str:
        return f"/net/tscratch/people/{self.username}/.cache"

    def get_grid_entrypoint(self) -> str:
        return "lizrd/grid/grid_entrypoint.sh"

    def get_cemetery_directory(self):
        return (
            f"/net/pr2/projects/plgrid/plggllmeffi/{self.username}/llm_random_cemetery"
        )

    def get_subprocess_args(
        self,
        slurm_command,
        setup_args,
        training_args,
        singularity_env_arguments,
        runner_params,
        n_consecutive: int = 1,
    ):
        return [
            slurm_command,
            f"--nodes={setup_args['n_nodes']}",
            f"--gpus={setup_args['n_gpus']}",
            f"--gpus-per-node={setup_args['n_gpus'] // setup_args['n_nodes']}",
            f"--array=0-{n_consecutive-1}%1",
            "--partition=plgrid-gpu-a100",
            f"--cpus-per-gpu={setup_args['cpus_per_gpu']}",
            f"--mem={max(125, setup_args['mem_per_gpu']*setup_args['n_gpus'])}G",
            "--account=plgllmefficont-gpu-a100",
            f"--job-name={training_args['name']}",
            f"--time={setup_args['time']}",
            f"{setup_args['grid_entrypoint']}",
            "srun",
            "singularity",
            "run",
            "--bind=/net:/net",
            *singularity_env_arguments,
            make_singularity_mount_paths(setup_args, training_args),
            "--nv",
            setup_args["singularity_image"],
            *self.get_runner_command(setup_args["runner"], runner_params, setup_args),
        ]


class HeliosBackend(MachineBackend):
    max_exp_time = 2 * 24 * 60 * 60

    def get_default_train_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/net/scratch/hscra/plgrid/plgmaciejpioro/c4/train"
        elif dataset_type == "fineweb-edu":
            return "/net/scratch/hscra/plgrid/plgmaciejpioro/fineweb-edu/train/train"
        return super().get_default_train_dataset_path(dataset_type)

    def get_default_validation_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/net/scratch/hscra/plgrid/plgmaciejpioro/c4/validation"
        elif dataset_type == "fineweb-edu":
            return "/net/scratch/hscra/plgrid/plgmaciejpioro/fineweb-edu/train/train"
        return super().get_default_train_dataset_path(dataset_type)

    def get_common_directory(self) -> str:
        return "/net/storage/pr3/plgrid/plggllmeffi"

    def get_cache_path(self) -> str:
        return f"/net/scratch/hscra/plgrid/plgmaciejpioro/{self.username}/.cache"

    def get_grid_entrypoint(self) -> str:
        return "lizrd/grid/grid_entrypoint_helios.sh"

    def get_cemetery_directory(self):
        return (
            f"/net/storage/pr3/plgrid/plggllmeffi/{self.username}/llm_random_cemetery"
        )

    def get_subprocess_args(
        self,
        slurm_command,
        setup_args,
        training_args,
        singularity_env_arguments,
        runner_params,
        n_consecutive: int = 1,
    ):
        assert (
            setup_args["n_gpus"] % 4 == 0
        ), "Helios only supports using whole nodes (cf. https://docs.cyfronet.pl/display/~plgpawlik/Helios)"

        return [
            slurm_command,
            f"--nodes={setup_args['n_nodes']}",
            f"--gpus={setup_args['n_gpus']}",
            f"--gpus-per-node={setup_args['n_gpus'] // setup_args['n_nodes']}",
            f"--array=0-{n_consecutive-1}%1",
            "--partition=plgrid-gpu-gh200",
            "--exclusive",  # request all non-gpu resources on node
            "--account=plgllmefficont-gpu-gh200",
            f"--job-name={training_args['name']}",
            f"--time={setup_args['time']}",
            f"{setup_args['grid_entrypoint']}",
            "srun",
            *self.get_runner_command(setup_args["runner"], runner_params, setup_args),
        ]


class IdeasBackend(MachineBackend):
    max_exp_time = 7 * 24 * 60 * 60

    def get_common_directory(self) -> str:
        return "/raid/NFS_SHARE/llm-random"

    def get_cache_path(self) -> str:
        common_dir = self.get_common_directory()
        return f"{common_dir}/.cache"

    def get_grid_entrypoint(self) -> str:
        return "lizrd/grid/grid_entrypoint.sh"

    def get_default_train_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/raid/NFS_SHARE/datasets/c4/train/c4_train"
        return super().get_default_train_dataset_path(dataset_type)

    def get_default_validation_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/raid/NFS_SHARE/datasets/c4/validation/c4_validation"
        return super().get_default_train_dataset_path(dataset_type)

    def get_cemetery_directory(self):
        return f"~/llm_random_cemetery"

    def get_subprocess_args(
        self,
        slurm_command,
        setup_args,
        training_args,
        singularity_env_arguments,
        runner_params,
        n_consecutive: int = 1,
    ):
        assert setup_args["n_nodes"] == 1, "multi-node on Ideas not implemented"
        return [
            slurm_command,
            f"--gres=gpu:ampere:{setup_args['n_gpus']}",
            f"--array=0-{n_consecutive-1}%1",
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
            *self.get_runner_command(setup_args["runner"], runner_params, setup_args),
        ]


class EntropyBackend(MachineBackend):
    max_exp_time = 14 * 24 * 60 * 60

    def get_common_directory(self) -> str:
        return "/home/jkrajewski_a100"

    def get_cache_path(self) -> str:
        return "/local_storage_2/dataset_cache"

    def get_grid_entrypoint(self) -> str:
        return "lizrd/grid/grid_entrypoint.sh"

    def get_default_train_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/local_storage_2/llm-random/datasets/c4_train"
        return super().get_default_train_dataset_path(dataset_type)

    def get_default_validation_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/local_storage_2/llm-random/datasets/c4_validation"
        return super().get_default_train_dataset_path(dataset_type)

    def get_cemetery_directory(self):
        return f"~/llm_random_cemetery"

    def get_subprocess_args(
        self,
        slurm_command,
        setup_args,
        training_args,
        singularity_env_arguments,
        runner_params,
        n_consecutive: int = 1,
    ):
        assert setup_args["n_nodes"] == 1, "multi-node on Entropy not implemented"
        return [
            slurm_command,
            "--partition=a100",
            f"--gres=gpu:a100:{setup_args['n_gpus']}",
            f"--array=0-{n_consecutive-1}%1",
            f"--cpus-per-gpu={setup_args['cpus_per_gpu']}",
            f"--mem={max(125, setup_args['mem_per_gpu']*setup_args['n_gpus'])}G",
            f"--job-name={training_args['name']}",
            f"--time={setup_args['time']}",
            f"{setup_args['grid_entrypoint']}",
            "singularity",
            "run",
            *singularity_env_arguments,
            make_singularity_mount_paths(setup_args, training_args),
            "-B /local_storage_1",
            "-B /local_storage_2",
            "--nv",
            setup_args["singularity_image"],
            *self.get_runner_command(setup_args["runner"], runner_params, setup_args),
        ]


class WriterBackend(MachineBackend):
    max_exp_time = 7 * 24 * 60 * 60

    def get_common_directory(self) -> str:
        return "/home/ubuntu/llm-random-group"

    def get_cache_path(self) -> str:
        return "/home/ubuntu/.cache"

    def get_grid_entrypoint(self) -> str:
        return "lizrd/grid/grid_entrypoint.sh"

    def get_default_train_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/home/ubuntu/llm-random-group/datasets/c4_train"
        return super().get_default_train_dataset_path(dataset_type)

    def get_default_validation_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/home/ubuntu/llm-random-group/datasets/c4_validation"
        return super().get_default_train_dataset_path(dataset_type)

    def get_cemetery_directory(self):
        return "/home/ubuntu/llm-random-group/llm-random-cemetery"

    def get_subprocess_args(
        self,
        slurm_command,
        setup_args,
        training_args,
        singularity_env_arguments,
        runner_params,
        n_consecutive: int = 1,
    ):
        assert setup_args["n_nodes"] == 1, "multi-node on Writer not implemented"
        return [
            slurm_command,
            f"--gres=gpu:a100:{setup_args['n_gpus']}",
            f"--array=0-{n_consecutive-1}%1",
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
            *self.get_runner_command(setup_args["runner"], runner_params, setup_args),
        ]


class AWS1Backend(MachineBackend):
    def get_common_directory(self) -> str:
        return "/home/ubuntu/"

    def get_cache_path(self) -> str:
        return "/home/ubuntu/.cache"

    def get_grid_entrypoint(self) -> str:
        return "lizrd/grid/grid_entrypoint.sh"

    def get_default_train_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/data/datasets/data/train"
        return super().get_default_train_dataset_path(dataset_type)

    def get_default_validation_dataset_path(self, dataset_type: str):
        if dataset_type == "c4":
            return "/data/datasets/data/validation"
        return super().get_default_train_dataset_path(dataset_type)

    def get_cemetery_directory(self):
        return "/home/ubuntu/llm-random-cemetery"

    def get_singularity_image(self) -> str:
        return "/data/sparsity_2024.02.06_16.14.02.sif"

    def get_subprocess_args(
        self,
        slurm_command,
        setup_args,
        training_args,
        singularity_env_arguments,
        runner_params,
        n_consecutive: int = 1,
    ):
        assert setup_args["n_nodes"] == 1, "multi-node on AWS1 not implemented"
        if n_consecutive != 1:
            raise Exception(
                "Cluster does not support checkpoint manager feature. Works only with slurm system."
            )
        return [
            "singularity",
            "run",
            *singularity_env_arguments,
            make_singularity_mount_paths(setup_args, training_args),
            "--nv",
            setup_args["singularity_image"],
            *self.get_runner_command(setup_args["runner"], runner_params, setup_args),
        ]


class LocalBackend(MachineBackend):
    def get_common_directory(self) -> str:
        return os.getenv("HOME")

    def get_cache_path(self) -> str:
        return f"{os.getenv('HOME')}/.cache/huggingface/datasets"

    def get_grid_entrypoint(self) -> str:
        return None

    def get_cemetery_directory(self):
        raise Exception("Local machine should not use cemetery")

    def get_subprocess_args(
        self,
        slurm_command,
        setup_args,
        training_args,
        singularity_env_arguments,
        runner_params,
        n_consecutive: int = 1,
    ):
        raise Exception("Local machine should use main function")


COMMON_DEFAULT_INFRASTRUCTURE_ARGS = {
    "gres": "gpu:1",
    "time": "1-00:00:00",
    "n_nodes": 1,
    "n_gpus": 1,
    "cpus_per_gpu": 8,
    "mem_per_gpu": 125,
    "nodelist": None,
    "hf_datasets_cache": f"~/.cache/huggingface/datasets",
    "runs_multiplier": 1,
    "runner": "research.conditional.train.cc_train",
    "interactive_debug_session": False,
}


def get_machine_backend(node=None, connection=None) -> MachineBackend:
    if node is None:
        node = platform.uname().node
    username = os.environ.get("USER") if connection is None else connection.user
    if node == "asusgpu0":
        return EntropyBackend(username)
    elif "athena" in node:
        return AthenaBackend(username)
    elif node == "login01":
        return IdeasBackend(username)
    elif (
        hashlib.sha256(node.encode()).hexdigest()
        == "53cb84932d7356993300456b370e2e796c68d28be3e584c17f5eeacad9d36a12"
    ):  # no need for anyone to know the hostname :)
        return WriterBackend(username)
    elif (
        hashlib.sha256(node.encode()).hexdigest()
        == "b7ac4f788a9ebbb762abd91b07030d07fe9e41b9a6b2fcb25062bbb26edc60e3"
    ):  # no need for anyone to know the hostname :)
        return AWS1Backend(username)
    elif "helios" in node:
        return HeliosBackend(username)
    else:
        return LocalBackend(username)


def resolve_get_machine_backend_function(
    alternative_module_name: Optional[str] = None,
) -> Callable[..., MachineBackend]:
    if alternative_module_name is not None:
        module = importlib.import_module(alternative_module_name)
        alternative_get_machine_backend = getattr(module, "get_machine_backend")
        assert alternative_get_machine_backend is not None
        return alternative_get_machine_backend
    else:
        return get_machine_backend
