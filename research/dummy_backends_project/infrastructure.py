import os
import platform
import hashlib


from lizrd.grid.infrastructure import MachineBackend
from lizrd.grid.setup_arguments import make_singularity_mount_paths


class WriterBackend(MachineBackend):
    max_exp_time = 7 * 24 * 60 * 60

    def get_common_directory(self) -> str:
        return "/home/ubuntu/llm-random-group"

    def get_cache_path(self) -> str:
        return "/home/ubuntu/.cache"

    def get_grid_entrypoint(self) -> str:
        return "research/dummy_backends_project/entrypoint_writer.sh"

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
        print("USING OUR CUSTOM SUBPROCESS ARGS")
        return [
            slurm_command,
            f"--gres=gpu:a100:{setup_args['n_gpus']}",
            f"--array=0-{n_consecutive-1}%1",
            f"--cpus-per-task=1",
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


COMMON_DEFAULT_INFRASTRUCTURE_ARGS = {
    "gres": "gpu:1",
    "time": "1-00:00:00",
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
