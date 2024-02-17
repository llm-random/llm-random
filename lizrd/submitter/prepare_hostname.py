import platform
from typing import List, Optional

from constants import (
    GRID_ENTRYPOINT,
    REPOSITORY_NAME,
    clusters,
    IMAGE_NAME,
    MachineBackend,
)


class HostInfraConfig:
    def __init__(
        self,
        interactive_session,
        job_repository_path,
        n_gpus,
        cpus_per_gpu,
        time,
        job_name,
        singularity_image,
        hf_datasets_cache,
        neptune_key,
        wandb_key,
        dataset_name,
        nodelist=None,
        train_dataset_path=None,
        validation_dataset_path=None,
    ):
        self.node = self.get_machine_backend()
        self.slurm_command = "srun" if interactive_session else "sbatch"
        self.n_gpus = n_gpus
        self.cpus_per_gpu = cpus_per_gpu
        self.time = time
        self.job_name = job_name
        self.singularity_image = singularity_image
        self.grid_entrypoint = GRID_ENTRYPOINT
        self.nodelist = nodelist
        self.train_dataset_path = train_dataset_path
        self.validation_dataset_path = validation_dataset_path

        if train_dataset_path is None or validation_dataset_path is None:
            (
                default_train_dataset_path,
                default_validation_dataset_path,
            ) = self.dataset_paths(self.node, dataset_name)
            self.train_dataset_path = (
                default_train_dataset_path
                if train_dataset_path is None
                else train_dataset_path
            )
            self.validation_dataset_path = (
                default_validation_dataset_path
                if validation_dataset_path is None
                else validation_dataset_path
            )

        self.singularity_env_arguments = self.make_singularity_env_arguments(
            hf_datasets_cache_path=hf_datasets_cache,
            neptune_key=neptune_key,
            wandb_key=wandb_key,
        )

        self.singularity_mount_paths = self.make_singularity_mount_paths(
            job_repository_path,
            hf_datasets_cache,
            train_dataset_path,
            validation_dataset_path,
        )

    def dataset_paths(self, node, dataset_name):
        if dataset_name in clusters[node.value]["datasets"]:
            return (
                clusters[node.value]["datasets"][dataset_name]["train"],
                clusters[node.value]["datasets"][dataset_name]["validation"],
            )
        else:
            return None, None

    def get_singularity_image(self, node):
        return f"{clusters[node.value]['common_directory']}/images/{IMAGE_NAME}"

    def get_machine_backend(self) -> MachineBackend:
        node = platform.uname().node
        if node == "asusgpu0":
            return MachineBackend.ENTROPY
        elif "athena" in node:
            return MachineBackend.ATHENA
        elif node == "login01":
            return MachineBackend.IDEAS
        else:
            return MachineBackend.LOCAL

    def get_setup_arguments(self):
        CLUSTER_NAME = self.node

        if CLUSTER_NAME == MachineBackend.ENTROPY:
            subprocess_args = [
                self.slurm_command,
                "--partition=a100",
                f"--gres=gpu:a100:{self.n_gpus}",
                f"--cpus-per-gpu={self.cpus_per_gpu}",
                f"--mem={1000 // self.self.n_gpus}G",
                f"--job-name={self.job_name}",
                f"--time={self.time}",
                self.grid_entrypoint,
                "singularity",
                "run",
                *self.singularity_env_arguments,
                self.singularity_mount_paths,
                "--nv",
                self.singularity_image,
            ]
        elif CLUSTER_NAME == MachineBackend.ATHENA:
            subprocess_args = [
                self.slurm_command,
                f"--gres=gpu:{self.n_gpus}",
                "--partition=plgrid-gpu-a100",
                f"--cpus-per-gpu={self.cpus_per_gpu}",
                "--account=plgplggllmeffi-gpu-a100",
                f"--job-name={self.job_name}",
                f"--time={self.time}",
                self.grid_entrypoint,
                "singularity",
                "run",
                "--bind=/net:/net",
                *self.singularity_env_arguments,
                self.singularity_mount_paths,
                "--nv",
                self.singularity_image,
            ]
        elif CLUSTER_NAME == MachineBackend.IDEAS:
            subprocess_args = [
                self.slurm_command,
                f"--gres=gpu:ampere:{self.n_gpus}",
                f"--cpus-per-gpu={self.cpus_per_gpu}",
                f"--job-name={self.job_name}",
                f"--time={self.time}",
                "--mem=32G",
                self.nodelist,
                self.grid_entrypoint,
                "singularity",
                "run",
                *self.singularity_env_arguments,
                self.singularity_mount_paths,
                "--nv",
                self.singularity_image,
            ]
        else:
            raise ValueError(f"Unknown cluster: {CLUSTER_NAME}")
        return subprocess_args

    def make_singularity_mount_paths(
        self,
        job_repository_path: str,
        hf_datasets_cache_path: str,
        train_dataset_path: str,
        validation_dataset_path: str,
    ) -> str:
        singularity_mount_paths = f"-B={job_repository_path}:/{REPOSITORY_NAME}"
        is_hf_datasets_cache_needed = (
            train_dataset_path is None or validation_dataset_path is None
        )
        singularity_mount_paths += (
            f",{hf_datasets_cache_path}:{hf_datasets_cache_path}"
            if is_hf_datasets_cache_needed
            else ""
        )
        singularity_mount_paths += (
            f",{train_dataset_path}:{train_dataset_path}"
            if train_dataset_path is not None
            else ""
        )
        singularity_mount_paths += (
            f",{validation_dataset_path}:{validation_dataset_path}"
            if validation_dataset_path is not None
            else ""
        )
        return singularity_mount_paths

    def make_singularity_env_arguments(
        self,
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
