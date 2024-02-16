import os
import unittest
from lizrd.hostname_setup.hostname_setup import get_subprocess_args
from lizrd.hostname_setup.utils import MachineBackend


class TestClusterConfigs(unittest.TestCase):
    def test_ideas_cluster_config(self):
        cluster = MachineBackend.IDEAS
        command = "srun"
        setup_args = {
            "n_gpus": 1,
            "cpus_per_gpu": 10,
            "time": "1:00:00",
            "singularity_image": "sparsity_2024.02.06_16.14.02.sif",
            "nodelist": "--nodelist=nodename",
            "hf_datasets_cache": "/somepath/.cache",
            "runner": "research.conditional.train.random_train",
        }
        training_args = {
            "name": "super_random_name",
            "all_config_paths": "full_config0.yaml",
            "dataset_type": "c4",
            "runner": "lizrd.scripts.run_train",
            "train_dataset_path": "/some_path/c4_train",
            "validation_dataset_path": "/some_path/c4_validation",
        }
        neptune_key = "R@nd0mN3ptun3K3y"
        wandb_key = "R@nd0mW@ndbK3y"

        returned_args = get_subprocess_args(
            cluster, command, setup_args, training_args, neptune_key, wandb_key
        )

        expected_args = [
            "srun",
            "--gres=gpu:ampere:1",
            "--cpus-per-gpu=10",
            "--job-name=super_random_name",
            "--time=1:00:00",
            "--mem=32G",
            "--nodelist=nodename",
            "lizrd/scripts/grid_entrypoint.sh",
            "singularity",
            "run",
            "--env",
            "HF_DATASETS_CACHE=/somepath/.cache,NEPTUNE_API_TOKEN=R@nd0mN3ptun3K3y,WANDB_API_KEY=R@nd0mW@ndbK3y",
            f"-B={os.getcwd()}:/llm-random,/some_path/c4_train:/some_path/c4_train,/some_path/c4_validation:/some_path/c4_validation",
            "--nv",
            "sparsity_2024.02.06_16.14.02.sif",
            "python3",
            "-m",
            "research.conditional.train.random_train",
            "--name",
            "super_random_name",
            "--all_config_paths",
            "full_config0.yaml",
            "--dataset_type",
            "c4",
            "--runner",
            "lizrd.scripts.run_train",
            "--train_dataset_path",
            "/some_path/c4_train",
            "--validation_dataset_path",
            "/some_path/c4_validation",
            "--n_gpus",
            "1",
        ]

        self.assertEqual(len(returned_args), len(expected_args))
        for a, b in zip(returned_args, expected_args):
            self.assertEqual(a, b)

    def test_entropy_cluster_config(self):
        cluster = MachineBackend.ENTROPY
        command = "sbatch"
        setup_args = {
            "n_gpus": 1,
            "cpus_per_gpu": 10,
            "time": "1:00:00",
            "singularity_image": "sparsity_2024.02.06_16.14.02.sif",
            "hf_datasets_cache": "/somepath/.cache",
            "runner": "research.conditional.train.random_train",
        }
        training_args = {
            "name": "super_random_name",
            "all_config_paths": "full_config0.yaml",
            "dataset_type": "c4",
            "runner": "lizrd.scripts.run_train",
            "train_dataset_path": "/some_path/c4_train",
            "validation_dataset_path": "/some_path/c4_validation",
        }
        neptune_key = "R@nd0mN3ptun3K3y"
        wandb_key = "R@nd0mW@ndbK3y"

        returned_args = get_subprocess_args(
            cluster, command, setup_args, training_args, neptune_key, wandb_key
        )

        expected_args = [
            "sbatch",
            "--partition=a100",
            "--gres=gpu:a100:1",
            "--cpus-per-gpu=10",
            "--mem=1000G",
            "--job-name=super_random_name",
            "--time=1:00:00",
            "lizrd/scripts/grid_entrypoint.sh",
            "singularity",
            "run",
            "--env",
            "HF_DATASETS_CACHE=/somepath/.cache,NEPTUNE_API_TOKEN=R@nd0mN3ptun3K3y,WANDB_API_KEY=R@nd0mW@ndbK3y",
            f"-B={os.getcwd()}:/llm-random,/some_path/c4_train:/some_path/c4_train,/some_path/c4_validation:/some_path/c4_validation",
            "--nv",
            "sparsity_2024.02.06_16.14.02.sif",
            "python3",
            "-m",
            "research.conditional.train.random_train",
            "--name",
            "super_random_name",
            "--all_config_paths",
            "full_config0.yaml",
            "--dataset_type",
            "c4",
            "--runner",
            "lizrd.scripts.run_train",
            "--train_dataset_path",
            "/some_path/c4_train",
            "--validation_dataset_path",
            "/some_path/c4_validation",
            "--n_gpus",
            "1",
        ]

        self.assertEqual(len(returned_args), len(expected_args))
        for a, b in zip(returned_args, expected_args):
            self.assertEqual(a, b)

    def test_athena_cluster_config(self):
        cluster = MachineBackend.ATHENA
        command = "sbatch"
        setup_args = {
            "n_gpus": 1,
            "cpus_per_gpu": 10,
            "time": "1:00:00",
            "singularity_image": "sparsity_2024.02.06_16.14.02.sif",
            "hf_datasets_cache": "/somepath/.cache",
            "runner": "research.conditional.train.random_train",
        }
        training_args = {
            "name": "super_random_name",
            "all_config_paths": "full_config0.yaml",
            "dataset_type": "c4",
            "runner": "lizrd.scripts.run_train",
            "train_dataset_path": "/some_path/c4_train",
            "validation_dataset_path": "/some_path/c4_validation",
        }
        neptune_key = "R@nd0mN3ptun3K3y"
        wandb_key = "R@nd0mW@ndbK3y"

        returned_args = get_subprocess_args(
            cluster, command, setup_args, training_args, neptune_key, wandb_key
        )

        expected_args = [
            "sbatch",
            "--gres=gpu:1",
            "--partition=plgrid-gpu-a100",
            "--cpus-per-gpu=10",
            "--account=plgplggllmeffi-gpu-a100",
            "--job-name=super_random_name",
            "--time=1:00:00",
            "lizrd/scripts/grid_entrypoint.sh",
            "singularity",
            "run",
            "--bind=/net:/net",
            "--env",
            "HF_DATASETS_CACHE=/somepath/.cache,NEPTUNE_API_TOKEN=R@nd0mN3ptun3K3y,WANDB_API_KEY=R@nd0mW@ndbK3y",
            f"-B={os.getcwd()}:/llm-random,/some_path/c4_train:/some_path/c4_train,/some_path/c4_validation:/some_path/c4_validation",
            "--nv",
            "sparsity_2024.02.06_16.14.02.sif",
            "python3",
            "-m",
            "research.conditional.train.random_train",
            "--name",
            "super_random_name",
            "--all_config_paths",
            "full_config0.yaml",
            "--dataset_type",
            "c4",
            "--runner",
            "lizrd.scripts.run_train",
            "--train_dataset_path",
            "/some_path/c4_train",
            "--validation_dataset_path",
            "/some_path/c4_validation",
            "--n_gpus",
            "1",
        ]

        self.assertEqual(len(returned_args), len(expected_args))
        for a, b in zip(returned_args, expected_args):
            self.assertEqual(a, b)
