import unittest
from unittest.mock import patch

from lizrd.grid.grid import create_subprocess_args
from lizrd.grid.infrastructure import (
    IdeasBackend,
    EntropyBackend,
    AthenaBackend,
)

hf_dataset_cache = "/some/path/for/hf/dataset/cache"
experiment_path = "/some/path/for/experiment"
# repo_path = "/some/path/for/repo" # TODO change when llm-random is not hardcoded value
image_path = "/some/path/for/image"
neptune_api_key = "r@nd0mN3ptun3Ap1K3y"
wandb_api_key = "r@nd0mW@ndbAp1K3y"


def unify_arguments(arguments):
    ix = arguments.index("python3")
    setup_args = arguments[:ix]
    training_args = arguments[ix + 3 :]
    sorted_training_args = sort_training_args(training_args)
    return setup_args + arguments[ix : ix + 3] + sorted_training_args


def sort_training_args(values):
    pairs = []
    current_arg = None
    current_values = []

    for value in values:
        if value.startswith("--"):
            if current_arg is not None:
                pairs.append((current_arg, current_values))
            current_arg = value
            current_values = []
        else:
            current_values.append(value)

    if current_arg is not None:
        pairs.append((current_arg, current_values))

    pairs.sort(key=lambda x: x[0])

    result = []
    for arg, values in pairs:
        result.append(arg)
        values.sort()
        result.extend(values)
    return result


class IdeasTestBackend(IdeasBackend):
    def get_singularity_image(self) -> str:
        return image_path

    def get_cache_path(self) -> str:
        return hf_dataset_cache


class AthenaTestBackend(AthenaBackend):
    def get_singularity_image(self) -> str:
        return image_path

    def get_cache_path(self) -> str:
        return hf_dataset_cache


class EntropyTestBackend(EntropyBackend):
    def get_singularity_image(self) -> str:
        return image_path

    def get_cache_path(self) -> str:
        return hf_dataset_cache


class TestGrid(unittest.TestCase):
    def assertUnifiedEqual(self, a, b):
        unified_a = [unify_arguments(config) for config in a]
        unified_b = [unify_arguments(config) for config in b]

        self.assertEqual(unified_a, unified_b)

    @patch("os.getcwd")
    def test_baseline_generated_args(self, os_getcwd):
        os_getcwd.return_value = experiment_path
        CLUSTER = IdeasTestBackend()
        expected_output = [
            [
                "sbatch",
                "--gres=gpu:ampere:0",
                "--cpus-per-gpu=8",
                "--job-name=baseline_test",
                "--time=5-05:00:00",
                "--mem=125G",
                None,
                "lizrd/grid/grid_entrypoint.sh",
                "singularity",
                "run",
                "--env",
                f"HF_DATASETS_CACHE={hf_dataset_cache},NEPTUNE_API_TOKEN={neptune_api_key},WANDB_API_KEY={wandb_api_key}",
                f"-B={experiment_path}:/llm-random,{hf_dataset_cache}:{hf_dataset_cache}",
                "--nv",
                f"{image_path}",
                "python3",
                "-m",
                "research.conditional.train.cc_train",
                "--batch_size",
                "4",
                "--cutoff",
                "16",
                "--project_name",
                "pmtest/llm-random-tests",
                "--name",
                "baseline_test",
                "--mixed_precision",
                "--mixed_precision_dtype",
                "float16",
                "--tags",
                "test",
                "model_type=bert",
                "ff_mode=vanilla",
                "--logger_types",
                "neptune",
                "--n_steps",
                "10",
                "--dmodel",
                "64",
                "--dff",
                "256",
                "--n_blocks",
                "2",
                "--logging_interval_heavy",
                "2",
                "--logging_interval_loss",
                "1",
                "--grad_clip",
                "0.5",
                "--scheduler",
                "constant",
                "--lr_warmup_steps",
                "4",
                "--n_att_heads",
                "4",
                "--learning_rate",
                "0.0001",
                "--dataset_type",
                "wikibook",
                "--use_dummy_dataset",
                "--init_type",
                "kaiming_uniform",
                "--init_scale",
                "1.0",
                "--git_branch",
                "cool_git_branch",
                "--path_to_entry_config",
                "configs/test/test_baseline.yaml",
                "--all_config_paths",
                "configs/test/test_baseline.yaml,full_config0.yaml",
                "--model_type",
                "bert",
                "--ff_mode",
                "vanilla",
                "--n_gpus",
                "0",
            ]
        ]
        experiments, _ = create_subprocess_args(
            "configs/test/test_baseline.yaml",
            "cool_git_branch",
            f"{neptune_api_key}",
            f"{wandb_api_key}",
            CLUSTER,
            skip_confirmation=True,
            skip_copy_code=True,
        )
        returned_output = [experiment[0] for experiment in experiments]
        self.assertUnifiedEqual(returned_output, expected_output)

    @patch("os.getcwd")
    def test_compare_bmm_generated_args(self, os_getcwd):
        os_getcwd.return_value = experiment_path
        CLUSTER = EntropyTestBackend()
        expected_output = [
            [
                "sbatch",
                "--partition=a100",
                "--gres=gpu:a100:1",
                "--cpus-per-gpu=8",
                "--mem=125G",
                "--job-name=granular_4_mini",
                "--time=0-05:00:00",
                "lizrd/grid/grid_entrypoint.sh",
                "singularity",
                "run",
                "--env",
                f"HF_DATASETS_CACHE={hf_dataset_cache},NEPTUNE_API_TOKEN={neptune_api_key},WANDB_API_KEY={wandb_api_key}",
                f"-B={experiment_path}:/llm-random,/local_storage_2/llm-random/datasets/c4_train:/local_storage_2/llm-random/datasets/c4_train,/local_storage_2/llm-random/datasets/c4_validation:/local_storage_2/llm-random/datasets/c4_validation",
                "--nv",
                f"{image_path}",
                "python3",
                "-m",
                "research.conditional.train.cc_train",
                "--name",
                "granular_4_mini",
                "--granularity",
                "4",
                "--dmodel",
                "256",
                "--n_blocks",
                "4",
                "--n_att_heads",
                "4",
                "--ff_mode",
                "expert_choice",
                "--softmax_over",
                "experts",
                "--group_granular_moe_by_batch",
                "--granular_moe_one_hot_impl",
                "--layer_norm_in_expert_choice",
                "--mixed_precision_dtype",
                "bfloat16",
                "--expansion_rate",
                "32",
                "--effective_dff_x",
                "4",
                "--learning_rate",
                "1e-4",
                "--init_type",
                "truncated_normal",
                "--init_scale",
                "0.1",
                "--model_type",
                "gpt",
                "--dataset_type",
                "c4",
                "--mixed_precision",
                "--flash_attention",
                "--logger_types",
                "neptune",
                "--project_name",
                "pmtest/llm-random",
                "--logging_interval_heavy",
                "5000",
                "--logging_interval_loss",
                "1000",
                "--save_weights_path",
                "model_ckpt",
                "--save_weights_interval",
                "25000",
                "--cutoff",
                "256",
                "--batch_size",
                "256",
                "--n_steps",
                "275000",
                "--final_lr_step",
                "250000",
                "--scheduler",
                "cosine",
                "--lr_warmup_steps",
                "2500",
                "--final_lr_fraction",
                "0.1",
                "--grad_clip",
                "0.5",
                "--git_branch",
                "cool_git_branch",
                "--path_to_entry_config",
                "configs/experiments/expert_choice/compare_bmm_einsum.yaml",
                "--all_config_paths",
                "configs/baselines/common.yaml,configs/baselines/gpt/dense/common.yaml,configs/baselines/gpt/expert_choice/common.yaml,configs/baselines/gpt/expert_choice/granularity/4/mini.yaml,configs/baselines/gpt/expert_choice/mini.yaml,configs/experiments/expert_choice/compare_bmm_einsum.yaml,full_config0.yaml",
                "--tags",
                "use_torch_bmm=F",
                "--n_gpus",
                "1",
                "--train_dataset_path",
                "/local_storage_2/llm-random/datasets/c4_train",
                "--validation_dataset_path",
                "/local_storage_2/llm-random/datasets/c4_validation",
            ],
            [
                "sbatch",
                "--partition=a100",
                "--gres=gpu:a100:1",
                "--cpus-per-gpu=8",
                "--mem=125G",
                "--job-name=granular_4_mini",
                "--time=0-05:00:00",
                "lizrd/grid/grid_entrypoint.sh",
                "singularity",
                "run",
                "--env",
                f"HF_DATASETS_CACHE={hf_dataset_cache},NEPTUNE_API_TOKEN={neptune_api_key},WANDB_API_KEY={wandb_api_key}",
                f"-B={experiment_path}:/llm-random,/local_storage_2/llm-random/datasets/c4_train:/local_storage_2/llm-random/datasets/c4_train,/local_storage_2/llm-random/datasets/c4_validation:/local_storage_2/llm-random/datasets/c4_validation",
                "--nv",
                f"{image_path}",
                "python3",
                "-m",
                "research.conditional.train.cc_train",
                "--name",
                "granular_4_mini",
                "--granularity",
                "4",
                "--dmodel",
                "256",
                "--n_blocks",
                "4",
                "--n_att_heads",
                "4",
                "--ff_mode",
                "expert_choice",
                "--softmax_over",
                "experts",
                "--group_granular_moe_by_batch",
                "--use_torch_bmm",
                "--granular_moe_one_hot_impl",
                "--layer_norm_in_expert_choice",
                "--mixed_precision_dtype",
                "bfloat16",
                "--expansion_rate",
                "32",
                "--effective_dff_x",
                "4",
                "--learning_rate",
                "1e-4",
                "--init_type",
                "truncated_normal",
                "--init_scale",
                "0.1",
                "--model_type",
                "gpt",
                "--dataset_type",
                "c4",
                "--mixed_precision",
                "--flash_attention",
                "--logger_types",
                "neptune",
                "--project_name",
                "pmtest/llm-random",
                "--logging_interval_heavy",
                "5000",
                "--logging_interval_loss",
                "1000",
                "--save_weights_path",
                "model_ckpt",
                "--save_weights_interval",
                "25000",
                "--cutoff",
                "256",
                "--batch_size",
                "256",
                "--n_steps",
                "275000",
                "--final_lr_step",
                "250000",
                "--scheduler",
                "cosine",
                "--lr_warmup_steps",
                "2500",
                "--final_lr_fraction",
                "0.1",
                "--grad_clip",
                "0.5",
                "--git_branch",
                "cool_git_branch",
                "--path_to_entry_config",
                "configs/experiments/expert_choice/compare_bmm_einsum.yaml",
                "--all_config_paths",
                "configs/baselines/common.yaml,configs/baselines/gpt/dense/common.yaml,configs/baselines/gpt/expert_choice/common.yaml,configs/baselines/gpt/expert_choice/granularity/4/mini.yaml,configs/baselines/gpt/expert_choice/mini.yaml,configs/experiments/expert_choice/compare_bmm_einsum.yaml,full_config1.yaml",
                "--tags",
                "use_torch_bmm=T",
                "--n_gpus",
                "1",
                "--train_dataset_path",
                "/local_storage_2/llm-random/datasets/c4_train",
                "--validation_dataset_path",
                "/local_storage_2/llm-random/datasets/c4_validation",
            ],
        ]
        experiments, _ = create_subprocess_args(
            "configs/experiments/expert_choice/compare_bmm_einsum.yaml",
            "cool_git_branch",
            f"{neptune_api_key}",
            f"{wandb_api_key}",
            CLUSTER,
            skip_confirmation=True,
            skip_copy_code=True,
        )
        returned_output = [experiment[0] for experiment in experiments]
        self.assertUnifiedEqual(returned_output, expected_output)

    @patch("os.getcwd")
    def test_lr_grid(self, os_getcwd):
        os_getcwd.return_value = experiment_path
        CLUSTER = AthenaTestBackend()
        expected_output = [
            [
                "sbatch",
                "--gres=gpu:2",
                "--partition=plgrid-gpu-a100",
                "--mem=250G",
                "--account=plgsubslearnath-gpu-a100",
                "--job-name=lr_grid",
                "--time=40:00:00",
                "lizrd/grid/grid_entrypoint.sh",
                "singularity",
                "run",
                "--bind=/net:/net",
                "--env",
                f"HF_DATASETS_CACHE={hf_dataset_cache},NEPTUNE_API_TOKEN={neptune_api_key},WANDB_API_KEY={wandb_api_key}",
                f"-B={experiment_path}:/llm-random,{hf_dataset_cache}:{hf_dataset_cache}",
                "--nv",
                f"{image_path}",
                "python3",
                "-m",
                "research.conditional.train.cc_train",
                "--n_blocks",
                "16",
                "--model_type",
                "gpt",
                "--dmodel",
                "16",
                "--n_att_heads",
                "4",
                "--n_steps",
                "100",
                "--scheduler",
                "cosine",
                "--init_type",
                "truncated_normal",
                "--init_scale",
                "0.02",
                "--dataset_type",
                "c4",
                "--batch_size",
                "32",
                "--cutoff",
                "256",
                "--name",
                "lr_grid",
                "--git_branch",
                "cool_git_branch",
                "--path_to_entry_config",
                "lizrd/test/test_lr_grid.yaml",
                "--all_config_paths",
                "lizrd/test/test_lr_grid.yaml,full_config0.yaml",
                "--tags",
                "learning_rate=5e-4",
                "--learning_rate",
                "5e-4",
                "--n_gpus",
                "2",
                "--logger_types",
                "stdout",
            ],
            [
                "sbatch",
                "--gres=gpu:2",
                "--partition=plgrid-gpu-a100",
                "--mem=250G",
                "--account=plgsubslearnath-gpu-a100",
                "--job-name=lr_grid",
                "--time=40:00:00",
                "lizrd/grid/grid_entrypoint.sh",
                "singularity",
                "run",
                "--bind=/net:/net",
                "--env",
                f"HF_DATASETS_CACHE={hf_dataset_cache},NEPTUNE_API_TOKEN={neptune_api_key},WANDB_API_KEY={wandb_api_key}",
                f"-B={experiment_path}:/llm-random,{hf_dataset_cache}:{hf_dataset_cache}",
                "--nv",
                f"{image_path}",
                "python3",
                "-m",
                "research.conditional.train.cc_train",
                "--n_blocks",
                "16",
                "--model_type",
                "gpt",
                "--dmodel",
                "16",
                "--n_att_heads",
                "4",
                "--n_steps",
                "100",
                "--scheduler",
                "cosine",
                "--init_type",
                "truncated_normal",
                "--init_scale",
                "0.02",
                "--dataset_type",
                "c4",
                "--batch_size",
                "32",
                "--cutoff",
                "256",
                "--name",
                "lr_grid",
                "--git_branch",
                "cool_git_branch",
                "--path_to_entry_config",
                "lizrd/test/test_lr_grid.yaml",
                "--all_config_paths",
                "lizrd/test/test_lr_grid.yaml,full_config1.yaml",
                "--tags",
                "learning_rate=7e-4",
                "--learning_rate",
                "7e-4",
                "--n_gpus",
                "2",
                "--logger_types",
                "stdout",
            ],
        ]
        experiments, _ = create_subprocess_args(
            "lizrd/test/test_lr_grid.yaml",
            "cool_git_branch",
            f"{neptune_api_key}",
            f"{wandb_api_key}",
            CLUSTER,
            skip_confirmation=True,
            skip_copy_code=True,
        )
        returned_output = [experiment[0] for experiment in experiments]

        self.assertUnifiedEqual(returned_output, expected_output)
