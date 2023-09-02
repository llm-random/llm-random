"""
Script to grid search in recycle layers. Run this script from the root of the project:
$ python3 research/reinitialization/scripts/grid.py
Remember to set RUNNER and PARAMS in the script or add an argument parser.
"""

import datetime
import os
import pprint
import subprocess
import sys
from time import sleep

from lizrd.scripts.grid_utils import (
    create_grid,
    get_cache_path,
    get_sparsity_image,
    get_train_main_function,
    multiply_grid,
    timestr_to_minutes,
    get_machine_backend,
    MachineBackend,
    get_grid_entrypoint,
    unpack_params,
)
from lizrd.support.code_versioning_support import copy_and_version_code
from lizrd.support.misc import load_with_inheritance

if __name__ == "__main__":
    CLUSTER_NAME = get_machine_backend()
    PROCESS_CALL_FUNCTION = lambda args, env: subprocess.run(
        [str(arg) for arg in args if arg is not None], env=env
    )

    try:
        path = sys.argv[1]
    except IndexError:
        raise ValueError("No config path specified. Aborting...")

    if path.endswith(".yaml"):
        configs, all_config_paths = load_with_inheritance(path)
    else:
        raise ValueError("config path point to a .yaml")

    for config in configs:
        config["params"]["path_to_entry_config"] = sys.argv[1]
        config["params"]["all_config_paths"] = ",".join(all_config_paths)

    grid = []
    total_no_experiments = 0
    total_minutes = 0

    for i, grid_args in enumerate(configs):
        print(f"\nProcessing config {i}...")
        pprint.pprint(grid_args)

        RUNS_MULTIPLIER = grid_args.get("runs_multiplier", 1)  ######
        TIME = grid_args.get("time", "1-00:00:00")  ######

        RUNNER = grid_args["runner"]
        PARAMS = grid_args["params"]
        GRES = grid_args.get("gres", "gpu:1")
        DRY_RUN = grid_args.get("dry_run", False)
        SINGULARITY_IMAGE = grid_args.get(
            "singularity_image", get_sparsity_image(CLUSTER_NAME)
        )
        HF_DATASETS_CACHE = grid_args.get(
            "hf_datasets_cache", get_cache_path(CLUSTER_NAME)
        )
        NODELIST = grid_args.get("nodelist", None)
        N_GPUS = grid_args.get("n_gpus", 1)
        CPUS_PER_GPU = grid_args.get("cpus_per_gpu", 8)
        CUDA_VISIBLE_DEVICES = grid_args.get("cuda_visible", None)

        PARAMS["setup_args"] = dict()
        for name, param in zip(
            [
                "gres",
                "time",
                "n_gpus",
                "runner",
                "cpus_per_gpu",
                "nodelist",
                "cuda_visible",
                "hf_datasets_cache",
                "singularity_image",
            ],
            [
                GRES,
                TIME,
                N_GPUS,
                RUNNER,
                CPUS_PER_GPU,
                NODELIST,
                CUDA_VISIBLE_DEVICES,
                HF_DATASETS_CACHE,
                SINGULARITY_IMAGE,
            ],
        ):
            PARAMS["setup_args"][name] = param

        single_exp_grid = create_grid(PARAMS)
        single_exp_grid = multiply_grid(single_exp_grid, RUNS_MULTIPLIER)
        grid.extend(single_exp_grid)

        no_experiments = len(single_exp_grid)
        total_no_experiments += no_experiments

        minutes_per_exp = timestr_to_minutes(TIME)
        total_minutes_from_this_grid = no_experiments * minutes_per_exp
        total_minutes += total_minutes_from_this_grid

    if CLUSTER_NAME == MachineBackend.LOCAL and len(grid) > 1:
        raise ValueError(
            f"Running more than one experiment locally is not supported (you are trying to run {len(grid)} experiments). Aborting..."
        )

    interactive_options_per_config = [
        grid_args.get("interactive_debug", False) for grid_args in configs
    ]

    assert (
        len(set(interactive_options_per_config)) == 1
    ), f"`interactive_debug` must be the same for all configs"

    interactive_debug_session = interactive_options_per_config[0]

    if not CLUSTER_NAME == MachineBackend.LOCAL:
        if not interactive_debug_session:
            user_input = input(
                f"Will run {total_no_experiments} experiments, using up {total_minutes} minutes, i.e. around {round(total_minutes / 60)} hours\n"
                f"Continue? [Y/n]"
            )
        else:
            user_input = input(
                f"Will run an INTERACTIVE experiment, which will be the first one from the supplied configs. \nContinue? [Y/n]"
            )
        if user_input.lower() not in ("", "y", "Y"):
            print("Aborting...")
            exit(1)

    if not (interactive_debug_session or CLUSTER_NAME == MachineBackend.LOCAL):
        exp_name = next(iter(grid))["name"]
        name_for_branch = (
            f"{exp_name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )
        copy_and_version_code(name_for_branch, name_for_branch, False)
    else:
        print(
            f"Running in debug mode or locally, skip copying code to a new directory."
        )

    slurm_command = "srun" if interactive_debug_session else "sbatch"

    for i, param_set in enumerate(grid):
        name = param_set["name"]
        param_set["n_gpus"] = param_set["setup_args"]["n_gpus"]
        if param_set["setup_args"].get("nodelist", None) is not None:
            param_set["setup_args"]["nodelist"] = (
                "--nodelist=" + param_set["setup_args"]["nodelist"]
            )

        env = None
        runner_params = []

        for k_packed, v_packed in param_set.items():
            if k_packed == "setup_args":
                continue
            for k, v in zip(*unpack_params(k_packed, v_packed)):
                if isinstance(v, bool):
                    if v:
                        runner_params.append(f"--{k}")
                    else:
                        pass  # simply don't add it if v == False
                    continue
                else:
                    runner_params.append(f"--{k}")
                    if isinstance(v, list):
                        runner_params.extend([str(s) for s in v])
                    else:
                        runner_params.append(str(v))

        if CLUSTER_NAME == MachineBackend.ENTROPY:
            subprocess_args = [
                slurm_command,
                "--partition=common",
                "--qos=16gpu7d",
                f"--gres={param_set['setup_args']['gres']}",
                f"--job-name={name}",
                f"--time={param_set['setup_args']['time']}",
                get_grid_entrypoint(CLUSTER_NAME),
                "python3",
                "-m",
                param_set["setup_args"]["runner"],
                *runner_params,
            ]
        elif CLUSTER_NAME == MachineBackend.ATHENA:
            subprocess_args = [
                slurm_command,
                f"--gres=gpu:{param_set['setup_args']['n_gpus']}",
                "--partition=plgrid-gpu-a100",
                f"--cpus-per-gpu={param_set['setup_args']['cpus_per_gpu']}",
                "--account=plgplggllmeffi-gpu-a100",
                f"--job-name={name}",
                f"--time={param_set['setup_args']['time']}",
                get_grid_entrypoint(CLUSTER_NAME),
                "singularity",
                "run",
                "--bind=/net:/net",
                f"--env",
                f"HF_DATASETS_CACHE={param_set['setup_args']['hf_datasets_cache']}",
                f"-B={os.getcwd()}:/sparsity,{param_set['setup_args']['hf_datasets_cache']}:{param_set['setup_args']['hf_datasets_cache']}",
                "--nv",
                param_set["setup_args"]["singularity_image"],
                "python3",
                "-m",
                param_set["setup_args"]["runner"],
                *runner_params,
            ]
        elif CLUSTER_NAME == MachineBackend.IDEAS:
            subprocess_args = [
                slurm_command,
                f"--gres=gpu:{param_set['setup_args']['n_gpus']}",
                f"--cpus-per-gpu={param_set['setup_args']['cpus_per_gpu']}",
                f"--job-name={name}",
                f"--time={param_set['setup_args']['time']}",
                "--mem=32G",
                param_set["setup_args"]["nodelist"],
                get_grid_entrypoint(CLUSTER_NAME),
                "singularity",
                "run",
                f"--env",
                f"HF_DATASETS_CACHE={param_set['setup_args']['hf_datasets_cache']}",
                f"-B={os.getcwd()}:/sparsity,{param_set['setup_args']['hf_datasets_cache']}:{param_set['setup_args']['hf_datasets_cache']}",
                "--nv",
                param_set["setup_args"]["singularity_image"],
                "python3",
                "-m",
                param_set["setup_args"]["runner"],
                *runner_params,
            ]
        elif CLUSTER_NAME == MachineBackend.ENTROPY_GPU:
            if param_set["setup_args"]["cuda_visible"] is not None:
                env = os.environ.copy()
                env.update(
                    {"CUDA_VISIBLE_DEVICES": param_set["setup_args"]["cuda_visible"]}
                )
            subprocess_args = [
                "singularity",
                "run",
                f"--env",
                f"HF_DATASETS_CACHE={param_set['setup_args']['hf_datasets_cache']}",
                f"-B={os.getcwd()}:/sparsity,{param_set['setup_args']['hf_datasets_cache']}:{param_set['setup_args']['hf_datasets_cache']}",
                "--nv",
                param_set["setup_args"]["singularity_image"],
                "python3",
                "-m",
                param_set["setup_args"]["runner"],
                *runner_params,
            ]
        elif CLUSTER_NAME == MachineBackend.LOCAL:
            # We run the experiment directly, not through a grid entrypoint script
            # because we want to be able to debug it
            runner_main_function = get_train_main_function(
                param_set["setup_args"]["runner"]
            )
            runner_main_function(None, runner_params=runner_params)
            exit(0)
        else:
            raise ValueError(f"Unknown cluster name: {CLUSTER_NAME}")
        print(f"running experiment {i} from {name}...")
        PROCESS_CALL_FUNCTION(subprocess_args, env)
        sleep(5)
        if interactive_debug_session:
            print("Ran only the first experiment in interactive mode. Aborting...")
            break
