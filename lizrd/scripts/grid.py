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
    runner = get_machine_backend()
    PROCESS_CALL_FUNCTION = lambda args, env: subprocess.run(
        [str(arg) for arg in args if arg is not None], env=env
    )
    try:
        path = sys.argv[1]
    except IndexError:
        raise ValueError("No config path specified. Aborting...")

    if path.endswith(".yaml"):
        configs, all_config_paths = load_with_inheritance(sys.argv[1])
    else:
        raise ValueError("config path point to a .yaml")

    for config in configs:
        config["params"]["path_to_entry_config"] = sys.argv[1]
        config["params"]["all_config_paths"] = all_config_paths

    grid = []
    total_no_experiments = 0
    total_minutes = 0

    for i, grid_args in enumerate(configs):
        print(f"\nProcessing config {i}...")
        pprint.pprint(grid_args)
        RUNNER = grid_args["runner"]
        PARAMS = grid_args["params"]
        TIME = grid_args.get("time", "1-00:00:00")
        GRES = grid_args.get("gres", "gpu:titanv:1")
        DRY_RUN = grid_args.get("dry_run", False)
        SINGULARITY_IMAGE = grid_args.get(
            "singularity_image", get_sparsity_image(runner)
        )
        HF_DATASETS_CACHE = grid_args.get("hf_datasets_cache", get_cache_path(runner))
        RUNS_MULTIPLIER = grid_args.get("runs_multiplier", 1)
        INTERACTIVE_DEBUG = grid_args.get("interactive_debug", False)
        NODELIST = grid_args.get("nodelist", None)
        N_GPUS = grid_args.get("n_gpus", 1)
        CPUS_PER_GPU = grid_args.get("cpus_per_gpu", 8)
        CUDA_VISIBLE_DEVICES = grid_args.get("cuda_visible", None)

        if SINGULARITY_IMAGE is None and runner != MachineBackend.LOCAL:
            raise ValueError(
                "Singularity image is not specified (in JSON or env variable)"
            )

        if NODELIST is not None:
            NODELIST = "--nodelist=" + NODELIST

        PARAMS["temp_args"] = dict()
        for name, param in zip(
            [
                "gres",
                "time",
                "n_gpus",
                "runner",
                "cpus_per_gpu",
                "nodelist",
                "cuda_visible",
            ],
            [GRES, TIME, N_GPUS, RUNNER, CPUS_PER_GPU, NODELIST, CUDA_VISIBLE_DEVICES],
        ):
            PARAMS["temp_args"][name] = param

        partial_grid = create_grid(PARAMS)
        partial_grid = multiply_grid(partial_grid, RUNS_MULTIPLIER)
        grid.extend(partial_grid)

        no_experiments = len(partial_grid)
        total_no_experiments += no_experiments

        minutes_per_exp = timestr_to_minutes(TIME)
        total_minutes_from_this_grid = no_experiments * minutes_per_exp
        total_minutes += total_minutes_from_this_grid

    if len(grid) > 1 and runner == MachineBackend.LOCAL and not DRY_RUN:
        raise ValueError(
            f"Running more than one experiment locally is not supported (you are trying to run {len(grid)} experiments). Aborting..."
        )

    if not runner == MachineBackend.LOCAL:
        if not INTERACTIVE_DEBUG:
            user_input = input(
                f"Will run {total_no_experiments} experiments, using up {total_minutes} minutes, i.e. around {round(total_minutes / 60)} hours\n"
                f"Continue? [Y/n]"
            )
        else:
            user_input = input(f"Will run an INTERACTIVE experiment. \nContinue? [Y/n]")
        if user_input.lower() not in ("", "y", "Y"):
            print("Aborting...")
            exit(1)

    slurm_command = "srun" if INTERACTIVE_DEBUG else "sbatch"
    assert all(
        [
            grid_args.get("INTERACTIVE_DEBUG", False) == INTERACTIVE_DEBUG
            for grid_args in configs
        ]
    ), "`interactive_debug` must be the same for all configs"

    if not (INTERACTIVE_DEBUG or runner == MachineBackend.LOCAL):
        exp_name = next(iter(grid))["name"]
        name_for_branch = (
            f"{exp_name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )
        COPIED_CODE_PATH = copy_and_version_code(
            name_for_branch, name_for_branch, False
        )
    else:
        print(
            f"Running in debug mode or locally, skip copying code to a new directory."
        )

    for i, param_set in enumerate(grid):
        name = param_set["name"]
        param_set["n_gpus"] = param_set["temp_args"]["n_gpus"]
        env = None

        runner_params = []
        for k_packed, v_packed in param_set.items():
            if k_packed == "temp_args":
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

        if runner == MachineBackend.ENTROPY:
            subprocess_args = [
                slurm_command,
                "--partition=common",
                "--qos=16gpu7d",
                f"--gres={param_set['temp_args']['gres']}",
                f"--job-name={name}",
                f"--time={param_set['temp_args']['time']}",
                get_grid_entrypoint(runner),
                "python3",
                "-m",
                param_set["temp_args"]["runner"],
                *runner_params,
            ]
        elif runner == MachineBackend.ATHENA:
            subprocess_args = [
                slurm_command,
                f"--gres=gpu:{param_set['temp_args']['n_gpus']}",
                "--partition=plgrid-gpu-a100",
                f"--cpus-per-gpu={param_set['temp_args']['cpus_per_gpu']}",
                "--account=plgplggllmeffi-gpu-a100",
                f"--job-name={name}",
                f"--time={param_set['temp_args']['time']}",
                get_grid_entrypoint(runner),
                "singularity",
                "run",
                "--bind=/net:/net",
                f"--env",
                f"HF_DATASETS_CACHE={HF_DATASETS_CACHE}",
                f"-B={COPIED_CODE_PATH}:/sparsity,{HF_DATASETS_CACHE}:{HF_DATASETS_CACHE}",
                "--nv",
                SINGULARITY_IMAGE,
                "python3",
                "-m",
                param_set["temp_args"]["runner"],
                *runner_params,
            ]
        elif runner == MachineBackend.IDEAS:
            subprocess_args = [
                slurm_command,
                f"--gres=gpu:{param_set['temp_args']['n_gpus']}",
                f"--cpus-per-gpu={param_set['temp_args']['cpus_per_gpu']}",
                f"--job-name={name}",
                f"--time={param_set['temp_args']['time']}",
                "--mem=32G",
                param_set["temp_args"]["nodelist"],
                get_grid_entrypoint(runner),
                "singularity",
                "run",
                f"--env",
                f"HF_DATASETS_CACHE={HF_DATASETS_CACHE}",
                f"-B={COPIED_CODE_PATH}:/sparsity,{HF_DATASETS_CACHE}:{HF_DATASETS_CACHE}",
                "--nv",
                SINGULARITY_IMAGE,
                "python3",
                "-m",
                param_set["temp_args"]["runner"],
                *runner_params,
            ]
        elif runner == MachineBackend.ENTROPY_GPU:
            if param_set["temp_args"]["cuda_visible"] is not None:
                env = os.environ.copy()
                env.update(
                    {"CUDA_VISIBLE_DEVICES": param_set["temp_args"]["cuda_visible"]}
                )
            subprocess_args = [
                "singularity",
                "run",
                f"--env",
                f"HF_DATASETS_CACHE={HF_DATASETS_CACHE}",
                f"-B={COPIED_CODE_PATH}:/sparsity,{HF_DATASETS_CACHE}:{HF_DATASETS_CACHE}",
                "--nv",
                SINGULARITY_IMAGE,
                "python3",
                "-m",
                param_set["temp_args"]["runner"],
                *runner_params,
            ]
        elif runner == MachineBackend.LOCAL:
            # We run the experiment directly, not through a grid entrypoint script
            # because we want to be able to debug it
            runner_main_function = get_train_main_function(
                param_set["temp_args"]["runner"]
            )
            runner_main_function(None, runner_params=runner_params)
            exit(0)
        else:
            raise ValueError(f"Unknown runner: {runner}")

        if DRY_RUN:
            print(" ".join(subprocess_args))
        else:
            print(f"running experiment {i} from {name}...")
            PROCESS_CALL_FUNCTION(subprocess_args, env)
            sleep(5)
        if INTERACTIVE_DEBUG:
            break
