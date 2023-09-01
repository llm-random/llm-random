"""
Script to grid search in recycle layers. Run this script from the root of the project:
$ python3 research/reinitialization/scripts/grid.py
Remember to set RUNNER and PARAMS in the script or add an argument parser.
"""

import datetime
import os
import subprocess
import sys
from time import sleep

import yaml

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

RUNNER = "research.reinitialization.train.reinit_train"


# ^ - grid over that
# * - apply function
PARAMS = {
    "project_name": f"{os.getenv('USER')}/mp",
    "name": "mp",
    "ff_layer": "regular",
    "batch_size": 128,
    "cutoff": 128,
    "^mixed_precision": [True, False],
    "tags": ["test"],
    "use_clearml": True,
    "pruner_n_steps": 100,
}

TIME = "1-00:00:00"
GRES = "gpu:titanv:1"
DRY_RUN = False
CODE_PATH = os.getcwd()
INTERACTIVE_DEBUG = False
RUNS_MULTIPLIER = 1
PUSH_TO_GIT = False
COPIED_CODE_PATH = ""
if __name__ == "__main__":
    runner = get_machine_backend()
    HF_DATASETS_CACHE = get_cache_path(runner)
    SINGULARITY_IMAGE = get_sparsity_image(runner)
    NODELIST = None
    N_GPUS = 1
    CPUS_PER_GPU = 8
    CUDA_VISIBLE_DEVICES = None
    PROCESS_CALL_FUNCTION = lambda args, env: subprocess.run(
        [str(arg) for arg in args if arg is not None], env=env
    )

    try:
        path = sys.argv[1]
    except IndexError:
        raise ValueError("config path must be specified as the first argument")

    if path.endswith(".yaml"):
        grid_args = yaml.safe_load(open(sys.argv[1]))
    else:
        raise ValueError("config path point to a .yaml")
    grid_args["params"]["path_to_config"] = sys.argv[1]

    RUNNER = grid_args.get("runner", RUNNER)
    PARAMS = grid_args.get("params", PARAMS)
    TIME = grid_args.get("time", TIME)
    GRES = grid_args.get("gres", GRES)
    DRY_RUN = grid_args.get("dry_run", DRY_RUN)
    SINGULARITY_IMAGE = grid_args.get("singularity_image", SINGULARITY_IMAGE)
    HF_DATASETS_CACHE = grid_args.get("hf_datasets_cache", HF_DATASETS_CACHE)
    RUNS_MULTIPLIER = grid_args.get("runs_multiplier", RUNS_MULTIPLIER)
    INTERACTIVE_DEBUG = grid_args.get("interactive_debug", INTERACTIVE_DEBUG)
    PUSH_TO_GIT = grid_args.get("push_to_git", PUSH_TO_GIT)
    NODELIST = grid_args.get("nodelist", NODELIST)
    N_GPUS = grid_args.get("n_gpus", N_GPUS)
    CPUS_PER_GPU = grid_args.get("cpus_per_gpu", CPUS_PER_GPU)
    CUDA_VISIBLE_DEVICES = grid_args.get("cuda_visible", CUDA_VISIBLE_DEVICES)

    if SINGULARITY_IMAGE is None and runner != MachineBackend.LOCAL:
        raise ValueError("Singularity image is not specified (in JSON or env variable)")

    if NODELIST is not None:
        NODELIST = "--nodelist=" + NODELIST

    grid = create_grid(PARAMS)
    grid = multiply_grid(grid, RUNS_MULTIPLIER)
    no_experiments = len(grid)
    minutes_per_exp = timestr_to_minutes(TIME)

    if len(grid) > 1 and runner == MachineBackend.LOCAL and not DRY_RUN:
        raise ValueError(
            f"Running more than one experiment locally is not supported (you are trying to run {len(grid)} experiments). Aborting..."
        )

    total_minutes = no_experiments * minutes_per_exp
    if not runner == MachineBackend.LOCAL:
        if not INTERACTIVE_DEBUG:
            user_input = input(
                f"Will run {no_experiments} experiments, using up {total_minutes} minutes, i.e. around {round(total_minutes / 60)} hours"
                f"\nExperiment settings: \n{RUNNER=} \n{TIME=} \n{N_GPUS=} \nContinue? [Y/n] "
            )
        else:
            user_input = input(f"Will run an INTERACTIVE experiment. \nContinue? [Y/n]")
        if user_input.lower() not in ("", "y", "Y"):
            print("Aborting...")
            exit(1)

    slurm_command = "srun" if INTERACTIVE_DEBUG else "sbatch"

    if not (INTERACTIVE_DEBUG or runner == MachineBackend.LOCAL):
        exp_name = next(iter(grid))["name"]
        name_for_branch = (
            f"{exp_name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )
        COPIED_CODE_PATH = copy_and_version_code(
            name_for_branch, name_for_branch, PUSH_TO_GIT
        )
    else:
        print(
            f"Running in debug mode or locally, skip copying code to a new directory."
        )

    for i, param_set in enumerate(grid):
        name = param_set["name"]
        param_set["n_gpus"] = N_GPUS
        env = None

        runner_params = []
        for k_packed, v_packed in param_set.items():
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
                f"--gres={GRES}",
                f"--job-name={name}",
                f"--time={TIME}",
                get_grid_entrypoint(runner),
                "python3",
                "-m",
                RUNNER,
                *runner_params,
            ]
        elif runner == MachineBackend.ATHENA:
            subprocess_args = [
                slurm_command,
                f"--gres=gpu:{N_GPUS}",
                "--partition=plgrid-gpu-a100",
                f"--cpus-per-gpu={CPUS_PER_GPU}",
                "--account=plgplggllmeffi-gpu-a100",
                f"--job-name={name}",
                f"--time={TIME}",
                get_grid_entrypoint(runner),
                "singularity",
                "run",
                "--bind=/net:/net",
                f"--env",
                f"HF_DATASETS_CACHE={HF_DATASETS_CACHE}",
                f"-B={CODE_PATH}:/sparsity,{HF_DATASETS_CACHE}:{HF_DATASETS_CACHE}",
                "--nv",
                SINGULARITY_IMAGE,
                "python3",
                "-m",
                RUNNER,
                *runner_params,
            ]
        elif runner == MachineBackend.IDEAS:
            subprocess_args = [
                slurm_command,
                f"--gres=gpu:{N_GPUS}",
                f"--cpus-per-gpu={CPUS_PER_GPU}",
                f"--job-name={name}",
                f"--time={TIME}",
                "--mem=32G",
                NODELIST,
                get_grid_entrypoint(runner),
                "singularity",
                "run",
                f"--env",
                f"HF_DATASETS_CACHE={HF_DATASETS_CACHE}",
                f"-B={CODE_PATH}:/sparsity,{HF_DATASETS_CACHE}:{HF_DATASETS_CACHE}",
                "--nv",
                SINGULARITY_IMAGE,
                "python3",
                "-m",
                RUNNER,
                *runner_params,
            ]
        elif runner == MachineBackend.ENTROPY_GPU:
            if CUDA_VISIBLE_DEVICES is not None:
                env = os.environ.copy()
                env.update({"CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES})
            subprocess_args = [
                "singularity",
                "run",
                f"--env",
                f"HF_DATASETS_CACHE={HF_DATASETS_CACHE}",
                f"-B={CODE_PATH}:/sparsity,{HF_DATASETS_CACHE}:{HF_DATASETS_CACHE}",
                "--nv",
                SINGULARITY_IMAGE,
                "python3",
                "-m",
                RUNNER,
                *runner_params,
            ]
        elif runner == MachineBackend.LOCAL:
            # We run the experiment directly, not through a grid entrypoint script
            # because we want to be able to debug it
            runner_main_function = get_train_main_function(RUNNER)
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
