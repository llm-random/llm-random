"""
Script to grid search in recycle layers. Run this script from the root of the project:
$ python3 research/reinitialization/scripts/grid.py
Remember to set RUNNER and PARAMS in the script or add an argument parser.
"""

import datetime
import json
import os
import subprocess
import sys
from time import sleep

from lizrd.scripts.grid_utils import (
    create_grid,
    multiply_grid,
    timestr_to_minutes,
    get_machine_backend,
    MachineBackend,
    get_grid_entrypoint,
)
from lizrd.support.code_versioning_support import version_code


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
SINGULARITY_IMAGE = (
    "/net/pr2/projects/plgrid/plggllmeffi/images/sparsity_2023.02.12_21.20.53.sif"
)
CODE_PATH = os.getcwd()
INTERACTIVE_DEBUG = False
RUNS_MULTIPLIER = 1

if __name__ == "__main__":
    runner = get_machine_backend()

    if len(sys.argv) > 1:
        grid_args = json.load(open(sys.argv[1]))
        RUNNER = grid_args.get("runner", RUNNER)
        PARAMS = grid_args.get("params", PARAMS)
        TIME = grid_args.get("time", TIME)
        GRES = grid_args.get("gres", GRES)
        DRY_RUN = grid_args.get("dry_run", DRY_RUN)
        SINGULARITY_IMAGE = grid_args.get("singularity_image", SINGULARITY_IMAGE)
        RUNS_MULTIPLIER = grid_args.get("runs_multiplier", RUNS_MULTIPLIER)
        INTERACTIVE_DEBUG = grid_args.get("interactive_debug", INTERACTIVE_DEBUG)

    grid = create_grid(PARAMS)
    grid = multiply_grid(grid, RUNS_MULTIPLIER)
    no_experiments = len(grid)
    minutes_per_exp = timestr_to_minutes(TIME)

    if len(grid) > 1 and runner == MachineBackend.LOCAL and not DRY_RUN:
        raise ValueError(
            f"Running more than one experiment locally is not supported (you are trying to run {len(grid)} experiments). Aborting..."
        )

    if not INTERACTIVE_DEBUG:
        exp_name = next(iter(grid))["name"]
        name_for_branch = (
            f"{exp_name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )
        print(f"Creating branch {name_for_branch}")
        version_code(name_for_branch, name_for_branch)
    else:
        print(f"Running in debug mode, skipping branch creation.")

    total_minutes = no_experiments * minutes_per_exp

    print(
        f"Will run {no_experiments} experiments, using up {total_minutes} minutes, i.e. around {round(total_minutes / 60)} hours"
        f"\nSbatch settings: \n{RUNNER=} \n{TIME=} \n{GRES=} \n"
    )
    if not INTERACTIVE_DEBUG:
        user_input = input("Continue? [Y/n] ")
        if user_input.lower() not in ("", "y", "Y"):
            print("Aborting...")
            exit(1)

    for i, param_set in enumerate(grid):
        name = param_set["name"]
        param_set["tags"] = " ".join(param_set["tags"])

        runner_params = []
        for k, v in param_set.items():
            if isinstance(v, bool):
                if v:
                    runner_params.append(f"--{k}")
                else:
                    pass  # simply don't add it if v == False
                continue
            else:
                runner_params.append(f"--{k}")
                if isinstance(v, list):
                    v = " ".join([str(s) for s in v])
                runner_params.append(v)
        if runner == MachineBackend.ENTROPY:
            subprocess_args = [
                "sbatch",
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
            run_command = "srun" if INTERACTIVE_DEBUG else "sbatch"
            subprocess_args = [
                run_command,
                "--partition=plgrid-gpu-a100",
                "-G1",
                "--cpus-per-gpu=8",
                f"--job-name={name}",
                f"--time={TIME}",
                get_grid_entrypoint(runner),
                "singularity",
                "run",
                "--bind=/net:/net",
                "--env HF_DATASETS_CACHE=/net/pr2/projects/plgrid/plggllmeffi/.cache",
                f"-B={CODE_PATH}:/sparsity",
                "--nv",
                SINGULARITY_IMAGE,
                "python3",
                "-m",
                RUNNER,
                *runner_params,
            ]
        elif runner == MachineBackend.LOCAL:
            subprocess_args = [
                get_grid_entrypoint(runner),
                "python3",
                "-m",
                RUNNER,
                *runner_params,
            ]
        else:
            raise ValueError(f"Unknown runner: {runner}")

        if not DRY_RUN:
            subprocess.run(
                [str(s) for s in subprocess_args],
            )
            sleep(10)
        else:
            print(" ".join([str(s) for s in subprocess_args]))
