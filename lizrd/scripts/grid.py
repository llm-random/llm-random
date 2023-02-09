"""
Script to grid search in recycle layers. Run this script from the root of the project:
$ python3 research/reinitialization/scripts/grid.py
Remember to set TRAINER and PARAMS in the script or add an argument parser.
"""

import datetime
import subprocess
import os
import sys
import json
from time import sleep
from lizrd.scripts.grid_utils import create_grid, timestr_to_minutes

from lizrd.scripts.experiment_code_versioning import experiment_code_versioning


TRAINER = "research.reinitialization.train.reinit_train"

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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        grid_args = json.load(open(sys.argv[1]))
        TRAINER = grid_args.get("trainer", TRAINER)
        PARAMS = grid_args.get("params", PARAMS)
        TIME = grid_args.get("time", TIME)
        GRES = grid_args.get("gres", GRES)
        DRY_RUN = grid_args.get("dry_run", DRY_RUN)

    grid = create_grid(PARAMS)
    no_experiments = len(grid)
    minutes_per_exp = timestr_to_minutes(TIME)

    name = next(iter(grid))["name"]
    name_for_branch = f"{name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    experiment_code_versioning(name_for_branch)

    user_input = input(
        f"Will run {no_experiments} experiments, using up {no_experiments * minutes_per_exp} minutes."
        f"\nSbatch settings: \n{TRAINER=} \n{TIME=} \n{GRES=} \nContinue? [Y/n] "
    )
    if user_input.lower() not in ("", "y", "Y"):
        print("Aborting")
        exit(1)

    for i, param_set in enumerate(grid):
        name = param_set["name"]
        param_set["tags"] = " ".join(param_set["tags"])

        trainer_params = []
        for k, v in param_set.items():
            if isinstance(v, bool):
                if v:
                    trainer_params.append(f"--{k}")
                else:
                    pass  # simply don't add it if v == False
                continue
            else:
                trainer_params.append(f"--{k}")
                trainer_params.append(v)

        subprocess_args = [
            "sbatch",
            "--partition=common",
            "--qos=16gpu7d",
            f"--gres={GRES}",
            f"--job-name={name}",
            f"--time={TIME}",
            "lizrd/scripts/grid_entrypoint.sh",
            "python3",
            "-m",
            TRAINER,
            *trainer_params,
        ]

        if not DRY_RUN:
            subprocess.run(
                [str(s) for s in subprocess_args],
            )
            sleep(1)
        else:
            print(" ".join([str(s) for s in subprocess_args]))
