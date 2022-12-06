"""
Script to grid search in recycle layers. Run this script from the root of the project:
$ python3 research/reinitialization/scripts/grid.py
Remember to set TRAINER and PARAMS in the script or add an argument parser.
"""

import copy
from itertools import product
import subprocess
from time import sleep
from typing import List, Tuple
import sys
import json


def split_params(params: dict) -> Tuple[list, list, list]:
    functions = []
    grids = []
    normals = []
    for k, v in params.items():
        if k[0] == "^":
            grids.append((k[1:], v))
        elif k[0] == "*":
            functions.append((k[1:], v))
        else:
            normals.append((k, v))
    return grids, functions, normals


def create_grid(params: dict) -> List[dict]:
    grids, functions, normals = split_params(params)
    base_params = {k: v for k, v in normals}
    out_params = []
    grids_keys = [k for k, v in grids]
    grids_values = product(*(v for k, v in grids))
    for value in grids_values:
        out_dict = copy.deepcopy(base_params)
        grid_dict = dict(zip(grids_keys, value))
        tags = [f"{k}={str(v)}" for k, v in grid_dict.items()]
        if out_dict.get("tags") is None:
            out_dict["tags"] = []
        out_dict["tags"].extend(tags)
        out_dict = {**out_dict, **grid_dict}
        for func_name, func in functions:
            out_dict[func_name] = func(out_dict)
        out_params.append(out_dict)

    return out_params


def param_to_str(param) -> str:
    if isinstance(param, str):
        return " ".join(param)
    else:
        return str(param)


TRAINER = "research.nonlinearities.train.nonlinearities_train"

# ^ - grid over that
# * - apply function

PARAMS = {
    # "deterministic": False,
    "ff_mode": "vanilla",
    "^dff": [128, 512, 1024, 2048],
    # "^seed": [2137, 69, 420],
    "^attention_thinning_coeff": [0.7, 0.3, 0.1, 0.03],
    "name": "att_size_influence",
    "use_clearml": True,
    "batch_size": 64,
    "cutoff": 128,
    "dmodel": 256,
    "dff": 1024,
    "n_att_heads": 8,
    "learning_rate": 8e-4,
    "n_blocks": 4,
    "mask_percent": 0.15,
    "n_steps": 100001,
    "project_name": "nonlinearities/common_setup_experiments/att_size_dff_incluence",
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        grid_args = json.load(open(sys.argv[1]))
        TRAINER = grid_args["trainer"]
        PARAMS = grid_args["params"]

    grid = create_grid(PARAMS)

    user_input = input(f"Will run {len(grid)} experiments. [Y/n]")
    if user_input.lower() not in ("", "y", "Y"):
        print("Aborting")
        exit(1)

    for i, param_set in enumerate(grid):
        name = param_set["name"]
        param_set["tags"] = " ".join(param_set["tags"])

        trainer_params = []
        for k, v in param_set.items():
            if type(v) == bool and v in [True, False]:
                if v:
                    trainer_params.append(f"--{k}")
                else:
                    pass  # simply don't add it if v == False
                continue
            else:
                trainer_params.append(f"--{k}")
                if v != "" and v is not None:
                    trainer_params.append(v)

        subprocess_args = [
            "sbatch",
            "--partition=common",
            "--qos=16gpu7d",
            "--gres=gpu:1",
            f"--job-name={name}",
            "--time=3-00:00:00",
            f"--output=/home/simontwice/sparsity/not_important2_{i}.txt",
            "lizrd/scripts/grid_entrypoint.sh",
            "python3",
            "-m",
            TRAINER,
            *trainer_params,
        ]
        sleep(0.5)
        subprocess.run(
            [str(s) for s in subprocess_args],
        )
