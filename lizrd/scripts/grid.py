"""
Script to grid search in recycle layers. Run this script from the root of the project:
$ python3 research/reinitialization/scripts/grid.py
Remember to set TRAINER and PARAMS in the script or add an argument parser.
"""

import copy
from itertools import product
import subprocess
from typing import List, Tuple
import os


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

TIME = "1-00:00:00"
DELAY = 0

PARAMS = {
    "ff_mode": "vanilla",
    "^learning_rate": [8e-4],
    "name": "vanilla_baseline",
    "attention_thinning_coeff": 0.2,
    "^dff": [512, 768, 1024, 1536, 2048, 3072, 4096],
    "use_clearml": True,
    "batch_size": 64,
    "cutoff": 128,
    "dmodel": 256,
    "n_att_heads": 4,
    "n_blocks": 4,
    "mask_percent": 0.15,
    "n_steps": 100001,
    "project_name": "nonlinearities/vanilla_dff_comparison",
}

if __name__ == "__main__":
    grid = create_grid(PARAMS)

    user_input = input(
        f"Will run {len(grid)} experiments. Sbatch settings: \n{TRAINER=} \n{TIME=} \n{DELAY=} \nContinue? [Y/n] "
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
            "--gres=gpu:titanv:1",
            f"--job-name={name}",
            f"--time={TIME}",
            # f"--begin=now+{DELAY}hour",
            f"--output=/home/simontwice/sparsity/not_important{i}.txt",
            "python3",
            "-m",
            "lizrd/scripts/grid_entrypoint.sh",
            TRAINER,
            *trainer_params,
        ]

        subprocess.run(
            [str(s) for s in subprocess_args],
        )
