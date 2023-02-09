"""
Script to grid search in recycle layers. Run this script from the root of the project:
$ python3 research/reinitialization/scripts/grid.py
Remember to set TRAINER and PARAMS in the script or add an argument parser.
"""

import copy
import datetime
from itertools import product
import subprocess
from typing import List, Tuple
import os
import sys
import json
from time import sleep
import platform
from enum import Enum
import os

from lizrd.scripts.experiment_code_versioning import experiment_code_versioning


class Runner(Enum):
    ENTROPY = 1
    ATHENA = 2
    LOCAL = 3


def get_runner() -> Runner:
    node = platform.uname().node
    if node == "asusgpu0":
        return Runner.ENTROPY
    elif "athena" in node:
        return Runner.ATHENA
    else:
        return Runner.LOCAL


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


def shorten_arg(arg: str) -> str:
    ARG_TO_ABBR = {
        "reinit_dist": "rd",
        "ff_layer": "ff",
        "mask_loss_weight": "mlw",
        "class_loss_weight": "clw",
        "mask_percent": "mp",
        "n_steps": "ns",
        "n_steps_eval": "nse",
        "immunity": "im",
        "pruner": "pr",
        "pruner_prob": "prp",
        "pruner_delay": "prd",
        "pruner_n_steps": "prns",
    }
    return ARG_TO_ABBR.get(arg, arg)


def shorten_val(val: str) -> str:
    VAL_TO_ABBR = {
        # ff layers
        "regular": "r",
        "unstruct_prune": "up",
        "struct_prune": "sp",
        "unstruct_magnitude_prune": "ump",
        "struct_magnitude_prune": "smp",
        "unstruct_magnitude_recycle": "umr",
        "struct_magnitude_recycle_with_immunity": "smri",
        "masked_ff": "mf",
        "separate_direction_magnitude_ff": "sdmf",
        # reinit dist
        "zero": "0",
        "init": "i",
        "follow_normal": "fn",
    }
    if isinstance(val, bool):
        return "T" if val else "F"
    if isinstance(val, str):
        return VAL_TO_ABBR.get(val, val)
    if isinstance(val, int):
        if val % 1_000_000 == 0:
            return f"{val // 1_000_000}M"
        if val % 1_000 == 0:
            return f"{val // 1_000}k"
        return str(val)

    return str(val)


def make_tags(arg, val) -> str:
    return f"{shorten_arg(arg)}={shorten_val(val)}"


# parse time to minutes
def timestr_to_minutes(time: str) -> int:
    if "-" in time:
        days, hours, minutes, seconds = time.split("-")[1].split(":")
    else:
        days, hours, minutes, seconds = 0, *time.split(":")
    return int(days) * 24 * 60 + int(hours) * 60 + int(minutes)


def create_grid(params: dict) -> List[dict]:
    grids, functions, normals = split_params(params)
    base_params = {k: v for k, v in normals}
    out_params = []
    grids_keys = [k for k, v in grids]
    grids_values = product(*(v for k, v in grids))
    for value in grids_values:
        out_dict = copy.deepcopy(base_params)
        grid_dict = dict(zip(grids_keys, value))
        tags = [make_tags(k, v) for k, v in grid_dict.items()]
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


def get_grid_entrypoint(runner: Runner) -> str:
    if runner in [Runner.ENTROPY, Runner.LOCAL]:
        return "lizrd/scripts/grid_entrypoint.sh"
    elif runner == Runner.ATHENA:
        return "lizrd/scripts/grid_entrypoint_athena.sh"
    else:
        raise ValueError(f"Unknown runner: {runner}")


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
SINGULARITY_IMAGE = (
    "/net/pr2/projects/plgrid/plggllmeffi/images/sparsity_2023.02.09_09.25.42.sif"
)
CODE_PATH = os.getcwd()

if __name__ == "__main__":
    runner = get_runner()

    if len(sys.argv) > 1:
        grid_args = json.load(open(sys.argv[1]))
        TRAINER = grid_args.get("trainer", TRAINER)
        PARAMS = grid_args.get("params", PARAMS)
        TIME = grid_args.get("time", TIME)
        GRES = grid_args.get("gres", GRES)
        DRY_RUN = grid_args.get("dry_run", DRY_RUN)
        SINGULARITY_IMAGE = grid_args.get("singularity_image", SINGULARITY_IMAGE)

    grid = create_grid(PARAMS)
    no_experiments = len(grid)
    minutes_per_exp = timestr_to_minutes(TIME)

    if len(grid) > 1 and runner == Runner.LOCAL and not DRY_RUN:
        print(
            f"Running more than one experiment locally is not supported (you are trying to run {len(grid)} experiments). Aborting..."
        )
        exit(1)

    name = next(iter(grid))["name"]
    name_for_branch = f"{name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    experiment_code_versioning(name_for_branch)

    user_input = input(
        f"Will run {no_experiments} experiments, using up {no_experiments * minutes_per_exp} minutes."
        f"\nSbatch settings: \n{TRAINER=} \n{TIME=} \n{GRES=} \nContinue? [Y/n] "
    )
    if user_input.lower() not in ("", "y", "Y"):
        print("Aborting...")
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
        if runner == Runner.ENTROPY:
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
                TRAINER,
                *trainer_params,
            ]
        elif runner == Runner.ATHENA:
            subprocess_args = [
                "sbatch",
                "--partition=plgrid-gpu-a100",
                "-G1",
                f"--job-name={name}",
                f"--time={TIME}",
                get_grid_entrypoint(runner),
                "singularity",
                "exec",
                "--bind=/net:/net",
                "--env HF_DATASETS_CACHE=/net/pr2/projects/plgrid/plggllmeffi/.cache",
                f"-B={CODE_PATH}:/sparsity",
                "--nv",
                SINGULARITY_IMAGE,
                "python3",
                "-m",
                TRAINER,
                *trainer_params,
            ]
        elif runner == Runner.LOCAL:
            subprocess_args = [
                get_grid_entrypoint(runner),
                "python3",
                "-m",
                TRAINER,
                *trainer_params,
            ]
        else:
            raise ValueError(f"Unknown runner: {runner}")

        if not DRY_RUN:
            subprocess.run(
                [str(s) for s in subprocess_args],
            )
            sleep(1)
        else:
            print(" ".join([str(s) for s in subprocess_args]))
