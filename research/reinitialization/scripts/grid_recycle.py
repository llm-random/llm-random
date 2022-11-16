"""
Script to grid search in recycle layers
"""

from itertools import product
import subprocess

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

# custom functions to calculate parameters
def pruner_prob(grid_params):
    return 0

GRID_PARAMS = {
    "pruner_n_steps": [100, 1000, 10000],
    "total_frac_pruned": [0.25, 1, 4],
}

SCRIPT_PARAMS = {
    "pruner_prob": pruner_prob,
    "project_name": ["jkrajewski/reinitialization"],
    "name": ["grid_recycle"],
    "ff_layer": ["recycle"],
    "tag": ""
}

def find_tags(param_dict):
    tags = SCRIPT_PARAMS["tag"]
    for param_name in param_dict:
        tags += f"{param_name}_{param_dict[param_name]} "

def find_param_value(param_dict, param_name):
    if param_name in SCRIPT_PARAMS:
        if type(SCRIPT_PARAMS[param_name]) == list:
            return SCRIPT_PARAMS[param_name]
        else:
            return SCRIPT_PARAMS[param_name](param_dict)
    else:
        return param_dict[param_name]

for param_dict in product_dict(GRID_PARAMS):
    tags = find_tags(param_dict)

    subprocess.run(
        "sbatch",
        "--partition=common",
        "--qos=16gpu7d",
        "--gres=gpu:titanv:1",
        f"--job-name=${name}",
        f"--output=/home/jkrajewski/${name}.txt",
        "--time=0-10:00:00",
    )
