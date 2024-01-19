import argparse
import os
from collections import defaultdict

import neptune
import yaml


def unique_values_with_indices(data):
    indices = defaultdict(list)
    for idx, val in enumerate(data):
        indices[val].append(idx)
    return sorted(indices.items())


def get_groups_by_dim(group_dims, scaling_law):
    dicts = [params.dict() for params in scaling_law.runs]
    group_values = [tuple([params[d] for d in group_dims]) for params in dicts]
    groups = unique_values_with_indices(group_values)
    return groups


def neptune_connect(project_name):
    api_token = os.environ["NEPTUNE_API_TOKEN"]
    return neptune.init_project(api_token=api_token, project=project_name)


def read_yaml_file():
    parser = argparse.ArgumentParser(description="Read a yaml file")
    parser.add_argument("config_path", type=str, help="Path to the YAML file")
    args = parser.parse_args()

    with open(args.config_path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)


def binary_search(range, fun, tol=0.02):
    min_val, max_val = range
    while max_val - min_val > tol:
        mid_val = (max_val + min_val) / 2
        if fun(mid_val) < 0:
            max_val = mid_val
        else:
            min_val = mid_val
    return (max_val + min_val) / 2
