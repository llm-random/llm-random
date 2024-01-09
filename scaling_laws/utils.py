import argparse
import os
from collections import defaultdict

import neptune
import numpy as np
import pandas as pd
import yaml

from scaling_laws.calculate_params import TrainRun


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


def download_batch_sizes_from_neptune(project, tags, tags_negative, fixed):
    print("Downloading data from Neptune...")
    table = pd.concat([project.fetch_runs_table(tag=tag).to_pandas() for tag in tags])
    table.rename(columns=lambda x: x.replace("/", "_"), inplace=True)
    table = [row for _, row in table.iterrows() if all(tag not in tags_negative for tag in row["sys_tags"].split(','))]
    table = [TrainRun(**row, fixed=fixed) for row in table]
    proper = np.average([t.finished for t in table])
    runs = [t for t in table if t.finished]
    print(f"{len(runs)} ({proper*100:.2f}%) of runs finished properly")
    return runs


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
