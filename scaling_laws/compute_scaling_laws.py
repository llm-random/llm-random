import neptune
import os
import pandas as pd
import torch.nn as nn
import torch
import math
import numpy as np
import argparse
import yaml
from tqdm import trange
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.pyplot import cm
from itertools import chain


def unique_values_with_indices(data):
    indices = defaultdict(list)
    for idx, val in enumerate(data):
        indices[val].append(idx)
    return sorted(indices.items())


def neptune_connect(project_name):
    api_token = os.environ["NEPTUNE_API_TOKEN"]
    return neptune.init_project(api_token=api_token, project=project_name)


def download_batch_sizes_from_neptune(project, tags):
    table = pd.concat([project.fetch_runs_table(tag=tag).to_pandas() for tag in tags])
    table.rename(columns=lambda x: x.replace("/", "_"), inplace=True)
    return [TrainRun(**row) for _, row in table.iterrows()]


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


class TrainRun:
    def __init__(
        self,
        loss,
        args_granularity,
        args_expansion_rate,
        args_dmodel,
        args_n_steps,
        args_n_blocks,
        **_,
    ):
        self.loss = loss
        self.granularity = args_granularity
        self.expansion_rate = args_expansion_rate
        self.dmodel = args_dmodel
        self.n_steps = args_n_steps
        self.n_blocks = args_n_blocks
        self.n_params = self.calculate_params()
        self.iter_flops = self.calculate_iter_flops()
        self.flops = self.iter_flops * self.n_steps

    def calculate_iter_flops(self):
        return self.dmodel**2 * 4 * self.n_blocks  # TODO SHIT

    # no embedding
    def calculate_params(self):
        return (self.dmodel * self.expansion_rate + 4*self.dmodel**2) * self.n_blocks

    def dict(self):
        return self.__dict__

    def __repr__(self):
        return f"({self.dict()})"


class PowerLaw(nn.Module):
    def __init__(self, names, eps=1e-5):
        super(PowerLaw, self).__init__()
        self.p = nn.Parameter(torch.tensor(eps))
        self.names = names
        self.name = "*".join([f"ln({name})" for name in names])

    def forward(self, **params):
        value = torch.tensor(1.)
        for name in self.names:
            param = params.get(name, None)
            if param is None:
                raise Exception(f"No {self.name} param found in {params})")
            value *= torch.log(torch.tensor(param))
        return value * self.p

    def __repr__(self):
        return f"{self.p.item():.2}*{self.name}"


class ScalingLaw(nn.Module):
    def __init__(self, name, runs, power_laws, eps=1e-5):
        super().__init__()
        self.runs = runs
        self.name = name
        self.L0 = nn.Parameter(torch.tensor(eps))
        self.power_laws = nn.ModuleList([PowerLaw(names, eps) for names in power_laws])

    def expected_logloss(self, **params):
        return self.L0 + sum([p(**params) for p in self.power_laws])

    def __repr__(self):
        return (
            f"Scaling \"{self.name}\" {' + '.join([str(p) for p in self.power_laws])}"
            f" + {self.L0.item():.2}"
        )

    def forward(self):
        return sum(
            [
                (self.expected_logloss(**run.dict()) - math.log(run.loss)) ** 2
                for run in self.runs
            ]
        )


def optimize(model, num_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.train()
    with trange(num_steps) as iterator:
        for _ in iterator:
            optimizer.zero_grad()
            loss = model()
            iterator.set_description(f"Optimizing, error={loss:.4} {str(model)}")
            loss.backward()
            optimizer.step()


def plot_params(scaling_law, plot_dim, plot_points=1000):
    A = np.array([math.log(r.dict()[plot_dim]) for r in scaling_law.runs])
    B = np.array([math.log(r.loss) for r in scaling_law.runs])
    A_values = np.linspace(A.min(), A.max(), plot_points)

    group_dims = set(chain(*(set(p.names) for p in scaling_law.power_laws))) - {plot_dim}
    group_dims = sorted(list(group_dims))
    dicts = [params.dict() for params in scaling_law.runs]
    group_values = [tuple([params[d] for d in group_dims]) for params in dicts]
    groups = unique_values_with_indices(group_values)
    color = cm.rainbow(np.linspace(0, 1, len(groups)))
    plt.figure(dpi=200)

    for (group, indices), color in zip(groups, color):
        group_dict = dict(zip(group_dims, group))
        name = " ".join(f"{name}={val}" for name, val in group_dict.items())
        plt.scatter(A[indices], B[indices], color=color)

        B_predictions = [
            scaling_law.expected_logloss(**group_dict, **{plot_dim: np.exp(a)}).detach().numpy()
            for a in A_values
        ]
        plt.plot(A_values, B_predictions, color=color, label=name)
    plt.title(str(scaling_law), wrap=True, fontsize=6)
    plt.rc('font', size=7)
    plt.rc('legend', fontsize=6)
    plt.rc('axes', titlesize=6, labelsize=6)
    plt.xlabel(f"ln({plot_dim})")
    plt.ylabel("logperplexity")
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()


def compute_scaling_laws():
    config = read_yaml_file()
    print(config)

    project = neptune_connect(config["project_name"])

    for scaling_config in config["scalings"]:
        runs = download_batch_sizes_from_neptune(project, scaling_config["tags"])
        scaling_law = ScalingLaw(
            scaling_config["name"], runs, scaling_config["power_laws"]
        )
        optimize(scaling_law, num_steps=config["num_opt_steps"])

        plot_params(scaling_law, scaling_config["plot_dim"])

    pass


if __name__ == "__main__":
    compute_scaling_laws()
