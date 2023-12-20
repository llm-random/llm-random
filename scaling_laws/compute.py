import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from scaling_laws.scaling import ScalingLaw
from scaling_laws.utils import neptune_connect, download_batch_sizes_from_neptune, \
    read_yaml_file, get_groups_by_dim


def plot_loss_vs_predicted_loss(scaling_law, group_by="granularity"):
    groups = get_groups_by_dim(group_by, scaling_law)
    colors = cm.rainbow(np.linspace(0, 1, len(groups)))
    A = np.array([scaling_law.expected_logloss(**r.dict()).detach().numpy() for r in scaling_law.runs])
    B = np.array([math.log(r.loss) for r in scaling_law.runs])
    plt.figure(dpi=200)
    for (group, indices), color in zip(groups, colors):
        group_dict = dict(zip(group_by, group))
        label = " ".join(f"{name}={int(val)}" for name, val in group_dict.items())
        plt.scatter(A[indices], B[indices], color=color, s=3, label=label)
    plt.xlabel("ln(predicted_perplexity)")
    plt.ylabel("ln(perplexity)")
    legend = plt.legend()
    plt.title(f"Loss vs predicted loss for {scaling_law.name}")
    plt.tight_layout()
    plt.savefig(f"{scaling_law.name}_error_{group_by}.png")
    plt.show()


def plot_params(scaling_law, plot_dim, plot_points=1000):
    if plot_dim[0] == "predicted_loss":
        plot_loss_vs_predicted_loss(scaling_law, group_by=plot_dim[1:])
        return
    axis_dim = plot_dim[0]
    plt.figure(dpi=200)
    A = np.array([math.log10(r.dict()[axis_dim]) for r in scaling_law.runs])
    B = np.array([math.log(r.loss) for r in scaling_law.runs])
    A_values = np.linspace(A.min(), A.max() + 2.0*(A.max() - A.min()), plot_points)
    plot_minimal = axis_dim == "flops"

    group_dims = sorted(list(scaling_law.params_set - set(plot_dim)))
    groups = get_groups_by_dim(group_dims, scaling_law)
    colors = cm.rainbow(np.linspace(0, 1, len(groups)))

    B_predictions, names = [], []
    for (group, indices), color in zip(groups, colors):
        group_dict = dict(zip(group_dims, group))
        names.append(" ".join(f"{name}={int(val)}" for name, val in group_dict.items()))
        plt.scatter(A[indices], B[indices], color=color, s=5)
        B_predictions.append([scaling_law.resolve_params(**group_dict, **{axis_dim: np.power(10, a)}) for a in A_values])

    B_predictions = np.array(B_predictions)
    is_min = B_predictions.min(axis=0) == B_predictions

    for B_p, color, name, minimal in zip(B_predictions, colors, names, is_min):
        if not plot_minimal:
            plt.plot(A_values, B_p, color=color, label=name)
            continue
        plt.plot(A_values[~minimal], B_p[~minimal], color=color, linestyle="--", linewidth=0.7, alpha=0.5)
        plt.plot(A_values[minimal], B_p[minimal], color=color, linestyle="-", linewidth=2, label=name)

    plt.title(str(scaling_law), wrap=True, fontsize=6)
    plt.rc('font', size=7)
    top = min(B.max() + 0.2*(B.max() - B.min()), B_predictions.max())
    bottom = min(B_predictions.min(), B.min())
    plt.ylim(top=top, bottom=bottom)
    plt.rc('legend', fontsize=5)
    plt.rc('axes', titlesize=6, labelsize=6)
    plt.xlabel(f"log10({axis_dim})")
    plt.ylabel("ln(perplexity)")
    plt.tight_layout()
    legend = plt.legend(loc='upper right')
    legend.get_frame().set_alpha(0.4)
    plt.savefig(f"{scaling_law.name}_{axis_dim}|{'|'.join(plot_dim[1:])}.png")
    plt.show()


def one_scaling(project, tags, power_laws, name, num_opt_steps, fixed, use_chinchilla, **_):
    runs = download_batch_sizes_from_neptune(project, tags, fixed)
    scaling_law = ScalingLaw(name, runs, power_laws, fixed, use_chinchilla=use_chinchilla)
    eval = scaling_law.optimize(num_steps=num_opt_steps)
    print(f"Final scaling law approximation RMSE: {scaling_law()[1]}")
    return scaling_law


def compute_scaling_laws(project_name, scalings, plot_dims, config, **_):
    project = neptune_connect(project_name)
    scaling_laws = [one_scaling(project=project, **s_config, **config)
                    for s_config in scalings]

    for plot_dim in plot_dims:
        for scaling_law in scaling_laws:
            plot_params(scaling_law, plot_dim)


def run_from_config():
    config = read_yaml_file()
    print(config)
    compute_scaling_laws(**config, config=config)


if __name__ == "__main__":
    run_from_config()
