import os
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
    range = min(A.min(), B.min()), max(A.max(), B.max())
    plt.plot(range, range, color="grey", linestyle="--", linewidth=1)
    plt.xlabel("ln(predicted_perplexity)")
    plt.ylabel("ln(perplexity)")
    legend = plt.legend()
    rmse = scaling_law()[1]
    plt.title(f"Loss vs predicted loss for {scaling_law.name} (RMSE={rmse:.3f})")
    plt.tight_layout()
    filename = f"scaling_laws/plots/{scaling_law.name}/error_{group_by}.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()


def plot_params(scaling_laws, plot_dim, show_model_sizes, extrapolate_factor=2.0, plot_points=100, **_):
    axis_dim = plot_dim[0]
    if axis_dim == "predicted_loss":
        for scaling_law in scaling_laws:
            if any([d in scaling_law.fixed_params for d in plot_dim[1:]]):
                continue
            plot_loss_vs_predicted_loss(scaling_law, group_by=plot_dim[1:])
        return
    if axis_dim == "flops_save":
        flops_save = True
        axis_dim = "flops"
    full_flops = axis_dim == "flops" and len(plot_dim) == 3
    scaling_laws = [s for s in scaling_laws if axis_dim in s.params_set or axis_dim == "flops"]
    plt.figure(dpi=250)
    top, bottom = -np.inf, np.inf
    all_A = np.array([math.log10(r.dict()[axis_dim]) for s in scaling_laws for r in s.runs])
    A_values = np.linspace(all_A.min(), all_A.max() + extrapolate_factor*(all_A.max() - all_A.min()), plot_points)
    for ii, scaling_law in enumerate(scaling_laws):
        model_sizes = {k: (v, np.inf, dict(flops=np.inf)) for k, v in show_model_sizes.items()}
        A = np.array([math.log10(r.dict()[axis_dim]) for r in scaling_law.runs])
        B = np.array([math.log(r.loss) for r in scaling_law.runs])
        plot_minimal = axis_dim == "flops"

        group_dims = sorted(list(scaling_law.params_set - set(plot_dim)))
        groups = get_groups_by_dim(group_dims, scaling_law)
        cm_f = cm.get_cmap(scaling_law.cmap)
        colors = cm_f(np.linspace(0, 1, len(groups)))

        B_predictions, names = [], []
        for (group, indices), color in zip(groups, colors):
            group_dict = dict(zip(group_dims, group))
            names.append(f"{scaling_law.name} {' '.join(f'{name}={int(val)}' for name, val in group_dict.items())}")
            plt.scatter(A[indices], B[indices], color=color, s=5)
            results = [scaling_law.resolve_params(**group_dict, **{axis_dim: np.power(10, a)}) for a in A_values]
            if full_flops:
                for pred, params in results:
                    for k, (size, perplexity, old_params) in model_sizes.items():
                        if pred < perplexity and params['n_params'] >= float(size) and old_params['flops'] > params['flops']:
                            model_sizes[k] = (size, pred, params)
            b_preds, _ = zip(*results)
            B_predictions.append(b_preds)

        B_predictions = np.array(B_predictions)
        is_min = B_predictions.min(axis=0) == B_predictions

        for B_p, color, name, minimal in zip(B_predictions,  colors, names, is_min):
            if not plot_minimal:
                plt.plot(A_values, B_p, color=color, label=name)
                continue
            plt.plot(A_values[~minimal], B_p[~minimal], color=color, linestyle="--", linewidth=0.7, alpha=0.5)
            plt.plot(A_values[minimal], B_p[minimal], color=color, linestyle="-", linewidth=2, label=name)
        top = max(top, min(B.max() + 0.2*(B.max() - B.min()), B_predictions.max()))
        bottom = min(bottom, min(B_predictions.min(), B.min()))

        if full_flops:
            for k, (size, perplexity, params) in model_sizes.items():
                if not np.isfinite(perplexity):
                    continue
                plt.scatter(math.log10(params['flops']), perplexity, color="black", s=6, marker='x')
                plt.text(math.log10(params['flops']), perplexity, f"{k} steps={params['n_steps']:.2E}", color="grey", fontsize=4, rotation=30 + ii*30)

    plt.title('\n'.join([str(s) for s in scaling_laws]), wrap=True, fontsize=5)
    plt.rc('font', size=7)
    plt.ylim(top=top, bottom=bottom)
    plt.rc('legend', fontsize=5)
    plt.rc('axes', titlesize=6, labelsize=6)
    plt.xlabel(f"log10({axis_dim})")
    plt.ylabel("ln(perplexity)")
    plt.tight_layout()
    legend = plt.legend(loc='upper right')
    legend.get_frame().set_alpha(0.4)
    filename = f"scaling_laws/plots/{'_'.join([s.name for s in scaling_laws])}/{axis_dim}|{'|'.join(plot_dim[1:])}.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()


def one_scaling(project, tags, fixed, **params):
    runs = download_batch_sizes_from_neptune(project, tags, fixed)
    scaling_law = ScalingLaw(runs=runs, fixed=fixed, **params)
    rmse = scaling_law.optimize()
    print(f"Final {scaling_law.name} scaling law approximation RMSE: {rmse}")
    scaling_law.present_values_as_chinchila()
    return scaling_law


def compute_scaling_laws(project_name, scalings, plot_dims, config, **params):
    project = neptune_connect(project_name)
    scaling_laws = [one_scaling(project=project, **s_config, **config)
                    for s_config in scalings]

    for plot_dim in plot_dims:
        plot_params(scaling_laws, plot_dim, **params)


def run_from_config():
    config = read_yaml_file()
    print(config)
    compute_scaling_laws(**config, config=config)


if __name__ == "__main__":
    run_from_config()
