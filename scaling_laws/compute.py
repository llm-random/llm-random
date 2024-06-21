import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import cm
import matplotlib.ticker as mticker

import json

from scaling_laws.calculate_params import TrainRun
from scaling_laws.scaling import ScalingLaw
from scaling_laws.utils import neptune_connect, read_yaml_file, get_groups_by_dim


def plot_loss_vs_predicted_loss(scaling_law, no_title=False, group_by="granularity"):
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
    plt.xlabel("ln(predicted_loss)")
    plt.ylabel("ln(loss)")
    legend = plt.legend()
    rmse = scaling_law()[1]
    if not no_title:
        plt.title(f"Loss vs predicted loss for {scaling_law.name} (RMSE={rmse:.3f})")
    plt.tight_layout()
    filename = f"scaling_laws/plots/{scaling_law.name}/error_{group_by}.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()


def flops_save(scaling_laws, plot_points, no_title):
    get_best_params = lambda **params: scaling_laws[1].resolve_params(**params)
    get_baseline_params = lambda **params: scaling_laws[1].resolve_params(granularity=1, **params)
    get_denseline_params = lambda **params: scaling_laws[0].resolve_params(**params)

    start_point, end_point = map(math.log10, scaling_laws[-1].flops_range)
    A_values = 10**np.linspace(start_point, end_point, plot_points//4)

    best_params = [get_best_params(flops=f) for f in A_values]
    baseline_flops = np.array([get_baseline_params(loss=c["loss"])["flops"] for c in best_params])
    denseline_flops = np.array([get_denseline_params(loss=c["loss"])["flops"] for c in best_params])
    best_flops = np.array([c["flops"] for c in best_params])
    plt.figure(dpi=250)

    colors = cm.rainbow(np.linspace(0, 1, 10))

    plt.plot(best_flops, best_flops/best_flops, color=colors[1], linestyle="-", linewidth=2, label="Granular MoE")
    plt.plot(best_flops, baseline_flops/best_flops, color=colors[5], linestyle="-", linewidth=2, label="MoE")
    plt.plot(best_flops, denseline_flops/best_flops, color=colors[8], linestyle="-", linewidth=2, label="Transformer")

    if not no_title:
        plt.title("FLOPS overhead needed to achieve Granular MoE performance")
    plt.yscale("log")
    plt.xscale("log")
    ax = plt.gca()
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    plt.rc('font', size=10)
    plt.rc('legend', fontsize=10)
    plt.rc('axes', titlesize=10, labelsize=10)
    plt.xlabel(f"FLOPS")
    plt.ylabel(f"x Overhead")
    plt.tight_layout()
    legend = plt.legend(loc='upper right')
    legend.get_frame().set_alpha(0.4)
    filename = f"scaling_laws/plots/flops_save.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()


def opt_plot(scaling_laws, plot_points, no_title):
    get_best_params = lambda **params: scaling_laws[1].resolve_params(**params)

    start_point, end_point = map(math.log10, scaling_laws[-1].flops_range)
    A_values = 10**np.linspace(start_point, end_point, plot_points)

    params = [get_best_params(flops=f) for f in A_values]
    N_opt = np.array([c["n_params_active"] for c in params])
    D_opt = np.array([c["n_steps"] for c in params])
    granularity = np.array([c["granularity"] for c in params])
    plt.figure(dpi=250)

    grans = [2**gi for gi in range(1, 7)]
    colors = cm.rainbow(np.linspace(0, 1, len(grans)))

    for g,c in zip(grans, colors):
        indices = np.where((g*0.8 <= granularity) & (granularity <= g*1.5))[0]
        if len(indices) > 0:
            plt.plot(D_opt[indices], N_opt[indices], color=c, linestyle="-", linewidth=2, label=f"Granularity={g}")
    if not no_title:
        plt.title("Optimal scaling of N D and G")
    plt.yscale("log")
    plt.xscale("log")
    ax = plt.gca()
    yticks = [10**i for i in range(9, 13)]
    xticks = [10**i for i in range(9, 14)]
    plt.yticks(yticks, [present_v(tt).replace('.00', '') for tt in yticks], size='small')
    plt.xticks(xticks, [present_v(tt).replace('.00', '') for tt in xticks], size='small')
    plt.rc('font', size=10)
    plt.rc('legend', fontsize=10)
    plt.rc('axes', titlesize=10, labelsize=10)
    plt.xlabel(f"Training Tokens")
    plt.ylabel(f"Active Parameters")
    plt.tight_layout()
    legend = plt.legend(loc='upper right')
    legend.get_frame().set_alpha(0.4)
    filename = f"scaling_laws/plots/opt_save.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()


def plot_params(scaling_laws, plot_dim, show_model_sizes, show_points=False, extrapolate_factor=2.0, plot_points=100, no_title=False, **_):
    axis_dim = plot_dim[0]
    if axis_dim == "predicted_loss":
        for scaling_law in scaling_laws:
            if any([d in scaling_law.fixed_params for d in plot_dim[1:]]):
                continue
            plot_loss_vs_predicted_loss(scaling_law, group_by=plot_dim[1:], no_title=no_title)
        return
    if axis_dim == "flops_save":
        flops_save(scaling_laws, plot_points=plot_points, no_title=no_title)
        return
    if axis_dim == "opt":
        opt_plot(scaling_laws, plot_points=plot_points, no_title=no_title)
        return
    full_flops = axis_dim == "flops" and len(plot_dim) == 3
    scaling_laws = [s for s in scaling_laws if axis_dim in s.params_set or axis_dim == "flops"]
    plt.figure(dpi=250)
    top, bottom = -np.inf, np.inf
    all_A = np.array([math.log10(r.dict()[axis_dim]) for s in scaling_laws for r in s.runs])
    start_point, end_point = map(math.log10, scaling_laws[-1].flops_range)
    A_values = np.linspace(start_point, end_point, plot_points)
    for ii, scaling_law in enumerate(scaling_laws):
        model_sizes = {k: (v, np.inf, dict(flops=np.inf)) for k, v in show_model_sizes.items()}
        A = np.array([math.log10(r.dict()[axis_dim]) for r in scaling_law.runs])
        B = np.array([math.log(r.loss) for r in scaling_law.runs])
        plot_minimal = axis_dim == "flops"

        group_dims = sorted(list(scaling_law.params_set - set(plot_dim)))
        groups = get_groups_by_dim(group_dims, scaling_law)
        cm_f = cm.get_cmap(scaling_law.cmap)
        if len(groups) > 2:
            groups.append(((32.0,), []))
            groups.append(((64.0,), []))
            #groups.append(((128.0,), []))  # TUTAJ ODKOMENTOWAĆ JAK SIĘ DODA NOWE GRANULARNOŚCI
        colors = cm_f(np.linspace(0, 1, len(groups)))

        B_predictions, B_opt_params, names = [], [], []
        for (group, indices), color in zip(groups, colors):
            group_dict = dict(zip(group_dims, group))
            names.append(f"{scaling_law.name} {' '.join(f'{name}={int(val)}' for name, val in group_dict.items())}")
            if show_points:
                plt.scatter(A[indices], B[indices], color=color, s=5)
            b_opt_params = [scaling_law.resolve_params(**group_dict, **{axis_dim: np.power(10, a)}) for a in A_values]
            b_preds = [Bb["logloss"] for Bb in b_opt_params]
            B_predictions.append(b_preds)
            B_opt_params.append(b_opt_params)

        B_predictions = np.array(B_predictions)
        is_min = B_predictions.min(axis=0) == B_predictions

        for B_p, b_opt, color, name, minimal in zip(B_predictions, B_opt_params, colors, names, is_min):
            if full_flops:
                for i, (pred, params) in enumerate(zip(B_p, b_opt)):
                    if not minimal[i]:
                        continue
                    for k, (size, perplexity, old_params) in model_sizes.items():
                        if pred < perplexity and params['n_params_active'] >= float(size) and old_params['flops'] > params['flops']:
                            model_sizes[k] = (size, pred, params)
            if not plot_minimal:
                plt.plot(A_values, B_p, color=color, label=name)
                continue
            if name == 'Granular Scaling granularity=1':
                pass
                #plt.plot(A_values[~minimal], B_p[~minimal], color="blue", linestyle="-", linewidth=2, label="Vanilla MoE")
            else:
                pass
                #plt.plot(A_values[~minimal], B_p[~minimal], color=color, linestyle="--", linewidth=0.5, alpha=0.5)
            if sum(minimal) > 1:
                plt.plot(A_values[minimal], B_p[minimal], color=color, linestyle="-", linewidth=2, label=f"MoE Granularity{name[28:]}" if len(name) > 15 else name)
        top = max(top, min(B.max() + 1.0*(B.max() - B.min()), B_predictions.max()))
        bottom = min(bottom, min(B_predictions.min(), B.min()))

        if full_flops and scaling_law.name != 'Dense':
            for k, (size, perplexity, params) in model_sizes.items():
                if not np.isfinite(perplexity) and perplexity > 0:
                    continue
                plt.scatter(math.log10(params['flops']), perplexity, color="black", s=6, marker='x')
                plt.text(math.log10(params['flops']), perplexity, f"{k}", color="black", fontsize=13, rotation=0 + ii*0)
                print(f"{k}: steps={params['n_steps']:.2E} flops={params['flops']:.2E} perplexity={perplexity:.2E} active_params={params['n_params_active']:.2E} total_params={params['n_params_total']:.2E}")

    if not no_title:
        plt.title('\n'.join([str(s) for s in scaling_laws]), wrap=True, fontsize=5)
    plt.rc('font', size=10)
    plt.ylim(top=top, bottom=bottom if np.isfinite(bottom) else 0)
    plt.rc('legend', fontsize=10)
    plt.rc('axes', titlesize=10, labelsize=10)

    yticks = [0.4,0.6,0.8,1.0,1.2]
    xticks = range(18, 27)
    plt.yticks(yticks, [f"{np.exp(tt):.2}" for tt in yticks], size='small')
    plt.xticks(xticks, [f"1E{tt}" for tt in xticks], size='small')
    plt.xlabel(f"{axis_dim.upper()}")
    plt.ylabel("loss")
    plt.tight_layout()
    legend = plt.legend(loc='upper right')
    legend.get_frame().set_alpha(0.4)
    filename = f"scaling_laws/plots/{'_'.join([s.name for s in scaling_laws])}/{axis_dim}|{'|'.join(plot_dim[1:])}.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()


def download_from_neptune(project, tags, fixed, tags_negative=(), use_active_params=False, **_):
    neptune_cache = f"scaling_laws/tags_{'|'.join(tags)}_neptune_cache.csv"
    if os.path.exists(neptune_cache):
        print(f"Loading data from cache {neptune_cache}, remove it to download again...")
        table = pd.read_csv(neptune_cache, low_memory=False)
    else:
        print("Downloading data from Neptune...")
        # TODO this du8plicates when tags overlap
        table = pd.concat([project.fetch_runs_table(tag=tag).to_pandas() for tag in tags])
        table.rename(columns=lambda x: x.replace("/", "_"), inplace=True)
        table.to_csv(neptune_cache, index=False)
    table = [row for _, row in table.iterrows() if all(tag not in tags_negative for tag in row["sys_tags"].split(','))]
    table = [TrainRun(**row, fixed=fixed, use_active_params=use_active_params) for row in table]
    proper = np.average([t.finished for t in table])
    runs = [t for t in table if t.finished]
    print(f"{len(runs)} ({proper*100:.2f}%) of runs finished properly")
    return runs


def one_scaling(runs, fixed, use_active_params=False, **params):
    scaling_law = ScalingLaw(runs=runs, fixed=fixed, use_active_params=use_active_params, **params)
    _ = scaling_law.optimize()
    print(f"Final {scaling_law.name} scaling law approximation RMSE: {scaling_law()[1]}")
    scaling_law.present_values_as_chinchila()
    return scaling_law


def present_v(v):
    if 1e6 <= v < 1e9:
        return f"{v/1e6:.2f}M"
    elif 1e9 <= v < 1e12:
        return f"{v/1e9:.2f}B"
    elif 1e12 <= v < 1e15:
        return f"{v/1e12:.2f}T"
    elif v >= 1e15:
        return f"{v:.5e}"
    elif isinstance(v, float):
        return f"{v:.6}"
    else:
        return f"{v}"


def resolve_interactive(scaling_law):
    print("\n")
    print("Interactive params resolving.")
    print(f"Possible params: {scaling_law.params_set | {'granularity', 'flops', 'dmodel', 'n_blocks', 'loss'}}, most reasonable subsets should work.")
    print("Example jsons: \'{\"granularity\":1, \"flops\":1e20}\',  \'{\"n_params\":1e12}\',  \'{\"dmodel\":512}\'")
    text = "Enter proper json to resolve params, or 'stop' to stop: "
    while (prompt := input(text)) != 'stop':
        try:
            params = json.loads(prompt)
            final_params = scaling_law.resolve_params(**params)
            params_str = "\n".join([f"\t{k} = {present_v(v)}"for k, v in final_params.items()])
            print(f"Params: \n{params_str}")
        except Exception as e:
            print(e)
    return text


def resolve_prompts(scaling_law, prompts, **_):
    return {name: scaling_law.resolve_params(**values) for name, values in prompts.items()}


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    return {k: [d[k] for d in list_of_dicts] for k in list_of_dicts[0].keys()}


def prompt_intervals(prompts_out, **_):
    print("")
    promnt_values = list_of_dicts_to_dict_of_lists(prompts_out)
    for prompt_name, all_results in promnt_values.items():
        results_dict = list_of_dicts_to_dict_of_lists(all_results)
        print(f"{prompt_name}: ",end=" ")
        for name in sorted(results_dict.keys()):
            values = results_dict[name]
            p1 = np.percentile(values, 10)
            p2 = np.percentile(values, 90)
            if p1 == p2:
                print(f"{name}={present_v(p1)}", end=" ")
            else:
                print(f"{name}=[{present_v(p1)}, {present_v(p2)}]", end=" ")
        print("")


def compute_scaling_laws(project_name, scalings, plot_dims, config, repeat=1, **params):
    project = neptune_connect(project_name)
    prompts_out = []

    all_runs = [download_from_neptune(project, **s_config, **config) for s_config in scalings]
    for r in range(repeat):
        print(f"Repeat {r+1}/{repeat}")
        scaling_laws = [one_scaling(project=project, runs=runs, **s_config, **config)
                        for runs, s_config in zip(all_runs, scalings)]

        if plot_dims is not None:
            for plot_dim in plot_dims:
                plot_params(scaling_laws, plot_dim, **params)

        for scaling_law in scaling_laws:
            if scaling_law.resolve_interactive:
                prompts_out.append(resolve_prompts(scaling_law=scaling_law, **config))

    prompt_intervals(prompts_out)

    for scaling_law in scaling_laws:
        if scaling_law.resolve_interactive:
            resolve_interactive(scaling_law)


def run_from_config():
    config = read_yaml_file()
    print(config)
    compute_scaling_laws(**config, config=config)


if __name__ == "__main__":
    run_from_config()
