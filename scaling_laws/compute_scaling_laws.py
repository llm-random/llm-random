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
from numpy import log
from scipy.optimize import minimize_scalar


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
    table = [TrainRun(**row) for _, row in table.iterrows()]
    proper = np.average([t.finished for t in table])
    print(f"{proper*100:.2f}% of runs finished properly")
    return [t for t in table if t.finished]


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
        loss_interval_1000,
        args_granularity,
        args_expansion_rate,
        args_dmodel,
        args_n_steps,
        args_n_blocks,
        step,
        sys_state,
        **_,
    ):
        self.loss = loss_interval_1000
        self.granularity = args_granularity
        self.expansion_rate = args_expansion_rate
        self.dmodel = args_dmodel
        self.n_steps = args_n_steps
        self.n_blocks = args_n_blocks
        self.n_params = calculate_params(**self.dict())
        self.flops = calculate_flops(**self.dict())
        self.finished = sys_state == 'Inactive' and np.isfinite(self.loss) and step == self.n_steps

    def dict(self):
        return self.__dict__

    def __repr__(self):
        return f"({self.dict()})"


def calculate_flops(n_blocks, n_steps, **params):
    return calculate_block_flops(**params) * n_blocks * n_steps


def calculate_block_flops(dmodel, expansion_rate, granularity, **_):
    ff = dmodel ** 2 * 8
    router = dmodel * expansion_rate * granularity * 6
    einsum = 0  # current assumption
    attn = 0  # current assumption
    return ff + einsum + router + attn


def calculate_params(dmodel, expansion_rate, n_blocks, **_):
    # assume no params in routing and embeddings
    return dmodel**2 * (expansion_rate + 4) * n_blocks


dmodel_const = 64


def calculate_model_params_from_laws(expansion_rate, n_params, **_):
    params_const = n_params / (expansion_rate + 4)
    # TODO it's stupid but our configs kinda follow this
    # TODO check if this assumtion is aligned with standard scaling laws assumtion that flops ~ params*iter
    # we assume dmodel = dmodel_const * n_blocks
    # params_const = (n_blocks * (n_blocks*dmodel_const)**2) = n_blocks**3 * 64**2
    n_blocks = (params_const / dmodel_const**2)**(1/3)
    dmodel = dmodel_const * n_blocks
    return dict(dmodel=dmodel, n_blocks=n_blocks)


def calculate_n_steps_from_flops(flops, **params):
    model_params = calculate_model_params_from_laws(**params)
    dmodel, n_blocks = model_params['dmodel'], model_params['n_blocks']
    iter_flops = calculate_block_flops(dmodel=dmodel, **params) * n_blocks
    n_steps = flops / iter_flops
    new_params = dict(n_steps=n_steps, **model_params)
    assert np.isclose(calculate_flops(**params, **new_params), flops)
    return new_params


def calculate_n_params_from_flops(flops, n_steps, expansion_rate, granularity, **params):
    # F = n_steps * n_blocks * dmodel * (8*d + eg6)
    # 8d³+6egd - F/64/n = 0
    a = 8
    b = 6*granularity*expansion_rate
    c = 0
    d = - flops * dmodel_const / n_steps
    roots = np.roots([a, b, c, d])
    dmodel = np.real(roots[np.isreal(roots) & (roots > 0)][0])

    model_params = dict(dmodel=dmodel, n_blocks=dmodel/dmodel_const,
                        n_steps=n_steps, expansion_rate=expansion_rate,
                        granularity=granularity)
    n_params = calculate_params(**model_params, **params)
    new_params = dict(n_params=n_params, **model_params)
    assert np.isclose(calculate_flops(**params, **new_params), flops)
    return new_params


def calculate_n_params_and_steps_from_flops(flops, expansion_rate, granularity, scaling_laws, **params):
    x_p = scaling_laws.get_param_for("n_params")
    x_n = scaling_laws.get_param_for("n_steps")
    x_g = scaling_laws.get_param_for("granularity")
    c_p = scaling_laws.get_param_for("n_params", "granularity")
    c_n = scaling_laws.get_param_for("n_steps", "granularity")
    L = scaling_laws.L0.detach().item()
    g = granularity
    f = flops
    d_c = dmodel_const
    z = expansion_rate

    # derivative calculation
    # https://www.wolframalpha.com/input?i2d=true&i=%22D%5B%5C%2840%29Subscript%5Bx%2Cp%5D%2BSubscript%5Bc%2Cp%5DLog%5Bg%5D%5C%2841%29Log%5BDivide%5BPower%5Bd%2C3%5D%5C%2840%29z%2B4%5C%2841%29%2CSubscript%5Bd%2Cc%5D%5D%5D+%2B+Subscript%5Bx%2Cg%5DLog%5Bg%5D%2Cd%5D+%2B+%5C%2840%29Subscript%5Bx%2Cn%5D%2BSubscript%5Bc%2Cn%5DLog%5Bg%5D%5C%2841%29Log%5BDivide%5BSubscript%5Bd%2Cc%5Df%2CPower%5Bd%2C2%5D%5C%2840%298d+%2B+6z*g%5C%2841%29%5D%5D%22
    # solve for d
    p_fun = lambda d: calculate_params(dmodel=d, n_blocks=d/dmodel_const, expansion_rate=expansion_rate)
    n_fun = lambda d: dmodel_const*flops / (d**2*(8*d+6*granularity*expansion_rate))
    n_fun_slow = lambda d: calculate_n_steps_from_flops(flops=flops, n_params=p_fun(d), expansion_rate=expansion_rate, granularity=granularity)['n_steps']

    L_fun = lambda d: L + x_p*log(p_fun(d)) + x_n*log(n_fun(d)) + x_g*log(g) + \
        c_n*log(g)*log(n_fun(d)) + c_p*log(g)*log(p_fun(d))

    fun2 = lambda d: scaling_laws.expected_logloss(granularity=granularity, n_params=p_fun(d), n_steps=n_fun(d)).detach().item()
    function = lambda d: L_fun(d)
    function_der = lambda d: (
        (c_n*log(g) + x_n)*log((f*d_c)/(d**2*(8*d + 6*g*z))) +
        (3*(c_p*log(g) + x_p))/d
    )**2

    d_vals = [2**i for i in range(-1, 20)]
    assert np.allclose([n_fun(d) for d in d_vals], [n_fun_slow(d) for d in d_vals])
    assert np.allclose([L_fun(d) for d in d_vals], [fun2(d) for d in d_vals])

    # L_fun and fun2 are monotonic functions, this means current scaling laws
    # have some issue - increasing d_model with constant FLOPS always decreases loss
    # and this does not make sense - we should be able to find a point where loss is minimal
    # which balances num_steps and d_model

    res = minimize_scalar(function)
    res_der = minimize_scalar(function_der)
    res_raw = minimize_scalar(fun2)

    dmodel = res.x
    dmodel_der = res_der.x
    dmodel_raw = res_raw.x

    n_blocks = dmodel/dmodel_const
    n_params = p_fun(dmodel)
    n_steps = n_fun(dmodel)
    new_params = dict(n_params=n_params, n_steps=n_steps, flops=flops, dmodel=dmodel, n_blocks=n_blocks,
                      granularity=granularity, expansion_rate=expansion_rate, **params)
    assert np.isclose(calculate_flops(**new_params), flops)
    return new_params


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
        self.params_set = set(chain(*(set(p.names) for p in self.power_laws)))
        self.fixed_params = dict(expansion_rate=64)  # TODO auto check from runs and params_set

    def get_param_for(self, *names):
        for p in self.power_laws:
            if set(p.names) == set(names):
                return p.p.detach().item()
        return 0

    def expected_logloss(self, **params):
        return self.L0 + sum([p(**params) for p in self.power_laws])

    def __repr__(self):
        return (
            f"Scaling \"{self.name}\" {' + '.join([str(p) for p in self.power_laws])}"
            f" + {self.L0.item():.3}"
        )

    def forward(self):
        return sum(
            [
                (self.expected_logloss(**run.dict()) - math.log(run.loss)) ** 2
                for run in self.runs
            ]
        )

    def resolve_params(self, **params):
        lacking = [k for k in self.params_set if k not in params]
        if len(lacking) == 0:
            pass
        elif len(lacking) == 1 and lacking[0] == "n_steps" and "flops" in params:
            params.update(calculate_n_steps_from_flops(**params, **self.fixed_params))
        elif len(lacking) == 1 and lacking[0] == "n_params" and "flops" in params:
            params.update(calculate_n_params_from_flops(**params, **self.fixed_params))
        elif len(lacking) == 2 and "n_params" in lacking and "n_steps" in lacking and "flops" in params:
            params.update(calculate_n_params_and_steps_from_flops(**params,scaling_laws=self, **self.fixed_params))
        else:
            raise Exception(f"Missing params {lacking} that canno be resolved")
        return self.expected_logloss(**params).detach().numpy()

    def optimize_params_for_flops(self, flops):
        params = self.params_set
        optimization_goal = self.expected_logloss(**params)
        constaint = self.expected_flops(**params) - flops


def optimize(model, num_steps, early_stop=1000):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.train()
    min_loss = (np.inf, -1)
    with trange(num_steps) as iterator:
        for i in iterator:
            optimizer.zero_grad()
            loss = model()
            iterator.set_description(f"Optimizing, error={loss:.4} {str(model)}")
            loss.backward()
            optimizer.step()
            min_loss = min(min_loss, (loss.item(), i))
            if i - min_loss[1] > early_stop:
                print(f"Early stop at {i}")
                break


def plot_params(scaling_law, plot_dim, plot_points=1000):
    axis_dim = plot_dim[0]
    A = np.array([math.log10(r.dict()[axis_dim]) for r in scaling_law.runs])
    B = np.array([math.log(r.loss) for r in scaling_law.runs])
    A_values = np.linspace(A.min(), A.max() + 0.5*(A.max() - A.min()), plot_points)
    plot_minimal = axis_dim == "flops"

    group_dims = sorted(list(scaling_law.params_set - set(plot_dim)))
    dicts = [params.dict() for params in scaling_law.runs]
    group_values = [tuple([params[d] for d in group_dims]) for params in dicts]
    groups = unique_values_with_indices(group_values)
    colors = cm.rainbow(np.linspace(0, 1, len(groups)))
    plt.figure(dpi=200)

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
    plt.show()


def one_scaling(project, tags, power_laws, name, plot_dims, num_opt_steps, **_):
    runs = download_batch_sizes_from_neptune(project, tags)
    scaling_law = ScalingLaw(name, runs, power_laws)
    if os.path.exists("model.ckpt"):
        scaling_law.load_state_dict(torch.load("model.ckpt"))
    else:
        optimize(scaling_law, num_steps=num_opt_steps)
        torch.save(scaling_law.state_dict(), "model.ckpt")
    for plot_dim in plot_dims:
        plot_params(scaling_law, plot_dim)


def compute_scaling_laws():
    config = read_yaml_file()
    print(config)
    project = neptune_connect(config["project_name"])
    for scaling_config in config["scalings"]:
        one_scaling(project=project, **scaling_config, **config)


if __name__ == "__main__":
    compute_scaling_laws()
