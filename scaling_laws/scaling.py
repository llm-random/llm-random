from itertools import chain
from functools import partial

import numpy as np
import torch
from torch import nn as nn
from torch.optim import lr_scheduler as lr_scheduler
from tqdm import trange
import os
from scipy.optimize import minimize
import math

from scaling_laws.calculate_params import *


def init(name=None, eps=1e-5, dict=None, trainable=True, log=True):
    if dict and name and name in dict:
        eps = dict[name]
        if log: eps = np.log(eps)
    v = tensor(eps)
    return nn.Parameter(v) if trainable else v


def tensor(value=0.):
    return torch.tensor(value, dtype=torch.float64)


# that's log(a*param**b + c), assuming a and c are given in log
def logsumexp_poly(a, b, c, param):
    if param is None:
        return c
    c1 = b * param + a
    return c1 if c is None else torch.logsumexp(torch.stack([c1, c]), dim=0)


class PowerLaw(nn.Module):
    def __init__(self, names, eps=1e-5, exp_inter=False, poly_inter=True, init_dict=None, **_):
        super(PowerLaw, self).__init__()
        self.exp_inter = exp_inter
        self.poly_inter = poly_inter
        self.names = names
        init_dict = None if init_dict is None else init_dict.get(self.names[0])
        self.p = init(eps=-eps, dict=init_dict, name='p', log=False)
        init_f = partial(init, eps=eps, dict=init_dict)
        if len(names) == 2:
            self.a, self.b = init_f('a'), init_f('b', log=False)
            self.c = init_f('c') if self.poly_inter else tensor()
            self.name, self.condition = self.names
            if exp_inter:
                self.i = init_f('i')
        elif len(names) == 1:
            self.a, self.b, self.c = tensor(), tensor(), init_f('c')
            self.name = self.names[0]

    def get_tensors(self, params):
        for name in self.names:
            param = params.get(name, None)
            if param is None:
                raise Exception(f"No {name} param found in {params})")
            yield torch.log(tensor(param))
        if len(self.names) == 1:
            yield None

    def forward(self, **params):
        param, condition = self.get_tensors(params)
        log_multiplier = self.get_logmultiplier(condition)
        scaling = logsumexp_poly(log_multiplier, self.p, None, param)
        if self.exp_inter:
            scaling += self.i*param*condition
        return scaling

    def get_logmultiplier(self, condition_val=None):
        return logsumexp_poly(self.a, self.b, self.c, condition_val)

    def get_nocond_params(self, condition_val):
        a = torch.exp(self.get_logmultiplier(condition_val))
        b = self.p
        return a, b

    def repr_no_cond(self, condition_val):
        if self.exp_inter:
            return "Cant calculate"  # TODO?
        a, b = self.get_nocond_params(math.log(condition_val) if len(self.names) == 2 else None)
        return f"{a.item():6.4f}*{self.name}**{b.item():6.4f}"

    def __repr__(self):
        p, a, b, c = self.get_scaled_params()
        if len(self.names) == 2:
            text = f"{a:6.4f}*{self.condition}**{b:6.4f}"
            text = f"({text}+{c:6.4f})" if self.poly_inter else text
            text += f"*{self.name}**{p:6.4f}"
            if self.exp_inter:
                text += f"*{self.name}**({self.i.item():6.4f}*ln({self.condition}))"
            return text
        elif len(self.names) == 1:
            return f"{c:6.4f}*{self.name}**{p:6.4f}"
        else:
            raise Exception("Not implemented")

    def get_scaled_params(self):
        ret = [self.p, torch.exp(self.a), self.b, torch.exp(self.c)]
        return [x.detach().item() for x in ret]


class ScalingLaw(nn.Module):
    def __init__(self, name, runs, power_laws, fixed, cmap, flops_min, flops_max, eps=1e-5, num_opt_steps=None,
                 train_scaling_bias=True, lr=None, remove_random=0.0, init_dict=None, run_fit=True,
                 final_lr_fr=None, use_scipy=False, load_model=False, opt_log_loss=False, weight_decay=0, huber_delta=0.1,
                 resolve_interactive=False, use_active_params=False, **params):
        super().__init__()
        self.name = name
        self.checkpoint_name = f"scaling_laws/checkpoints/{name}_model.ckpt"
        self.L0 = init(eps=eps, dict=init_dict, name='L0', trainable=train_scaling_bias)
        self.loss = torch.nn.HuberLoss(delta=huber_delta) if huber_delta > 0 else torch.nn.MSELoss()
        self.cmap = cmap
        self.power_laws = nn.ModuleList([PowerLaw(names, eps, **params, init_dict=init_dict) for names in power_laws])
        self.params_set = set(chain(*(set(p.names) for p in self.power_laws)))
        self.fixed_params = fixed
        self.run_fit = run_fit
        self.num_opt_steps = num_opt_steps
        self.load_model = load_model
        self.lr = lr
        self.final_lr_fr = final_lr_fr
        self.use_scipy = use_scipy
        self.opt_log_loss = opt_log_loss
        self.weight_decay = weight_decay
        self.resolve_interactive = resolve_interactive
        self.conf_params = dict(use_active_params=use_active_params)
        self.flops_range = (float(flops_min), float(flops_max))
        self.flops_search_range_log = (math.log(self.flops_range[0]) - 10, math.log(self.flops_range[1]) + 10)
        self.flops_range_margin = (math.exp(self.flops_search_range_log[0]) * 1.1, math.exp(self.flops_search_range_log[1]) * 0.9)
        self.granularity_range = [2**g_i for g_i in range(0, 9)]

        self.all_runs = runs
        self.n_runs = len(self.all_runs )
        self.mask = np.array([False]*self.n_runs)
        self.train_ind = np.random.choice(np.arange(self.n_runs), size=int((1-remove_random)*self.n_runs))
        self.mask[self.train_ind] = True
        self.runs = [self.all_runs[i] for i in self.train_ind] if remove_random > 0 else self.all_runs
        pass

    def present_values_as_chinchila(self):
        print(f"{str(self)}")
        if 'granularity' not in self.fixed_params:
            for g in self.granularity_range:
                constant = self.get_constant() + sum([torch.exp(p(granularity=g)) for p in self.power_laws if p.names[0] == 'granularity'])
                print(
                    f"For constant granularity={g} Scaling \"{self.name}\" "
                    f"{' + '.join([p.repr_no_cond(g) for p in self.power_laws if p.names[0] != 'granularity'])}"
                    f" + {constant:6.3f}"
                )

    def get_param_for(self, *names):
        for p in self.power_laws:
            if set(p.names) == set(names):
                return p.get_scaled_params()
        return 0

    def get_constant(self):
        return torch.exp(self.L0).item()

    def formula(self, **params):
        parts = [self.L0] + [p(**params) for p in self.power_laws]
        return torch.logsumexp(torch.stack(parts), dim=0)

    def expected_loss(self, **params):
        return torch.exp(self.formula(**params))

    def expected_logloss(self, **params):
        return self.formula(**params)

    def __repr__(self):
        return (
            f"Scaling \"{self.name}\" {' + '.join([str(p) for p in self.power_laws])}"
            f" + {self.get_constant():5.3f}"
        )

    def forward(self):
        y = tensor([math.log(run.loss) if self.opt_log_loss else run.loss for run in self.runs])
        loss_fun = self.expected_logloss if self.opt_log_loss else self.expected_loss
        y_pred = torch.stack([loss_fun(**run.dict()) for run in self.runs])
        error = torch.exp(y_pred) - torch.exp(y) if self.opt_log_loss else y_pred - y
        rmse = (torch.mean(error**2))**.5
        weight_decay = sum(torch.norm(p)**2 for p in self.parameters())
        loss = self.loss(y_pred, y)
        loss += weight_decay*self.weight_decay
        return loss, rmse

    def find_opt_granularity(self, **params):
        params = [self.resolve_params(**params, granularity=g) for g in self.granularity_range]
        best = np.argmin(np.stack([p["loss"] for p in params]))
        return params[best]

    def resolve_params(self, **params):
        params.update(self.fixed_params)
        lacking = self.params_set - set(params.keys())
        params = self.resolve_model_size(lacking, params)

        if "loss" in params:
            params.update(calculate_compute_opt_for_loss(**params, scaling_laws=self))
        elif lacking == set():
            pass
        elif lacking == {"n_steps"} and "flops" in params:
            params.update(calculate_n_steps_from_flops(**self.conf_params, **params))
        elif lacking == {"n_params"} and "flops" in params:
            params.update(calculate_n_params_from_flops(**self.conf_params, **params))
        elif lacking == {"n_params", "n_steps"} and "flops" in params:
            params.update(calculate_n_params_and_steps_from_flops(**self.conf_params, **params, scaling_laws=self))
        elif "n_steps" in params and "n_params" in lacking:
            params.update(calculate_compute_opt_for_steps(**params, scaling_laws=self))
        elif "n_params" in params and "n_steps" in lacking:
            params.update(calculate_compute_opt_for_params(**params, scaling_laws=self))
        elif "granularity" in lacking and "flops" in params:
            params.update(self.find_opt_granularity(**params))
        else:
            raise Exception(f"Missing params {lacking} that cannot be resolved")

        self.update_other_hyperparams(params)
        return params

    def resolve_model_size(self, lacking, params):
        params = {k: (float(v) if isinstance(v, str) else v) for k, v in params.items()}
        if "n_params_active" in params:
            params.update(calculate_model_params_from_active_params(**params))
            del params["n_params_active"]
        if "dmodel" in params and "n_blocks" not in params:
            params["n_blocks"] = params["dmodel"] // dmodel_const
        if "dmodel" not in params and "n_blocks" in params:
            params["dmodel"] = params["n_blocks"] * dmodel_const

        if "n_params" in lacking and "dmodel" in params and "n_blocks" in params:
            params["n_params"] = calculate_params(**self.conf_params, **params)
            lacking -= {"n_params"}
        if "dmodel" in params:
            del params["dmodel"]
        if "n_blocks" in params:
            del params["n_blocks"]
        return params

    def update_other_hyperparams(self, params):
        # compute optimal already calculated here
        params.update(calculate_model_params_from_laws(**self.conf_params, **params))
        params.update(n_params_total=calculate_total_params(**params),
                      n_params_active=calculate_active_params(**params),
                      flops=calculate_flops(**self.conf_params, **params),
                      n_opt_steps=get_n_opt_step_from_tokens(params["n_steps"]))
        if params["flops"] < self.flops_range_margin[0] or params["flops"] > self.flops_range_margin[1]:
            raise Exception(f"Flops {params['flops']} out of range {self.flops_range}")

        logloss = self.expected_logloss(**params).detach().numpy()
        params.update(logloss=logloss, loss=np.exp(logloss))

    def optimize(self):
        if not self.run_fit:
            return self()[1].item()

        if self.load_model and os.path.exists(self.checkpoint_name):
            print(f"Loading model {self.name} from {self.checkpoint_name}")
            self.load_state_dict(torch.load(self.checkpoint_name))
            return
        if self.use_scipy:
            self.optimize_scipy()
        else:
            self.optimize_pytorch()
        os.makedirs(os.path.dirname(self.checkpoint_name), exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_name)
        return self()[1].item()

    def optimize_pytorch(self, early_stop=10000):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=self.final_lr_fr, total_iters=self.num_opt_steps)
        self.train()
        min_eval = (np.inf, -1)
        with trange(self.num_opt_steps) as iterator:
            for i in iterator:
                optimizer.zero_grad()
                loss, rmse = self()
                loss.backward()
                iterator.set_description(f"Optimizing, error={rmse:7.5f} (loss={loss:7.5f}) {str(self)}")
                optimizer.step()
                scheduler.step()
                min_eval = min(min_eval, (rmse.item(), i))
                if i - min_eval[1] > early_stop:
                    print(f"Early stop at {i}")
                    break

    def closure(self):
        self.zero_grad()
        loss, rmse = self()
        loss.backward()
        return loss.item()

    def to_grads(self):
        self.closure()
        return torch.nn.utils.parameters_to_vector([x.grad for x in self.parameters()]).detach().numpy()

    def to_params(self):
        return torch.nn.utils.parameters_to_vector(self.parameters()).detach().numpy()

    def from_params(self, vec):
        torch.nn.utils.vector_to_parameters(torch.from_numpy(vec.astype(np.float32)), self.parameters())
        return self

    def optimize_scipy(self):
        print(f"{self.name} optimization with scipy... ", end='')
        x0 = self.to_params()
        result = minimize(lambda x: self.from_params(x).closure(), x0, method='BFGS', options={'disp': True},
                          jac=lambda x: self.from_params(x).to_grads(),
                          callback=lambda x: self.from_params(x))
        self.from_params(result.x)
        print("finished")
