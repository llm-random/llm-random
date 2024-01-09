from itertools import chain

import numpy as np
import torch
from torch import nn as nn
from torch.optim import lr_scheduler as lr_scheduler
from tqdm import trange
import os
from scipy.optimize import minimize
import math

from scaling_laws.calculate_params import calculate_n_steps_from_flops, calculate_n_params_from_flops, \
    calculate_n_params_and_steps_from_flops


def init(eps):
    return nn.Parameter(torch.tensor(eps))


# that's log(a*param**b + c), assuming a and c are given in log
def logsumexp_poly(a, b, c, param):
    if param is None:
        return c
    c1 = b * param + a
    return c1 if c is None else torch.logsumexp(torch.stack([c1, c]), dim=0)


class PowerLaw(nn.Module):
    def __init__(self, names, eps=1e-5, exp_inter=False, poly_inter=True, **_):
        super(PowerLaw, self).__init__()
        self.exp_inter = exp_inter
        self.poly_inter = poly_inter
        self.p = init(-eps)
        self.names = names
        if len(names) == 2:
            self.a, self.b = init(eps), init(eps)
            self.c = init(eps) if self.poly_inter else torch.tensor(0)
            self.name, self.condition = self.names
            if exp_inter:
                self.i = init(eps)
        elif len(names) == 1:
            self.a, self.b, self.c = torch.tensor(0), torch.tensor(0), init(eps)
            self.name = self.names[0]

    def get_tensors(self, params):
        for name in self.names:
            param = params.get(name, None)
            if param is None:
                raise Exception(f"No {name} param found in {params})")
            yield torch.log(torch.tensor(param))
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

    def get_nocond_params(self):
        return

    def repr_no_cond(self, condition_val):
        if self.exp_inter:
            return "Cant calculate"  # TODO?
        a = torch.exp(self.get_logmultiplier(condition_val))
        return f"{a:6.4f}*{self.name}**{self.p.item():6.4f}"

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
    def __init__(self, name, runs, power_laws, fixed, cmap, eps=1e-5, num_opt_steps=None, lr=None,
                 final_lr_fr=None, use_scipy=False, load_model=False, opt_log_loss=False, weight_decay=0, huber_delta=0.1, **params):
        super().__init__()
        self.runs = runs
        self.name = name
        self.checkpoint_name = f"scaling_laws/checkpoints/{name}_model.ckpt"
        self.L0 = init(eps)
        self.loss = torch.nn.HuberLoss(delta=huber_delta) if huber_delta > 0 else torch.nn.MSELoss()
        self.cmap = cmap
        self.power_laws = nn.ModuleList([PowerLaw(names, eps, **params) for names in power_laws])
        self.params_set = set(chain(*(set(p.names) for p in self.power_laws)))
        self.fixed_params = fixed
        self.num_opt_steps = num_opt_steps
        self.load_model = load_model
        self.lr = lr
        self.final_lr_fr = final_lr_fr
        self.use_scipy = use_scipy
        self.opt_log_loss = opt_log_loss
        self.weight_decay = weight_decay

    def present_values_as_chinchila(self):
        print(f"{str(self)}")
        if 'granularity' not in self.fixed_params:
            for g in [2**g_i for g_i in range(0, 8)]:
                constant = self.L0.item() + sum([p(granularity=g) for p in self.power_laws if p.names[0] == 'granularity'])
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
            f" + {torch.exp(self.L0).item():5.3f}"
        )

    def forward(self):
        y = torch.tensor([math.log(run.loss) if self.opt_log_loss else run.loss for run in self.runs])
        loss_fun = self.expected_logloss if self.opt_log_loss else self.expected_loss
        y_pred = torch.stack([loss_fun(**run.dict()) for run in self.runs])
        error =  torch.exp(y_pred) - torch.exp(y) if self.opt_log_loss else y_pred - y
        rmse = (torch.mean(error**2))**.5
        weight_decay = sum(torch.norm(p) for p in self.parameters())
        loss = self.loss(y_pred, y)
        loss += weight_decay*self.weight_decay
        return loss, rmse

    def resolve_params(self, **params):
        lacking = [k for k in self.params_set if k not in params]
        params.update(self.fixed_params)
        if len(lacking) == 0:
            pass
        elif len(lacking) == 1 and lacking[0] == "n_steps" and "flops" in params:
            params.update(calculate_n_steps_from_flops(**params))
        elif len(lacking) == 1 and lacking[0] == "n_params" and "flops" in params:
            params.update(calculate_n_params_from_flops(**params))
        elif len(lacking) == 2 and "n_params" in lacking and "n_steps" in lacking and "flops" in params:
            params.update(calculate_n_params_and_steps_from_flops(**params, scaling_laws=self))
        else:
            raise Exception(f"Missing params {lacking} that cannot be resolved")
        return self.expected_logloss(**params).detach().numpy(), params

    def optimize(self):
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
        loss *= 1  # for >1, this was a trick to help with numerical stability
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
        print(result)
        self.from_params(result.x)
        print("finished")
