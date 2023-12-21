from itertools import chain

import numpy as np
import torch
from torch import nn as nn
from torch.optim import lr_scheduler as lr_scheduler
from tqdm import trange
import os

from scaling_laws.calculate_params import calculate_n_steps_from_flops, calculate_n_params_from_flops, \
    calculate_n_params_and_steps_from_flops


def init(eps):
    return nn.Parameter(torch.tensor(eps))


class PowerLaw(nn.Module):
    def __init__(self, names, eps=1e-5, use_chinchilla=True, with_interaction=True):
        super(PowerLaw, self).__init__()
        self.use_chinchilla = use_chinchilla
        self.with_interaction = with_interaction
        self.p = init(-eps)
        self.names = names
        if not self.use_chinchilla:
            self.name = "*".join([name if self.use_chinchilla else f"ln({name})" for name in names])
        elif len(names) == 2:
            self.a, self.b, self.c = map(init, [eps]*3)
            self.name, self.condition = self.names
            if with_interaction:
                self.i, self.pi = init(eps), init(eps)  # TODO, this is controversial
        elif len(names) == 1:
            self.a, self.b, self.c = 0, 0, init(eps)
            self.name = self.names[0]

    def get_tensors(self, params):
        for name in self.names:
            param = params.get(name, None)
            if param is None:
                raise Exception(f"No {name} param found in {params})")
            yield torch.tensor(param)
        if len(self.names) == 1 and self.use_chinchilla:
            yield None

    def forward(self, **params):
        if self.use_chinchilla:
            param, condition = self.get_tensors(params)
            if condition is None:
                return self.c * param ** self.p
            scaling = (self.a * condition ** self.b + self.c) * param ** self.p
            if self.with_interaction:
                scaling *= (torch.exp(self.i*(torch.log(condition)*torch.log(param))) + self.pi)
            return scaling

        params = [torch.log(x) for x in self.get_tensors(params)]
        return torch.prod(torch.stack(params)) * self.p

    def __repr__(self):
        if self.use_chinchilla and len(self.names) == 2:
            text = f"({self.a.item():.2}*{self.condition}**{self.b.item():.2} " \
                   f"+ {self.c.item():.2})*{self.name}**{self.p.item():.2}"
            if self.with_interaction:
                text += f"*(exp({self.i.item():.2}*ln({self.condition})*ln({self.name})) + {self.pi.item():.2})"
            return text
        elif self.use_chinchilla and len(self.names) == 1:
            return f"{self.c.item():.2}*{self.name}**{self.p.item():.2}"
        else:
            return f"{self.p.item():.2}*{self.name}"


class ScalingLaw(nn.Module):
    def __init__(self, name, runs, power_laws, fixed, cmap, use_chinchilla=True, eps=1e-5, **_):
        super().__init__()
        self.runs = runs
        self.name = name
        self.checkpoint_name = f"scaling_laws/checkpoints/{name}_model.ckpt"
        self.use_chinchilla = use_chinchilla
        self.L0 = init(eps)
        self.loss = torch.nn.HuberLoss(delta=0.01)
        self.cmap = cmap
        self.power_laws = nn.ModuleList([PowerLaw(names, eps, use_chinchilla) for names in power_laws])
        self.params_set = set(chain(*(set(p.names) for p in self.power_laws)))
        self.fixed_params = fixed

    def get_param_for(self, *names):
        for p in self.power_laws:
            if set(p.names) == set(names):
                if not self.use_chinchilla:
                    return p.p.detach().item()
                ret = [p.p, p.a, p.b, p.c]
                return [x.detach().item() if x != 0 else x for x in ret]
        return 0

    def expected_logloss(self, get_log=True, **params):
        val = self.L0 + sum([p(**params) for p in self.power_laws])
        return torch.log(val) if (self.use_chinchilla and get_log) else val  # TODO check if it's ok?

    def __repr__(self):
        return (
            f"Scaling \"{self.name}\" {' + '.join([str(p) for p in self.power_laws])}"
            f" + {self.L0.item():.3}"
        )

    def calc_loss(self, params, loss):
        y_pred = self.expected_logloss(**params, get_log=False)
        if self.use_chinchilla:
            y = torch.tensor(loss)
            eval_se = (y_pred - y)**2
            return self.loss(y_pred, y), eval_se
        else:
            y = torch.log(torch.tensor(loss))
            eval_se = (y_pred - y)**2
            return self.loss(y_pred, y), eval_se

    def forward(self):
        losses, se = zip(*[self.calc_loss(run.dict(), run.loss) for run in self.runs])
        rmse = (sum(se)/len(self.runs))**.5
        return sum(losses)/len(self.runs), rmse

    def resolve_params(self, **params):
        lacking = [k for k in self.params_set if k not in params]
        if len(lacking) == 0:
            pass
        elif len(lacking) == 1 and lacking[0] == "n_steps" and "flops" in params:
            params.update(calculate_n_steps_from_flops(**params, **self.fixed_params))
        elif len(lacking) == 1 and lacking[0] == "n_params" and "flops" in params:
            params.update(calculate_n_params_from_flops(**params, **self.fixed_params))
        elif len(lacking) == 2 and "n_params" in lacking and "n_steps" in lacking and "flops" in params:
            params.update(calculate_n_params_and_steps_from_flops(**params, scaling_laws=self, **self.fixed_params))
        else:
            raise Exception(f"Missing params {lacking} that cannot be resolved")
        return self.expected_logloss(**params).detach().numpy()

    def optimize(self, num_steps, early_stop=10000):
        if os.path.exists(self.checkpoint_name):
            self.load_state_dict(torch.load(self.checkpoint_name))
            return
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=num_steps)
        self.train()
        min_eval = (np.inf, -1)
        with trange(num_steps) as iterator:
            for i in iterator:
                optimizer.zero_grad()
                loss, rmse = self()
                iterator.set_description(f"Optimizing, error={rmse:.4} (loss={loss:.4}) {str(self)}")
                loss.backward()
                optimizer.step()
                scheduler.step()
                min_eval = min(min_eval, (rmse.item(), i))
                if i - min_eval[1] > early_stop:
                    print(f"Early stop at {i}")
                    break
        os.makedirs(os.path.dirname(self.checkpoint_name), exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_name)
        return rmse.item()
