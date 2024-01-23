import numpy as np
from scipy.optimize import minimize_scalar

from scaling_laws.utils import binary_search

dmodel_const = 64
ff_const = 4
router_const = 4
batch_size = 256
seq_len = 256
grad_accum = 8


class TrainRun:
    def __init__(
        self,
        loss_interval_1000,
        args_dmodel,
        args_n_steps,
        args_n_blocks,
        args_final_lr_step,
        step,
        sys_state,
        fixed,
        args_granularity=0,
        args_expansion_rate=1,
        use_active_params=False,
        **_,
    ):
        self.loss = loss_interval_1000
        self.granularity = args_granularity
        self.expansion_rate = args_expansion_rate
        self.dmodel = args_dmodel
        self.use_active = use_active_params
        self.n_opt_steps = args_n_steps
        self.n_steps = self.n_opt_steps * batch_size * seq_len * grad_accum
        self.n_tokens = self.n_steps
        self.n_blocks = args_n_blocks
        self.n_params_total = calculate_total_params(**self.dict())
        self.n_params_active = calculate_active_params(**self.dict())
        self.n_params = calculate_params(**self.dict())
        self.flops = calculate_flops(**self.dict())
        self.finished = sys_state == 'Inactive' and np.isfinite(self.loss) and \
                        step == self.n_opt_steps == args_final_lr_step and \
                        all([getattr(self, k) == v for k, v in fixed.items()]) and \
                        (self.granularity < 32)  # and \
#                        (self.n_params_total > 200000000 or self.n_opt_steps > 10000 or self.granularity > 0)  and \
#                        (self.n_params_total > 200000000 or self.n_opt_steps > 10000 or self.granularity == 0)  # filter for MoE, previous is for dense
                            # TODO: remove this hack and ^this, filer small axp and high granularity
        # 6291456 + 10
    def dict(self):
        return self.__dict__

    def __repr__(self):
        return f"({self.dict()})"


def calculate_flops(n_blocks, n_steps, **params):
    return calculate_block_flops(**params) * n_blocks * n_steps


def calculate_block_flops(dmodel, expansion_rate, granularity, **_):
    ff = dmodel ** 2 * ff_const
    router = dmodel * expansion_rate * granularity * router_const
    einsum = 0  # current assumption
    attn = 0  # current assumption
    return ff + einsum + router + attn


def calculate_params(use_active=False, **params):
    return calculate_active_params(**params) if use_active else calculate_total_params(**params)


def calculate_total_params(dmodel, expansion_rate, n_blocks, **_):
    # assume no params in routing and embeddings
    return dmodel**2 * 4*(2*expansion_rate + 1) * n_blocks


def calculate_active_params(dmodel, n_blocks, **_):
    return calculate_total_params(dmodel, 1, n_blocks)


def calculate_model_params_from_laws(expansion_rate, n_params, use_active=False, **_):
    expansion_rate = 1 if use_active else expansion_rate
    params_const = n_params / (4*(2*expansion_rate + 1))
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


def calculate_n_steps_from_dmodel(flops, dmodel, expansion_rate, granularity):
    divisor = (dmodel**2*(ff_const*dmodel+router_const*granularity*expansion_rate))
    return dmodel_const*flops / divisor if divisor != 0 else np.inf


def calculate_n_params_from_flops(flops, n_steps, expansion_rate, granularity, **params):
    # F = n_steps * n_blocks * dmodel * (8*d + eg6)
    # 8d³+6egd - F*64/n = 0
    a = ff_const
    b = router_const*granularity*expansion_rate
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


def calculate_n_params_and_steps_from_flops(flops, expansion_rate, granularity, scaling_laws, use_active=False, **params):
    p_fun = lambda d: calculate_params(dmodel=d, n_blocks=d/dmodel_const, expansion_rate=expansion_rate, use_active=use_active)
    n_fun = lambda d: calculate_n_steps_from_dmodel(flops=flops, dmodel=d, expansion_rate=expansion_rate, granularity=granularity)
    n_fun_slow = lambda d: calculate_n_steps_from_flops(flops=flops, n_params=p_fun(d), expansion_rate=expansion_rate, granularity=granularity, use_active=use_active)['n_steps']
    fun2 = lambda d: min(scaling_laws.expected_loss(granularity=granularity, n_params=p_fun(d), n_steps=n_fun(d)).detach().item(), 1e40)
    d_vals = [2**i for i in range(-1, 20)]
    assert np.allclose([n_fun(d) for d in d_vals], [n_fun_slow(d) for d in d_vals])
    res_raw = minimize_scalar(fun2)  #, options=dict(disp=1))   , method='bounded', bounds=(1, 1e6))
    dmodel = res_raw.x

    n_blocks = dmodel/dmodel_const
    n_params = p_fun(dmodel)
    n_steps = n_fun(dmodel)
    new_params = dict(n_params=n_params, n_steps=n_steps, flops=flops, dmodel=dmodel, n_blocks=n_blocks,
                      granularity=granularity, expansion_rate=expansion_rate)
    assert np.isclose(calculate_flops(**new_params, **params), flops)
    return new_params


def calculate_compute_opt_params(n_steps, scaling_laws, **params):
    compute_opt_params_diff = lambda lflops: n_steps - calculate_n_params_and_steps_from_flops(flops=np.exp(lflops), scaling_laws=scaling_laws, **params)["n_steps"]
    flops = np.exp(binary_search(range=scaling_laws.flops_range_log, fun=compute_opt_params_diff))
    return calculate_n_params_from_flops(n_steps=n_steps, flops=flops, scaling_laws=scaling_laws, **params)


def calculate_compute_opt_steps(n_params, scaling_laws, **params):
    compute_opt_steps_diff = lambda lflops: n_params - calculate_n_params_and_steps_from_flops(flops=np.exp(lflops), scaling_laws=scaling_laws, **params)["n_params"]
    flops = np.exp(binary_search(range=scaling_laws.flops_range_log, fun=compute_opt_steps_diff))
    return calculate_n_steps_from_flops(n_params=n_params, flops=flops, scaling_laws=scaling_laws, **params)



#    x_p = scaling_laws.get_param_for("n_params")
#    x_n = scaling_laws.get_param_for("n_steps")
#    x_g = scaling_laws.get_param_for("granularity")
#    c_p = scaling_laws.get_param_for("n_params", "granularity")
#    c_n = scaling_laws.get_param_for("n_steps", "granularity")
#    L = scaling_laws.L0.detach().item()
#    g = granularity
#    f = flops
#    d_c = dmodel_const
#    z = expansion_rate

    # derivative calculation
    # https://www.wolframalpha.com/input?i2d=true&i=%22D%5B%5C%2840%29Subscript%5Bx%2Cp%5D%2BSubscript%5Bc%2Cp%5DLog%5Bg%5D%5C%2841%29Log%5BDivide%5BPower%5Bd%2C3%5D%5C%2840%29z%2B4%5C%2841%29%2CSubscript%5Bd%2Cc%5D%5D%5D+%2B+Subscript%5Bx%2Cg%5DLog%5Bg%5D%2Cd%5D+%2B+%5C%2840%29Subscript%5Bx%2Cn%5D%2BSubscript%5Bc%2Cn%5DLog%5Bg%5D%5C%2841%29Log%5BDivide%5BSubscript%5Bd%2Cc%5Df%2CPower%5Bd%2C2%5D%5C%2840%298d+%2B+6z*g%5C%2841%29%5D%5D%22
    # solve for d
#    L_fun = lambda d: L + x_p*log(p_fun(d)) + x_n*log(n_fun(d)) + x_g*log(g) + \
#        c_n*log(g)*log(n_fun(d)) + c_p*log(g)*log(p_fun(d))

#
#    function = lambda d: L_fun(d)
#    function_der = lambda d: (
#        (c_n*log(g) + x_n)*log((f*d_c)/(d**2*(8*d + 6*g*z))) +
#        (3*(c_p*log(g) + x_p))/d
#    )**2

#

#    assert np.allclose([L_fun(d) for d in d_vals], [fun2(d) for d in d_vals])

    # L_fun and fun2 are monotonic functions, this means current scaling laws
    # have some issue - increasing d_model with constant FLOPS always decreases loss
    # and this does not make sense - we should be able to find a point where loss is minimal
    # which balances num_steps and d_model

#    res = minimize_scalar(function)
#    res_der = minimize_scalar(function_der)

    #dmodel_der = res_der.x
    #dmodel_raw = res_raw.x