import numpy as np
from scipy.optimize import minimize_scalar


dmodel_const = 64
ff_const = 4
router_const = 1
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
        args_granularity=1,
        args_expansion_rate=1,
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
        self.finished = sys_state == 'Inactive' and np.isfinite(self.loss) and \
                        step == self.n_steps == args_final_lr_step and \
                        all([getattr(self, k) == v for k, v in fixed.items()]) and \
                        (self.n_params > 20000000 or self.n_steps > 20000 or self.expansion_rate == 1) and \
                        (self.granularity < 32)  # TODO: remove this hack and ^this

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
    return (ff + einsum + router + attn) * batch_size * seq_len * grad_accum


def calculate_params(dmodel, expansion_rate, n_blocks, **_):
    # assume no params in routing and embeddings
    return dmodel**2 * (expansion_rate + 4) * n_blocks


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
    # 8d³+6egd - F*64/n = 0
    a = ff_const
    b = router_const*granularity*expansion_rate
    c = 0
    d = - flops * dmodel_const / (n_steps * batch_size * seq_len * grad_accum)
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
    n_fun = lambda d: dmodel_const*flops / (d**2*(ff_const*d+router_const*granularity*expansion_rate) * batch_size * seq_len * grad_accum)
    n_fun_slow = lambda d: calculate_n_steps_from_flops(flops=flops, n_params=p_fun(d), expansion_rate=expansion_rate, granularity=granularity)['n_steps']

#    L_fun = lambda d: L + x_p*log(p_fun(d)) + x_n*log(n_fun(d)) + x_g*log(g) + \
#        c_n*log(g)*log(n_fun(d)) + c_p*log(g)*log(p_fun(d))

    fun2 = lambda d: scaling_laws.expected_logloss(granularity=granularity, n_params=p_fun(d), n_steps=n_fun(d), get_log=False).detach().item()
#    function = lambda d: L_fun(d)
#    function_der = lambda d: (
#        (c_n*log(g) + x_n)*log((f*d_c)/(d**2*(8*d + 6*g*z))) +
#        (3*(c_p*log(g) + x_p))/d
#    )**2

    d_vals = [2**i for i in range(-1, 20)]
    assert np.allclose([n_fun(d) for d in d_vals], [n_fun_slow(d) for d in d_vals])
#    assert np.allclose([L_fun(d) for d in d_vals], [fun2(d) for d in d_vals])

    # L_fun and fun2 are monotonic functions, this means current scaling laws
    # have some issue - increasing d_model with constant FLOPS always decreases loss
    # and this does not make sense - we should be able to find a point where loss is minimal
    # which balances num_steps and d_model

#    res = minimize_scalar(function)
#    res_der = minimize_scalar(function_der)
    res_raw = minimize_scalar(fun2)

    dmodel = res_raw.x
    #dmodel_der = res_der.x
    #dmodel_raw = res_raw.x

    n_blocks = dmodel/dmodel_const
    n_params = p_fun(dmodel)
    n_steps = n_fun(dmodel)
    new_params = dict(n_params=n_params, n_steps=n_steps, flops=flops, dmodel=dmodel, n_blocks=n_blocks,
                      granularity=granularity, expansion_rate=expansion_rate, **params)
    assert np.isclose(calculate_flops(**new_params), flops)
    return new_params
