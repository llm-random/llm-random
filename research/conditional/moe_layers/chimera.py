import torch

from lizrd.core.initialization import get_init_weight
from research.conditional.utils.layer_manager import LoggingLayer


class Chimera(LoggingLayer):
    """Mixture-of-Experts Chimera layer. Expert and controller weights are shared between a Mot, EC and Switch submodules.
    start_prob refers to probability of choosing first mode. If start_prob != end_prob, then we change probabilities gradually using cosine scheduler
    """

    def __init__(
        self,
        mot: lambda: LoggingLayer,
        ec: lambda: LoggingLayer,
        switch: lambda: LoggingLayer,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        init_type: str,
        init_scale: float,
    ):
        super().__init__()
        self.mot = mot
        self.ec = ec
        self.switch = switch
        self.dmodel = dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.init_type = init_type
        self.init_scale = init_scale

    def __post_init__(self):
        super().__init__()
        assert (
            self.expert_size % self.n_experts == 0
        ), f"expert_size {self.expert_size} must be divisible by n_experts {self.n_experts}. We might support other granularities in the future."
        self.current_mode = "mot"
        self.possible_modes = []
        # instantiate submodules
        self.mot = self.mot()
        self.ec = self.ec()
        self.switch = self.switch()

        # initialize shared weights
        self.lin1 = torch.nn.Parameter(
            get_init_weight(
                shape=(self.n_experts, self.dmodel, self.expert_size),
                fan_in=self.dmodel,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )
        self.lin2 = torch.nn.Parameter(
            get_init_weight(
                shape=(self.n_experts, self.expert_size, self.dmodel),
                fan_in=self.expert_size,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )
        self.controller = torch.nn.Parameter(
            get_init_weight(
                shape=(self.dmodel, self.n_experts),
                fan_in=self.dmodel,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )

        # replace weights in submodules
        ## mot
        self.mot.lin1 = self.lin1
        self.mot.lin2 = self.lin2
        self.mot.controller = self.controller

        ## ec
        self.ec.lin1_weight = self.lin1
        self.ec.lin2_weight = self.lin2
        self.ec.gate = self.controller
        self.ec.expert_gating.gate = self.controller

        ## switch
        self.switch.lin1_weight = self.lin1
        self.switch.lin2_weight = self.lin2
        self.switch.router.gate = self.controller
        self.ec.chimera_layer = True
        self.ec.expert_gating.chimera_layer = True

        self.mot.chimera_layer = True

        self.switch.chimera_layer = True
        self.switch.router.chimera_layer = True

    def set_mode(self, mode):
        assert mode in [
            "mot",
            "ec",
            "switch",
        ], f"mode {mode} not supported. It must be one of ['mot', 'ec', 'switch']"
        self.current_mode = mode
        if mode not in self.possible_modes:
            self.possible_modes.append(mode)

    def get_current_module(self):
        if self.current_mode == "mot":
            return self.mot
        elif self.current_mode == "ec":
            return self.ec
        elif self.current_mode == "switch":
            return self.switch
        else:
            raise ValueError("current_mode not set")

    def forward(self, x):
        self.get_current_module().logging_switch = self.logging_switch
        return self.get_current_module().forward(x)

    def log_heavy(self):
        return self.get_current_module().log_heavy()
