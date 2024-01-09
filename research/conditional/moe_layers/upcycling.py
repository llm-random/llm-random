import dataclasses

from research.conditional.utils.layer_manager import LoggingLayer


@dataclasses.dataclass(eq=False, repr=False)
class MoEUpcycling(LoggingLayer):
    """Mixture-of-Experts upcycling layer.
    We first train vanilla transformer for start_mode_n_steps and then we use their weights to
    create MoE layer.
    """

    vanilla: lambda: LoggingLayer
    mot: lambda: LoggingLayer

    # dmodel: int
    # n_experts: int
    # expert_size: int
    # init_type: str
    # init_scale: float

    def __post_init__(self):
        super().__init__()
        # assert (
        #     self.expert_size % self.n_experts == 0
        # ), f"expert_size {self.expert_size} must be divisible by n_experts {self.n_experts}. We might support other granularities in the future."
        self.current_mode = "vanilla"

        # instantiate submodules
        self.vanilla = self.vanilla()
        self.mot = self.mot()

        # initialize shared weights
        # self.lin1 = torch.nn.Parameter(
        #     get_init_weight(
        #         shape=(self.n_experts, self.dmodel, self.expert_size),
        #         fan_in=self.dmodel,
        #         init_type=self.init_type,
        #         scale=self.init_scale,
        #     )
        # )
        # self.lin2 = torch.nn.Parameter(
        #     get_init_weight(
        #         shape=(self.n_experts, self.expert_size, self.dmodel),
        #         fan_in=self.expert_size,
        #         init_type=self.init_type,
        #         scale=self.init_scale,
        #     )
        # )

        # self.controller = torch.nn.Parameter(
        #     get_init_weight(
        #         shape=(self.dmodel, self.n_experts),
        #         fan_in=self.dmodel,
        #         init_type=self.init_type,
        #         scale=self.init_scale,
        #     )
        # )

        # replace weights in submodules
        ## mot
        # self.mot.lin1 = self.lin1
        # self.mot.lin2 = self.lin2
        # self.mot.controller = self.controller

        ## ec
        # self.ec.lin1_weight = self.lin1
        # self.ec.lin2_weight = self.lin2
        # self.ec.gate = self.controller
        # self.ec.expert_gating.gate = self.controller

        # ## switch
        # self.switch.lin1_weight = self.lin1
        # self.switch.lin2_weight = self.lin2
        # self.switch.router.gate = self.controller
        # self.ec.chimera_layer = True
        # self.ec.expert_gating.chimera_layer = True

        # self.mot.chimera_layer = True

        # self.switch.chimera_layer = True
        # self.switch.router.chimera_layer = True

    def change_to_mot(self):
        self.current_mode = "mot"

        # copy weights
        transposed_ff_pre_relu_layer = (
            self.vanilla.layer.logging_ff_pre_relu.weight.t().clone()
        )
        transposed_ff_post_relu_layer = (
            self.vanilla.layer.logging_ff_post_relu.weight.t().clone()
        )
        self.mot.lin1.data[:, :, :] = transposed_ff_pre_relu_layer.unsqueeze(0).expand(
            32, -1, -1
        )
        self.mot.lin2.data[:, :, :] = transposed_ff_post_relu_layer.unsqueeze(0).expand(
            32, -1, -1
        )

    def get_current_module(self):
        if self.current_mode == "vanilla":
            return self.vanilla
        elif self.current_mode == "mot":
            return self.mot
        else:
            raise ValueError("current_mode not set or unknown")

    def forward(self, x):
        self.get_current_module().logging_switch = self.logging_switch
        return self.get_current_module().forward(x)

    def log_heavy(self):
        if self.current_mode == "mot":
            return self.get_current_module().log_heavy()
        else:
            return {}
