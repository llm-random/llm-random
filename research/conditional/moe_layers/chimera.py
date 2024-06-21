from lizrd.core.misc import (
    LoggingLayer,
)
from research.conditional.moe_layers.expert_choice import ExpertChoiceFF
from research.conditional.moe_layers.token_choice import TokenChoiceFF


class ChimeraFF(LoggingLayer):
    def __init__(
        self,
        change_after,
        **models_args,
    ):
        super().__init__()
        self.expert_choice = ExpertChoiceFF(**models_args)
        get_gate_fun = self.expert_choice.gating.get_gate
        self.token_choice = TokenChoiceFF(**models_args, get_gate_fun=get_gate_fun)
        self.iter = 0
        self.change_after = change_after
        # TODO odpisać layenora do token choice
        # TODO przepiąć layernorma

    def choose_model(self):
        if self.training:
            self.iter += 1
        if self.iter <= self.change_after:
            return self.expert_choice
        else:
            return self.token_choice

    def forward(self, x):
        model = self.choose_model()
        return model(x)