from lizrd.core import misc
from lizrd.core.misc import resolve_activation_name
from research.conditional.utils.layer_manager import LoggingLayer, measure_time


class FeedForwardTimed(LoggingLayer):
    def __init__(self, dmodel, dff, activation_type="relu", no_ff=False):
        super().__init__()
        self.dmodel = dmodel
        self.no_ff = no_ff
        self.dff = dff
        self.logging_ff_pre_relu = misc.Linear(dmodel, dff)
        self.activation = resolve_activation_name(activation_type)
        self.logging_ff_post_relu = misc.Linear(dff, dmodel)

    def forward(self, x):
        with measure_time(self, "logging_ff_pre_relu"):
            if self.no_ff:
                return x
            x = self.logging_ff_pre_relu(x)
        with measure_time(self, "activation"):
            x = self.activation(x)
        with measure_time(self, "logging_ff_post_relu"):
            x = self.logging_ff_post_relu(x)
        return x
