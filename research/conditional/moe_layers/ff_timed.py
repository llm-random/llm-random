from lizrd.core import misc
from lizrd.core.misc import resolve_activation_name
from lizrd.core.misc import LoggingLayer, measure_time


class FeedForwardTimed(LoggingLayer):
    def __init__(
        self,
        dmodel,
        dff,
        activation_type="relu",
        no_ff=False,
        init_type="kaiming_uniform",
        init_scale=0.1,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.no_ff = no_ff
        self.dff = dff
        self.logging_ff_pre_relu = misc.Linear(
            dmodel, dff, init_type=init_type, init_scale=init_scale
        )
        self.activation = resolve_activation_name(activation_type)
        self.logging_ff_post_relu = misc.Linear(
            dff, dmodel, init_type=init_type, init_scale=init_scale
        )

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
