from plotly import express as px

from lizrd.core import misc
from lizrd.core.misc import resolve_activation_name
from research.conditional.utils.layer_manager import LoggingLayer, measure_time


class FeedForwardTimed(LoggingLayer):
    def __init__(self, dmodel, dff, activation_type="relu", no_ff=False):
        super().__init__()
        self.dmodel = dmodel
        self.no_ff = no_ff
        self.dff = dff
        self.lin1 = misc.Linear(dmodel, dff)
        self.activation = resolve_activation_name(activation_type)
        self.lin2 = misc.Linear(dff, dmodel)

    def forward(self, x):
        with measure_time(self, "forward"):
            x = self.lin1(x)
            x = self.activation(x)
            x = self.lin2(x)
        return x

    def log_heavy(self):
        instr_names = list(self.logging_cache["time"].keys())
        instr_times = list(self.logging_cache["time"].values())
        times_fig = px.bar(x=instr_names, y=instr_times)
        out = {"instruction_times_plot": times_fig}
        out.update(self.logging_cache["time"])
        return out
