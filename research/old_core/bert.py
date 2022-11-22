import torch
import lizrd.core.nn as nn

from einops.layers.torch import Rearrange

from lizrd.core import misc
from lizrd.support import ash
from lizrd.support.profile import TimerLayer

@ash.check("... d -> ... d")
class Attention(nn.Module):
    def __init__(self, dmodel, heads, dhead=None, layer_fun=None):
        super(Attention, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads
        self.heads = heads
        self.dhead = dhead
        self.dmodel = dmodel
        if layer_fun is None:
            layer_fun = lambda: misc.EinMix(
                "... dmodel -> ... (heads dhead)",
                weight_shape="dmodel heads dhead",
                bias_shape="heads dhead",
                dmodel=dmodel,
                heads=heads,
                dhead=dhead,
            )
        layer_fun_and_reshape = lambda: nn.Sequential(
            TimerLayer("QKVproj", layer_fun()),
            Rearrange("... (heads dhead) -> ... heads dhead", heads=heads, dhead=dhead),
        )

        self.Q = layer_fun_and_reshape()
        self.K = layer_fun_and_reshape()
        self.V = layer_fun_and_reshape()

        # self.A = Reduce('... seqlen1 heads dhead, ... seqlen2 heads dhead -> ... heads seqlen1 seqlen2')

        self.D = nn.Sequential(
            Rearrange("... heads dhead -> ... (heads dhead)", heads=heads, dhead=dhead),
            layer_fun(),
        )

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        a = torch.einsum("... l h d, ... L h d -> ... h l L", q, k)
        a = a * (1 / self.dhead**0.5)
        a = torch.softmax(a, dim=-1)
        prefinal = torch.einsum("... h l L, ... L h d -> ... l h d", a, v)
        output = self.D(prefinal)
        return output
