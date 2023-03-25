import einops
import torch
from torch import nn
import torch.nn.functional as F
from lizrd.core import misc
from lizrd.core.misc import EinMix
from lizrd.support import ash


@ash.check("... d -> ... d")
class PlusMinusFF(nn.Module):
    def __init__(self, dmodel: int, dff: int):
        super().__init__()
        self.lin1 = misc.Linear(dmodel, dff, bias=False)
        self.lin2 = misc.Linear(2 * dff, dmodel, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plus_x = self.lin1(x)
        minus_x = -plus_x
        plus_x = F.relu(plus_x)
        minus_x = F.relu(minus_x)
        x = torch.concat([plus_x, minus_x], dim=-1)
        return self.lin2(x)


@ash.check("... d -> ... d")
class MultiGroupedConv(nn.Module):
    def __init__(self, dmodel: int, dff: int, weight_const=1.0, mult_const=1.0):
        super().__init__()
        # example
        # dmodel = 1024
        # dff = 4096
        # dmA = 32
        # dmB = 32
        # dmC = 64
        # dmA * dmB = 32 * 32 = 1024
        # (dmA + dmB) * dmC = (32 + 32) * 64 = 4096

        # assert dmA * dmB == dmodel
        # assert (dmA + dmB) * dmC == dff

        dmA = int(dmodel ** 0.5)
        dmB = int(dmodel // dmA)
        assert dmodel * 0.9 < dmA * dmB < dmodel * 1.1
        dmC = int(dff * dmodel // ((dmA + dmB) * dmA * dmB))
        assert dff * 0.9 < (dmA + dmB) * dmC < dff * 1.1
        assert dmodel * dff * 0.9 < dmA * dmB * dmC * (dmA + dmB) < dmodel * dff * 1.1

        # TODO: lin1A and lin1B can be combined into one EinMix, with even less dimensions
        self.lin1A = EinMix(
            "... dmA dmB -> ... dmA dmBp dmC",
            weight_shape="dmA dmB dmBp dmC",
            # bias_shape="",
            dmA=dmA,
            dmB=dmB,
            dmBp=dmB,
            dmC=dmC
        )
        self.lin1A.layer.weight.data *= weight_const

        self.lin1B = EinMix(
            "... dmA dmB -> ... dmAp dmB dmC",
            weight_shape="dmA dmB dmAp dmC",
            # bias_shape="",
            dmA=dmA,
            dmB=dmB,
            dmAp=dmA,
            dmC=dmC
        )
        self.lin1B.layer.weight.data *= weight_const

        self.lin2A = EinMix(
            "... dmA dmBp dmC -> ... dmA dmB",
            weight_shape="dmA dmBp dmC dmB",
            # bias_shape="",
            dmA=dmA,
            dmB=dmB,
            dmBp=dmB,
            dmC=dmC
        )
        self.lin2A.layer.weight.data *= weight_const

        self.lin2B = EinMix(
            "... dmAp dmB dmC -> ... dmA dmB",
            weight_shape="dmAp dmB dmC dmA",
            # bias_shape="",
            dmA=dmA,
            dmB=dmB,
            dmAp=dmA,
            dmC=dmC
        )
        self.lin2B.layer.weight.data *= weight_const

        self.dmodel = dmodel
        self.dff = dff
        self.dmA = dmA
        self.dmB = dmB
        self.dmC = dmC
        self.mult_const = mult_const
        self.weight_const = weight_const

        self.permin = torch.randperm(dmodel)
        self.permout = torch.randperm(dmodel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.dmodel
        x = x[..., self.permin]
        if self.dmodel != self.dmA * self.dmB:
            # remove last elements so last dimensions is dmA * dmB
            assert self.dmodel > self.dmA * self.dmB
            x = x[..., :self.dmA * self.dmB]
        # reshape last dimensions to dmA dmB
        x = einops.rearrange(x, "... (dmA dmB) -> ... dmA dmB", dmA=self.dmA, dmB=self.dmB)
        col_x = self.lin1A(x)  # batch seqlen dmA dmBp dmC
        row_x = self.lin1B(x)  # batch seqlen dmAp dmB dmC
        x = (col_x + row_x) * self.mult_const
        x = F.relu(x)
        col_x = self.lin2A(x)  # batch seqlen dmA dmB
        row_x = self.lin2B(x)  # batch seqlen dmAp dmB
        x = (col_x + row_x) * self.mult_const
        x = einops.rearrange(x, "... dmA dmB -> ... (dmA dmB)", dmA=self.dmA, dmB=self.dmB)
        # add back last elements
        if self.dmodel != self.dmA * self.dmB:
            assert self.dmodel > self.dmA * self.dmB
            x = torch.cat([x, x.new_zeros(x.shape[:-1] + (self.dmodel - self.dmA * self.dmB,))], dim=-1)
        x = x[..., self.permout]
        return x
