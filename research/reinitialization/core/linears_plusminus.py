import torch
from torch import nn
import torch.nn.functional as F
from lizrd.core import misc
from lizrd.support import ash


@ash.check("... d -> ... d")
class PlusMinusFF(nn.Module):
    def __init__(self, dmodel: int, dff: int):
        super().__init__()
        self.lin1 = misc.Linear(dmodel, dff, bias=False)
        self.lin2 = misc.Linear(2 * dff, dmodel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plus_x = self.lin1(x)
        minus_x = -plus_x
        plus_x = F.relu(plus_x)
        minus_x = F.relu(minus_x)
        x = torch.concat([plus_x, minus_x], dim=-1)
        return self.lin2(x)
