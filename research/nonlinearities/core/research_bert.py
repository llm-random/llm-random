import torch
import lizrd.core.nn as nn

from lizrd.core import misc
from lizrd.core.bert import LowRank
from lizrd.support import ash
from lizrd.support.profile import TimerLayer


@ash.check("... d -> ... d")
def FeedForwardBottleneck(dmodel, exp_rate, bottleneck_chop_ratio=None):
    """
    :param dmodel: dimension of the model
    :param exp_rate: M/N, where N is the dimension of the model and M is the number of neurons in the middle layer (before ReLU)
    :param bottleneck_chop_ratio: only relevant for FeedForwardMultineck, carries out all the logic described next,
            and then divides the size of the bottleneck layer by the bottleneck_chop_ratio

    assumes that the number of parameters should be the same as for a FeedForward layer of the same input/output dimensions;
    with the bottleneck layer size being B, and in/out being N and M respectively, the resulting equation is:
        B(N+M) = NM

    we mainly want to compare ourselves with the original hyperparameters, so let's assume, that the M on th RHS is M = 4N
        B(N+M) = 4N^2
        B = 4N^2 / (N+M)

    now, with the expansion rate being a = M/N, we get
        B = 4N^2/(a+1)N = 4N/(a+1)
    to sum up, the above choice of the bottleneck size B guarantees that the number of parameters is the same as
    in a FeedForward layer of sizes d_in,d_out = N,4N
    """
    N = dmodel
    B = 4 * N / (exp_rate + 1)
    if bottleneck_chop_ratio is not None:
        B = int(B * bottleneck_chop_ratio)
    else:
        B = int(B)
    M = exp_rate * N
    return TimerLayer(
        "BottleneckFF",
        nn.Sequential(
            LowRank(N, M, B),
            nn.ReLU(inplace=True),
            LowRank(M, N, B),
        ),
    )


@ash.check("... d -> ... d")
class FeedForwardMultineck(nn.Module):
    """
    init params:
    :param dmodel: dimension of the model
    :param exp_rate: exp_rate: M/N, where M is the size of the "expanded" layer (before ReLU)
    :param n_heads: number of independent bottlenecks, later aggregated
    An iteration on FeedForwardBottleneck, where there are multiple bottlenecks, EACH writes to the output stream independently, like multiheadattention
    Assumes that the number of parameters should be the same as for a FeedForward layer of the same input/output dimensions;
    with the bottleneck layer size being B and the number of heads H, and in/out being N and M respectively, the resulting equation is:
        HB(N+M) = NM

    we mainly want to compare ourselves with the original hyperparameters, so let's assume, that the M on th RHS is M = 4N
        HB(N+M) = 4N^2
        HB = 4N^2 / (N+M)

    now, with the expansion rate being a = M/N, we get
        B = 4N^2/(a+1)NH = 4N/H(a+1)
    to sum up, the above choice of the bottleneck size B guarantees that the number of parameters is the same as
    in a FeedForward layer of sizes d_in,d_out = N,4N
    """

    # TODO: refactor this to use einsum instead of calling multiple separate instances of FeedForwardBottleneck.

    def __init__(self, dmodel, exp_rate, n_heads):
        super(FeedForwardMultineck, self).__init__()
        self.n_heads = n_heads
        self.bottleneck_heads = nn.ModuleList(
            [
                FeedForwardBottleneck(
                    dmodel, exp_rate, bottleneck_chop_ratio=1.0 / n_heads
                )
                for _ in range(n_heads)
            ]
        )

    def forward(self, x):
        x = torch.stack([head(x) for head in self.bottleneck_heads], axis=-1)
        x = torch.einsum("... h -> ...", x)
        x = x / torch.sqrt(self.n_heads)
        return x


@ash.check("... d -> ... d")
class FeedForwardInceptionNeck(nn.Module):
    """
    iteration on FeedForwardMultiNeck, where the heads are not the same size, but are defined by the head_sizes list (as fractions of the resulting B dimension)
    :return:
    """

    def __init__(self, dmodel, exp_rate, head_sizes):
        super(FeedForwardInceptionNeck, self).__init__()
        self.head_sizes = head_sizes
        self.bottleneck_heads = nn.ModuleList(
            [FeedForwardBottleneck(dmodel, exp_rate, bottleneck_chop_ratio=head_size) for head_size in head_sizes]
        )

    def forward(self, x):
        x = torch.stack([head(x) for head in self.bottleneck_heads], axis=-1)
        x = torch.einsum("... h -> ...", x)
        x = x / torch.sqrt(self.n_heads)
        return x

    pass


@ash.check("... d -> ... d")
class FeedForwardChoppedNeck(nn.Module):
    """
    Chop the input into chunks, and
    """
    pass
