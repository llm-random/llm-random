import torch

from lizrd.core.nn import Module


class Sum_norm(Module):
    def __init__(self, n_bias_copies):
        super(Sum_norm, self).__init__()
        self.n_bias_copies = n_bias_copies

    def forward(self, x):
        return x.sum(x, dim=-1) / (self.n_bias_copies**0.5)


class Max(Module):
    def __init__(self):
        super(Max, self).__init__()

    def forward(self, x):
        return x.max(x, dim=-1)


class MultineckShufflePermute(Module):
    def __init__(self, n_heads):
        super(MultineckShufflePermute, self).__init__()
        self.n_heads = n_heads

    def forward(self, x):
        x = torch.permute(x, (0, 1, 3, 2)).reshape(
            (x.shape[0], x.shape[1], self.n_heads, -1)
        )
        return x
