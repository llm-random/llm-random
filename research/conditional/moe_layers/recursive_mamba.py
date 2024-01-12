import mamba_ssm
import torch
import torch.nn as nn


# class Mamba(nn.Module):
#     def __init__(self, dmodel) -> None:
#         super().__init__()
#         self.dmodel = dmodel

#     def forward(self, x):
#         batch, length = x.shape[0:2]
#         return torch.randn(batch, length, self.dmodel)


class ParallelMamba(nn.Module):
    def __init__(self, dmodel, n_mambas) -> None:
        super().__init__()

        self.mambas = nn.ModuleList(
            [mamba_ssm.Mamba(d_model=dmodel) for _ in range(n_mambas)]
        )
        self.n_mambas = n_mambas

    def forward(self, x):
        return torch.cat(
            [mamba(x[:, i :: self.n_mambas]) for i, mamba in enumerate(self.mambas)],
            dim=1,
        )


class RecursiveMambaGenerator:
    def __init__(self, dmodel, n_levels) -> None:
        print("DMODEL", dmodel)
        self.dmodel = dmodel
        self.n_levels = n_levels
        self.level = n_levels - 1

    def __call__(self):
        self.level = (self.level + 1) % self.n_levels
        return ParallelMamba(self.dmodel, 2**self.level)
