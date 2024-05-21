import torch.nn as nn
from unittest.mock import Mock, patch


class TorchIdModule(nn.Module):
    def forward(self, x):
        return x
