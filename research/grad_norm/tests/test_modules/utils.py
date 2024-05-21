from unittest.mock import Mock, patch

import torch.nn as nn


class TorchIdModule(nn.Module):
    def forward(self, x):
        return x
