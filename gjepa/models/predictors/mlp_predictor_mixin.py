from typing import Optional

import torch.nn as nn
from torch import Tensor

class MLPPredictorMixin:
    _act_dict = {
        "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, 
        "elu": nn.ELU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[list[int]] = None,
        activation: Optional[str] = None
    ):
        if not hidden_channels:
            self.mlp = nn.Linear(in_channels, out_channels)
            return

        assert activation is not None, "Activation must be provided if hidden_channels is not empty."

        act_cls = self._act_dict[activation.lower()]

        layers = []
        dims = [in_channels] + hidden_channels + [out_channels]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(act_cls())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)
