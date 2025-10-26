import copy
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver


class CustomGNN(nn.Module, ABC):
    """Custom implementation of pyg.nn.models.BasicGNN with varying channel sizes."""

    handles_pos_encoding = False

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        act: str | Callable | None = "relu",
        act_kwargs: dict[str, Any] | None = None,
        norm: str | Callable | None = None,
        norm_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.0,
        weights_standarization: bool = False,
        plain_last: bool = True,
        residual: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        hidden_channels = [in_channels] + hidden_channels
        self.num_layers = len(hidden_channels)
        self.out_channels = hidden_channels[-1]
        self.plain_last = plain_last
        self.standardize_weights = weights_standarization

        if residual and len(set(hidden_channels[1:])) > 1:
            raise ValueError("When using residual connections, hidden dimensions must be equal")
        self.residual = residual

        self.convs = nn.ModuleList()
        for in_dim, out_dim in zip(hidden_channels[:-1], hidden_channels[1:]):
            # dirty fix for interpolation of out_dim
            if "out_dim" in kwargs:
                del kwargs["out_dim"]
            self.convs.append(self.init_conv(in_channels=in_dim, out_channels=out_dim, **kwargs))

        act_layer = activation_resolver(act, **(act_kwargs or {}))
        self.activations = nn.ModuleList(
            [copy.deepcopy(act_layer) for _ in range(self.num_layers - 1)]
        )

        # self.norm_layers = nn.ModuleList()
        # for hidden_dim in hidden_channels[1:]:
        #     if norm is None:
        #         self.norm_layers.append(nn.Identity())
        #     else:
        #         self.norm_layers.append(
        #             normalization_resolver(norm, hidden_dim, **(norm_kwargs or {}))
        #         )
        self.norm_layers = nn.ModuleList([nn.BatchNorm1d(h) for h in hidden_channels[1:]])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, **conv_kwargs: Any) -> Tensor:
        if self.standardize_weights:
            self.standardize_conv_weights()

        for i, (conv, norm, act) in enumerate(zip(self.convs, self.norm_layers, self.activations)):
            x_l = conv(x=x, edge_index=edge_index, **conv_kwargs)

            if not self.plain_last or i < self.num_layers - 1:
                x_l = norm(x_l)
                x_l = act(x_l)
                x_l = self.dropout(x_l)

            if self.residual and i > 0:
                x = x_l + x
            else:
                x = x_l

        return x

    @abstractmethod
    def init_conv(self, in_channels: int, out_channels: int, **kwargs: Any) -> MessagePassing:
        return NotImplemented

    def standardize_conv_weights(self) -> None:
        """Credit: https://github.com/nerdslab/bgrl/blob/main/bgrl/models.py"""
        skipped_first_conv = False
        for m in self.modules():
            if isinstance(m, MessagePassing):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight


class CustomGCN(CustomGNN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs: Any) -> MessagePassing:
        return init_conv("gcn", in_channels=in_channels, out_channels=out_channels, **kwargs)


class CustomSAGE(CustomGCN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs: Any) -> MessagePassing:
        return init_conv("sage", in_channels=in_channels, out_channels=out_channels, **kwargs)


def init_conv(name: str, in_channels: int, out_channels: int, **conv_kwargs: Any) -> MessagePassing:
    name = name.lower().strip()
    if name == "gcn":
        return GCNConv(in_channels=in_channels, out_channels=out_channels, **conv_kwargs)
    elif name == "sage":
        return SAGEConv(in_channels=in_channels, out_channels=out_channels, **conv_kwargs)
    else:
        raise ValueError(f"Invalid convolution name: '{name}'")
