from typing import Any
import traceback
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from gjepa.models.backbones.gnn import CustomGCN, CustomGNN, CustomSAGE


class PosEncoder(nn.Module):
    """Basic encoder for combining positional encodings with input features."""

    def __init__(
        self,
        feat_in_dim: int,
        pos_in_dim: int,
        hidden_dim: int,
        disable_pos: bool = False,  # for debug purposes only
    ):
        super().__init__()
        self.disable_pos = disable_pos

        if disable_pos:
            self.proj = nn.Linear(feat_in_dim, hidden_dim)
        else:
            self.proj = nn.Linear(feat_in_dim + pos_in_dim, hidden_dim)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
    ) -> Tensor:
        
        if not self.disable_pos:
            x = torch.cat([x, pos], dim=-1)

        x = self.proj(x)
        return x


class PositionalCustomGNN(nn.Module):
    """Version of CustomGNN which handles positional encodings."""

    handles_pos_encoding = True

    def __init__(
        self,
        conv: str,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        pos_dim: int,
        disable_pos: bool = False,
        **gnn_kwargs: Any,
    ) -> None:
        super().__init__()
        self.out_channels = hidden_dim
        self.num_layers = num_layers

        self.pos_encoder = PosEncoder(in_channels, pos_dim, hidden_dim, disable_pos)
        self.gnn = self.init_gnn(conv, hidden_dim, [hidden_dim] * (num_layers - 1), **gnn_kwargs)

    def forward(self, batch: Data) -> Tensor:
        x = batch.x
        edge_index = batch.edge_index
        pos = batch.positional_encoding
        batch = batch.batch
        x = self.pos_encoder(x, pos)

        output = self.gnn(x=x, edge_index=edge_index)
        return output

    @staticmethod
    def init_gnn(
        conv: str,
        in_channels: int,
        hidden_channels: list[int],
        **gnn_kwargs: Any,
    ) -> CustomGNN:
        if conv == "gcn":
            return CustomGCN(in_channels, hidden_channels, **gnn_kwargs)
        elif conv == "sage":
            return CustomSAGE(in_channels, hidden_channels, **gnn_kwargs)
        else:
            raise ValueError(f"Invalid conv: '{conv}'")
