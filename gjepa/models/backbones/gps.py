from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import ModuleList
from torch_geometric.nn import GPSConv

from gjepa.models.backbones.gnn import init_conv
from gjepa.models.backbones.gnn_pos import PosEncoder


class GraphGPS(nn.Module):
    """GraphGPS model, as introduced in https://arxiv.org/abs/2205.12454.

    Credits:
        - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py
        - https://github.com/rampasek/GraphGPS/blob/main/graphgps/layer/gps_layer.py
    """

    handles_pos_encoding = True
    supports_edge_weight = False
    supports_edge_attr = False

    def __init__(
        self,
        mpnn_conv: str | None,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        pos_dim: int,
        mpnn_kwargs: dict[str, Any] | None = None,
        gps_kwargs: dict[str, Any] | None = None,
        disable_pos: bool = False,  # Only for debug purposes
    ) -> None:
        super().__init__()
        self.out_channels = hidden_dim
        self.num_layers = num_layers

        self.pos_encoder = PosEncoder(in_channels, pos_dim, hidden_dim, disable_pos)

        self.gps_convs = ModuleList()
        for _ in range(num_layers):
            if mpnn_conv is not None:
                mpnn_net = init_conv(
                    mpnn_conv,
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    **(mpnn_kwargs or {}),
                )
            else:
                mpnn_net = None

            conv = GPSConv(
                channels=hidden_dim,
                conv=mpnn_net,
                **(gps_kwargs or {}),
            )
            self.gps_convs.append(conv)

    def forward(
        self, x: Tensor, edge_index: Tensor, pos: Tensor, batch: Tensor | None = None
    ) -> Tensor:
        assert batch is not None
        # to be used with to_dense_adj batch must be sorted
        sort_idx = torch.argsort(batch, stable=True)
        inv_sort_idx = torch.argsort(sort_idx, stable=True)
        x = x[sort_idx]
        edge_index = sort_idx[edge_index]
        pos = pos[sort_idx]
        batch = batch[sort_idx]

        x = self.pos_encoder(x=x, pos=pos)

        for i in range(self.num_layers):
            x = self.gps_convs[i](x=x, edge_index=edge_index, batch=batch)

        # revert sorting
        x = x[inv_sort_idx]

        return x
