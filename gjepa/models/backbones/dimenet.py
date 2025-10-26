import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.nn.models.dimenet import DimeNet
from torch_geometric.nn import radius_graph
# Upewnij się, że masz te same importy co w oryginalnym DimeNet:
from torch_geometric.nn.models.dimenet import triplets
# from torch_geometric.nn.models.dimenet_utils import glorot_orthogonal
# from torch_geometric.nn.dense.linear import Linear
from torch.nn import Embedding, Linear
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.data import Data

class DimeNetEncoder(DimeNet):

    handles_pos_encoding = True

    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        out_channels: int = 1,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        **kwargs
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_bilinear=num_bilinear,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers
        )
        self.kwargs = kwargs
        self.out_channels = hidden_channels

    def forward(
        self,
        batch: Data
    ) -> Tensor:
        
        pos = batch.pos
        z = batch.z
        batch = batch.batch
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=pos.size(0)
        )

        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)                        # [E, R]
        sbf = self.sbf(dist, angle, idx_kj)         # [T, S*R]

        x_e = self.emb(z, rbf, i, j)                # [E, H]

        for interaction_block in self.interaction_blocks:
            x_e = interaction_block(x_e, rbf, sbf, idx_kj, idx_ji)  # [E, H]

        h = scatter(x_e, i, dim=0, dim_size=pos.size(0), reduce='sum')  # [N, H]

        return h
