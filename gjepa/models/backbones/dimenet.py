import torch
from torch import Tensor
from torch_geometric.nn.models.dimenet import DimeNet, triplets
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data

class DimeNetEncoder(DimeNet):
    handles_pos_encoding = True

    def __init__(self, hidden_channels: int, num_blocks: int, num_bilinear: int,
                 num_spherical: int, num_radial: int, cutoff: float = 5.0,
                 max_num_neighbors: int = 32, envelope_exponent: int = 5,
                 num_before_skip: int = 1, num_after_skip: int = 2,
                 num_output_layers: int = 3, **kwargs):
        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,             # <-- key change
            num_blocks=num_blocks,
            num_bilinear=num_bilinear,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
        )
        self.out_channels = hidden_channels
        self.kwargs = kwargs
    def forward(self, batch: Data) -> Tensor:
        pos = batch.pos
        z = batch.z.long()                # ensure correct dtype for embedding
        batch_idx = batch.batch

        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch_idx, max_num_neighbors=self.max_num_neighbors
        )

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=pos.size(0)
        )

        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        pos_ji = pos[idx_j] - pos[idx_i]
        pos_ki = pos[idx_k] - pos[idx_i]
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)                  # [E, R]
        sbf = self.sbf(dist, angle, idx_kj)   # [T, S*R]

        x_e = self.emb(z, rbf, i, j)          # [E, H]

        # Node embeddings accumulated across blocks (DimeNet style):
        h = self.output_blocks[0](x_e, rbf, i, num_nodes=pos.size(0))  # [N, H]

        for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            x_e = interaction_block(x_e, rbf, sbf, idx_kj, idx_ji)      # [E, H]
            h = h + output_block(x_e, rbf, i, num_nodes=pos.size(0))    # [N, H]

        return h  # [N, H]
