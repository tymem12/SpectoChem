import torch
from torch import nn
from torch_geometric.nn import radius_graph, global_add_pool
from torch_geometric.data import Data

try:
    from torch_geometric.nn.models import SchNet
except ImportError:
    from torch_geometric.nn.models.schnet import SchNet

class InvariantProjectionEncoder(SchNet):
    handles_pos_encoding = True

    def __init__(self,
                 hidden_channels: int = 128,
                 num_filters: int = 128,
                 num_interactions: int = 6,
                 num_gaussians: int = 50,
                 cutoff: float = 10.0,
                 max_num_neighbors: int = 64,
                 readout: str = 'add',
                 use_mlp_projection: bool = True,
                 uma_mlp_hidden_dim: int = 256,
                 use_z_embedding: bool = True,
                 **kwargs):
        """
        Args:
            use_mlp_projection (bool): 
                If True: Concatenates invariants [S, V_norm, T_norm] and projects via MLP.
                If False: Sums invariants using learnable scalar weights.
            uma_mlp_hidden_dim (int): Hidden dimension for the MLP if use_mlp_projection is True.
            use_z_embedding (bool): 
                If True: Adds standard SchNet atom embedding e(z) to the UMA projection.
                If False: Uses only the UMA projection as the initial node feature.
        """
        super().__init__(hidden_channels=hidden_channels,
                         num_filters=num_filters,
                         num_interactions=num_interactions,
                         num_gaussians=num_gaussians,
                         cutoff=cutoff,
                         max_num_neighbors=max_num_neighbors,
                         readout=readout)
        
        self.kwargs = kwargs
        self.out_channels = self.hidden_channels
        self.use_mlp_projection = use_mlp_projection
        self.use_z_embedding = use_z_embedding

        uma_dim = hidden_channels 

        if self.use_mlp_projection:
            input_dim = 3 * uma_dim
            self.uma_projector = nn.Sequential(
                nn.Linear(input_dim, uma_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(uma_mlp_hidden_dim, hidden_channels)
            )
        else:
            self.w_scalar = nn.Parameter(torch.tensor(0.33))
            self.w_vector = nn.Parameter(torch.tensor(0.33))
            self.w_tensor = nn.Parameter(torch.tensor(0.33))

    def forward(self, batch: Data):
        # 1. Standard SchNet Inputs
        pos = batch.pos
        z = batch.z
        batch_idx = batch.batch
        batch_idx = torch.zeros_like(z) if batch_idx is None else batch_idx
        
        uma_repr = batch.representation 

        s = uma_repr[:, 0, :]  # [N, 128]

        # Channels 1-3: Vector (L=1) - take Euclidean Norm along channel dim
        v = torch.linalg.norm(uma_repr[:, 1:4, :], dim=1) # [N, 128]

        # Channels 4-8: Tensor (L=2) -> take Frobenius Norm along channel dim
        t = torch.linalg.norm(uma_repr[:, 4:9, :], dim=1) # [N, 128]

        # fuse UMA Invariants
        if self.use_mlp_projection:
            # concatenate [N, 128*3] -> [N, 384]
            combined = torch.cat([s, v, t], dim=-1)
            uma_h = self.uma_projector(combined) # -> [N, 128]
        else:
            # Weighted Sum
            uma_h = (self.w_scalar * s) + (self.w_vector * v) + (self.w_tensor * t)

        # create initial node Eembedding (h)
        if self.use_z_embedding:
            # add learnable atom type embedding (standard SchNet e(z))
            h_z = self.embedding(z)
            h = h_z + uma_h
        else:
            # use only geometric info
            h = uma_h

        edge_index, edge_weight = self.interaction_graph(pos, batch_idx)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        return h
