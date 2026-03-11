import torch
from torch import nn
from torch_geometric.nn import radius_graph, global_add_pool
from torch_geometric.data import Data

try:
    from torch_geometric.nn.models import SchNet
except ImportError:
    from torch_geometric.nn.models.schnet import SchNet

class SchNetEncoder(SchNet):
    handles_pos_encoding = True

    def __init__(self,
                 hidden_channels: int = 128,
                 num_filters: int = 128,
                 num_interactions: int = 6,
                 num_gaussians: int = 50,
                 cutoff: float = 10.0,
                 max_num_neighbors: int = 64,
                 readout: str = 'add',
                 **kwargs):
        super().__init__(hidden_channels=hidden_channels,
                         num_filters=num_filters,
                         num_interactions=num_interactions,
                         num_gaussians=num_gaussians,
                         cutoff=cutoff,
                         max_num_neighbors=max_num_neighbors,
                         readout=readout

                         )
        self.kwargs = kwargs
        self.out_channels = self.hidden_channels

    def forward(self, batch: Data):
        pos = batch.pos
        z = batch.z
        batch = batch.batch
        batch = torch.zeros_like(z) if batch is None else batch
        h = self.embedding(z)

        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        return h 
