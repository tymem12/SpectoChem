from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.models import GCN
from torch_geometric.nn.models.schnet import GaussianSmearing

import torch.nn.functional as F

class GCNEncoder(nn.Module):
    handles_pos_encoding = False

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        **kwargs
    ):
        super().__init__()

        self.kwargs = kwargs
        self.out_channels = hidden_channels
        self.embedding = nn.Embedding(100, hidden_channels)
        
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        
        self.rbf_proj = nn.Linear(num_gaussians, 1)

        self.gnn = GCN(
            hidden_channels, hidden_channels, num_layers, hidden_channels, dropout
        )

    def forward(self, batch: Data):
        x = self.embedding(batch.z)
        
        edge_attr = self.distance_expansion(batch.edge_weight)
        
        learned_edge_weight = self.rbf_proj(edge_attr).squeeze(-1)
        learned_edge_weight = F.softplus(learned_edge_weight)

        h = self.gnn(x, batch.edge_index, edge_weight=learned_edge_weight)
        return h