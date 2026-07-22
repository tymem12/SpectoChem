from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv

from torch_geometric.nn.models.schnet import GaussianSmearing

class GINEEncoder(nn.Module):
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
        self.dropout = dropout

        self.embedding = nn.Embedding(100, hidden_channels)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )

            self.convs.append(GINEConv(
                nn=mlp, edge_dim=num_gaussians
            ))

    def forward(self, batch: Data):
        x = self.embedding(batch.z)

        edge_attr = self.distance_expansion(batch.edge_weight)

        for conv in self.convs:
            x = conv(x, batch.edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
