from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.models import GAT

class GATEncoder(nn.Module):
    handles_pos_encoding = False

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0,
        heads: int = 4,
        **kwargs
    ):
        super().__init__()

        self.kwargs = kwargs

        self.out_channels = hidden_channels
        self.embedding = nn.Embedding(100, hidden_channels)

        self.gnn = GAT(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            dropout=dropout,
            heads=heads
        )

    def forward(self, batch: Data):
        x = self.embedding(batch.z)
        h = self.gnn(x, batch.edge_index, edge_weight=batch.edge_weight)
        return h
