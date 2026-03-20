from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.models import GCN

class GCNEncoder(nn.Module):
    handles_pos_encoding = False

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0,
        **kwargs
    ):
        super().__init__()

        self.kwargs = kwargs
        self.out_channels = hidden_channels
        self.embedding = nn.Embedding(100, hidden_channels)

        self.gnn = GCN(
            hidden_channels, hidden_channels, num_layers, hidden_channels, dropout
        )

    def forward(self, batch: Data):
        x = self.embedding(batch.z)
        h = self.gnn(x, batch.edge_index)
        return h
