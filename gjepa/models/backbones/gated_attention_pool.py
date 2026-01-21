import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import softmax


class GatedAttentionPoolModel(nn.Module):
    def __init__(self, hidden_channels: int = 128, gate_hidden: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.handles_pos_encoding = True
        self.kwargs = kwargs
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels

        fin = kwargs.get("in_channels", hidden_channels)
        self.input_proj = nn.Identity() if fin == hidden_channels else nn.Linear(fin, hidden_channels)

        self.attn_tanh = nn.Sequential(
            nn.Linear(hidden_channels, gate_hidden),
            nn.Tanh(),
        )
        self.attn_sigmoid = nn.Sequential(
            nn.Linear(hidden_channels, gate_hidden),
            nn.Sigmoid(),
        )
        self.attn_out = nn.Linear(gate_hidden, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden_channels)

    def forward(self, batch: Data):
        x = self.input_proj(batch.representation)
        x = self.dropout(x)

        h = self.attn_tanh(x) * self.attn_sigmoid(x)
        score = self.attn_out(h).squeeze(-1)

        alpha = softmax(score, batch.batch)

        graph_emb = torch.zeros(
            (int(batch.batch.max()) + 1, x.size(-1)),
            device=x.device,
            dtype=x.dtype
        )
        graph_emb.index_add_(0, batch.batch, x * alpha.unsqueeze(-1))

        output_shape = self.out_norm(graph_emb)
        return output_shape
