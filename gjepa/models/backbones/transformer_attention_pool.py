import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch


class TransformerAttentionPoolModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 1,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.handles_pos_encoding = True
        self.kwargs = kwargs

        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels

        fin = kwargs.get("in_channels", hidden_channels)
        self.input_proj = nn.Identity() if fin == hidden_channels else nn.Linear(fin, hidden_channels)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)

        self.query = nn.Parameter(torch.empty(1, 1, hidden_channels))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.pool_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_norm = nn.LayerNorm(hidden_channels)

    def forward(self, batch: Data):
        x = batch.representation
        x = self.input_proj(x)

        x_dense, mask = to_dense_batch(x, batch.batch)

        key_padding_mask = ~mask
        x_enc = self.encoder(x_dense, src_key_padding_mask=key_padding_mask)

        B = x_enc.size(0)
        q = self.query.expand(B, -1, -1)

        pooled, attn_weights = self.pool_attn(
            query=q,
            key=x_enc,
            value=x_enc,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        graph_emb = self.out_norm(pooled.squeeze(1))
        return graph_emb
