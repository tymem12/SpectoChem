import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch


class TransformerAttentionPoolModel(nn.Module):
    """
    Learnable attention pooling producing ONE embedding per compound (graph)
    from per-atom embeddings.

    Forward signature stays the same: forward(self, batch: Data)
    Expects:
      - batch.representation: [N, Fin]  (N = total atoms in the minibatch)
      - batch.batch:          [N]       (graph id per atom)
    Returns:
      - graph_emb:            [B, hidden_channels] (B = number of compounds)
    """

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

        # If your batch.representation dim is always 128 you can keep this as Identity.
        # If sometimes it differs, this will adapt it.
        fin = kwargs.get("in_channels", hidden_channels)
        self.input_proj = nn.Identity() if fin == hidden_channels else nn.Linear(fin, hidden_channels)

        # Set/graph encoder (operates on padded [B, max_nodes, F])
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)

        # Learnable query used to attend over node set -> one vector per graph
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
        x = batch.representation  # [N, Fin]
        x = self.input_proj(x)    # [N, hidden_channels]

        # Convert variable-size graphs to a padded dense tensor:
        # x_dense: [B, max_nodes, F], mask: [B, max_nodes] where True = real node
        x_dense, mask = to_dense_batch(x, batch.batch)

        # Transformer expects a padding mask with True meaning "ignore"
        key_padding_mask = ~mask  # [B, max_nodes] True for PAD positions

        # Encode node set (within each graph) with self-attention
        x_enc = self.encoder(x_dense, src_key_padding_mask=key_padding_mask)  # [B, max_nodes, F]

        # Attention pooling:
        # Query is learned and shared; each graph gets one pooled vector
        B = x_enc.size(0)
        q = self.query.expand(B, -1, -1)  # [B, 1, F]

        pooled, attn_weights = self.pool_attn(
            query=q,
            key=x_enc,
            value=x_enc,
            key_padding_mask=key_padding_mask,  # ensures PAD nodes get zero attention
            need_weights=False,
        )  # pooled: [B, 1, F]

        graph_emb = self.out_norm(pooled.squeeze(1))  # [B, F]
        return graph_emb
