import torch
from torch import nn
from torch_geometric.data import Data

try:
    from torch_geometric.nn.models import SchNet
except ImportError:
    from torch_geometric.nn.models.schnet import SchNet

# atencja - query - osadzenie z schneta; key/value: osadzenie z umy -> schnet (atencja albo przed albo po schnecie) 

class GeometricGatingEncoder(SchNet):
    handles_pos_encoding = True

    def __init__(self,
                 hidden_channels: int = 128,
                 num_filters: int = 128,
                 num_interactions: int = 6,
                 num_gaussians: int = 50,
                 cutoff: float = 10.0,
                 max_num_neighbors: int = 64,
                 readout: str = 'add',
                 # Attention & Gating Parameters
                 num_attention_heads: int = 4,
                 attention_dropout: float = 0.1,
                 post_interaction_gating: bool = False, # <--- NEW FLAG
                 **kwargs):
        """
        Args:
            post_interaction_gating (bool):
                If False: Gating happens BEFORE SchNet interactions (Query = Atom Identity).
                If True: Gating happens AFTER SchNet interactions (Query = Learned Chemical Environment).
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
        self.post_interaction_gating = post_interaction_gating

        # Cross-Attention Module
        # Query: SchNet Embedding (Batch, 1, Hidden)
        # Key/Value: UMA Invariants (Batch, 3, Hidden)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True 
        )

        # Layer Norm to stabilize the fusion
        self.norm = nn.LayerNorm(hidden_channels)

        # Extra MLP for post-interaction fusion to mix the signals properly
        if self.post_interaction_gating:
            self.final_mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )

    def forward(self, batch: Data):
        # 1. Standard SchNet Setup
        pos = batch.pos
        z = batch.z
        batch_idx = batch.batch
        batch_idx = torch.zeros_like(z) if batch_idx is None else batch_idx
        
        # 2. Prepare UMA "Key/Value" Sequence (The "Frozen Knowledge")
        uma_repr = batch.representation # [N, 9, 128]
        s = uma_repr[:, 0, :]                             # Scalar
        v = torch.linalg.norm(uma_repr[:, 1:4, :], dim=1) # Vector Magnitude
        t = torch.linalg.norm(uma_repr[:, 4:9, :], dim=1) # Tensor Magnitude
        
        # Sequence: [N, 3, 128]
        uma_kv = torch.stack([s, v, t], dim=1)

        # 3. Initial Node Embedding
        h = self.embedding(z)

        # 4. Compute SchNet Graph Structure
        edge_index, edge_weight = self.interaction_graph(pos, batch_idx)
        edge_attr = self.distance_expansion(edge_weight)

        # ==========================================
        # PATH A: Pre-Interaction Gating (Standard)
        # ==========================================
        if not self.post_interaction_gating:
            # Query is just Atom Identity
            h_query = h.unsqueeze(1)
            
            # Attend
            attn_out, _ = self.cross_attn(h_query, uma_kv, uma_kv, need_weights=False)
            h_geom_gated = attn_out.squeeze(1)
            
            # Fuse
            h = h + h_geom_gated
            h = self.norm(h)

            # Run Interactions
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # ==========================================
        # PATH B: Post-Interaction Gating (New)
        # ==========================================
        else:
            # Run Interactions First
            # Query becomes the "Learned Chemical Environment"
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)
            
            # Now `h` contains rich local info. Use IT to query UMA.
            h_query = h.unsqueeze(1)
            
            # Attend
            attn_out, _ = self.cross_attn(h_query, uma_kv, uma_kv, need_weights=False)
            h_geom_gated = attn_out.squeeze(1)

            # Residual Fuse: SchNet_Output + Gated_Geometry
            h = h + h_geom_gated
            h = self.norm(h)
            
            # Final mixing to smooth the result
            h = self.final_mlp(h)

        return h