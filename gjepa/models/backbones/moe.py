import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

from gjepa.models.backbones.schnet import SchNetEncoder

class MoESchNetEncoder(nn.Module):
    handles_pos_encoding = True

    def __init__(self,
                 num_experts: int = 3,
                 use_full_uma_for_router: bool = True,
                 router_hidden_dim: int = 64,
                 routing_mode: str = "soft",
                 hidden_channels: int = 128,
                 **kwargs):
        """
        Args:
            routing_mode (str): 
                "soft": Weighted average of all experts (Softmax).
                "hard": Selects only the best expert per atom (Gumbel-Softmax).
        """
        super().__init__()
        
        self.out_channels = hidden_channels

        self.num_experts = num_experts
        self.use_full_uma_for_router = use_full_uma_for_router
        self.routing_mode = routing_mode.lower()

        if self.routing_mode not in ["soft", "hard"]:
            raise ValueError(f"routing_mode must be 'soft' or 'hard', got {self.routing_mode}")

        self.experts = nn.ModuleList([
            SchNetEncoder(hidden_channels=hidden_channels, **kwargs)
            for _ in range(num_experts)
        ])
        
        uma_dim = hidden_channels 
        router_input_dim = 3 * uma_dim if use_full_uma_for_router else uma_dim

        self.router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_experts)
        )

    def forward(self, batch: Data):
        uma_repr = batch.representation # [N, 9, 128]
        s = uma_repr[:, 0, :]
        
        if self.use_full_uma_for_router:
            v = torch.linalg.norm(uma_repr[:, 1:4, :], dim=1)
            t = torch.linalg.norm(uma_repr[:, 4:9, :], dim=1)
            router_input = torch.cat([s, v, t], dim=-1)
        else:
            router_input = s

        logits = self.router(router_input) # [N, Num_Experts]

        if self.routing_mode == "soft":
            # standard Softmax: Weights sum to 1, all experts contribute slightly
            gate_weights = F.softmax(logits, dim=-1)
        elif self.routing_mode == "hard":
            # Gumbel-Softmax (Hard=True):
            # Forward: Returns One-Hot vector (only 1.0 for the winner, 0.0 for others).
            # Backward: Gradients flow as if it were soft.
            gate_weights = F.gumbel_softmax(logits, tau=1.0, hard=True)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(batch))
            
        # stack: [N_atoms, Num_Experts, Hidden_Dim]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # combine:
        #  - if hard Routing: Multiplies by [0, 1, 0] -> Selects exactly Expert 2's output
        #  - if soft Routing: Multiplies by [0.1, 0.8, 0.1] -> Mixes them
        final_h = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)

        return final_h
