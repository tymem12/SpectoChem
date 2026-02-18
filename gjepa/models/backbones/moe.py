import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

# Assuming SchNetEncoder is defined/imported as before
from gjepa.models.backbones.schnet import SchNetEncoder

class MoESchNetEncoder(nn.Module):
    def __init__(self,
                 num_experts: int = 3,
                 use_full_uma_for_router: bool = True,
                 router_hidden_dim: int = 64,
                 routing_mode: str = "soft",  # Options: "soft", "hard"
                 # SchNet Args passed to each expert
                 hidden_channels: int = 128,
                 **kwargs):
        """
        Args:
            routing_mode (str): 
                "soft": Weighted average of all experts (Softmax).
                "hard": Selects only the best expert per atom (Gumbel-Softmax).
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.use_full_uma_for_router = use_full_uma_for_router
        self.routing_mode = routing_mode.lower()
        
        if self.routing_mode not in ["soft", "hard"]:
            raise ValueError(f"routing_mode must be 'soft' or 'hard', got {self.routing_mode}")

        # 1. The Experts
        self.experts = nn.ModuleList([
            SchNetEncoder(hidden_channels=hidden_channels, **kwargs)
            for _ in range(num_experts)
        ])
        
        # 2. The Router
        uma_dim = hidden_channels 
        router_input_dim = 3 * uma_dim if use_full_uma_for_router else uma_dim

        # Note: We removed the final Softmax layer from the Sequential block
        # because we need raw logits for Gumbel-Softmax or custom Softmax logic.
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_experts)
        )

    def forward(self, batch: Data):
        # 1. Prepare Router Input
        uma_repr = batch.representation # [N, 9, 128]
        s = uma_repr[:, 0, :]
        
        if self.use_full_uma_for_router:
            v = torch.linalg.norm(uma_repr[:, 1:4, :], dim=1)
            t = torch.linalg.norm(uma_repr[:, 4:9, :], dim=1)
            router_input = torch.cat([s, v, t], dim=-1)
        else:
            router_input = s

        # 2. Compute Routing Logits (Raw scores, not probabilities yet)
        logits = self.router(router_input) # [N, Num_Experts]

        # 3. Determine Weights based on Mode
        if self.routing_mode == "soft":
            # Standard Softmax: Weights sum to 1, all experts contribute slightly
            gate_weights = F.softmax(logits, dim=-1)
            
        elif self.routing_mode == "hard":
            # Gumbel-Softmax (Hard=True):
            # Forward: Returns One-Hot vector (only 1.0 for the winner, 0.0 for others).
            # Backward: Gradients flow as if it were soft.
            gate_weights = F.gumbel_softmax(logits, tau=1.0, hard=True)

        # 4. Run Experts
        # (Note: In Graph MoEs, we typically run all experts and mask the output 
        # because splitting the graph for partial execution is extremely complex)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(batch))
            
        # Stack: [N_atoms, Num_Experts, Hidden_Dim]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # 5. Combine
        # If Hard Routing: Multiplies by [0, 1, 0] -> Selects exactly Expert 2's output
        # If Soft Routing: Multiplies by [0.1, 0.8, 0.1] -> Mixes them
        final_h = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)

        return final_h