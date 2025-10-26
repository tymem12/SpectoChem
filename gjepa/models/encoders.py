from typing import Any

from torch import Tensor, nn
from torch_geometric.data import Data

from gjepa.utils import import_from_string


class GNNEncoder(nn.Module):
    """Encoder network to encode input graph into latent space"""

    def __init__(self, gnn_cls: str, **gnn_kwargs: Any) -> None:
        super().__init__()

        gnn = import_from_string(gnn_cls)
        self.gnn = gnn(
            **gnn_kwargs,
        )

    @property
    def out_channels(self) -> int:
        return self.gnn.out_channels

    def forward(
        self,        
        batch: Data
    ) -> Tensor:
        
        if self.gnn.handles_pos_encoding:
            return self.gnn(batch)
        else:
            return self.gnn(batch)
